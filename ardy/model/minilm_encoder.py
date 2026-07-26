# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""MiniLM text encoder distilled for ARDY root/body conditioning.

The production LLM2Vec encoder returns one 4096-dimensional token.  Each ARDY
denoiser then applies a checkpoint-specific 4096 -> 1024 projection for its
root and body stages.  The student implemented here predicts the *bias-free*
outputs of those two projections directly and concatenates them:

    [root W @ e, body W @ e] -> [B, 1, 2048]

``DualConditionTextProjection`` in :mod:`ardy.model.backbone` selects the
appropriate half and adds the original checkpoint bias.  Keeping the bias in
the denoiser is important: ARDY's classifier-free guidance creates its
unconditional text input by zeroing the encoder output, and the legacy Linear
therefore still contributes its bias.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

ARTIFACT_CONFIG = "ardy_minilm_config.json"
HEADS_FILE = "condition_heads.safetensors"
BACKBONE_DIR = "backbone"
ARTIFACT_FORMAT_VERSION = 2
LEGACY_ARTIFACT_FORMAT_VERSION = 1
ARDY_CONDITION_DIM = 1024
POOLING_MODES = ("mean", "mean_cls_max_std")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"MiniLM artifact {field} must be a positive integer, got {value!r}")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MiniLM artifact {field} must be a non-empty string")
    return value


def _validate_compatible_models(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("MiniLM artifact compatible_ardy_models must be a non-empty list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("MiniLM artifact compatible_ardy_models must not contain duplicates")
    return value


def _pooled_width(hidden_dim: int, pooling_mode: str) -> int:
    if pooling_mode not in POOLING_MODES:
        raise ValueError(f"Unsupported pooling mode {pooling_mode!r}; expected one of {POOLING_MODES}.")
    return hidden_dim * (4 if pooling_mode == "mean_cls_max_std" else 1)


def _validate_artifact_config(config: Mapping[str, Any]) -> None:
    format_version = config.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in (LEGACY_ARTIFACT_FORMAT_VERSION, ARTIFACT_FORMAT_VERSION)
    ):
        raise ValueError(
            f"Unsupported MiniLM artifact format {format_version!r}; expected "
            f"{LEGACY_ARTIFACT_FORMAT_VERSION} or {ARTIFACT_FORMAT_VERSION}."
        )

    _nonempty_string(config.get("base_model"), "base_model")
    hidden_dim = _positive_int(config.get("hidden_dim"), "hidden_dim")
    _positive_int(config.get("adapter_dim"), "adapter_dim")
    condition_dim = _positive_int(config.get("condition_dim"), "condition_dim")
    if condition_dim != ARDY_CONDITION_DIM:
        raise ValueError(f"MiniLM artifact condition_dim must be {ARDY_CONDITION_DIM}, got {condition_dim}")
    output_dim = _positive_int(config.get("output_dim"), "output_dim")
    if output_dim != 2 * condition_dim:
        raise ValueError(
            f"MiniLM artifact output_dim must equal 2 * condition_dim ({2 * condition_dim}), got {output_dim}"
        )

    pooling_mode = config.get("pooling_mode", "mean")
    if not isinstance(pooling_mode, str):
        raise TypeError("MiniLM artifact pooling_mode must be a string")
    expected_pooled_dim = _pooled_width(hidden_dim, pooling_mode)
    pooled_dim = config.get("pooled_dim", expected_pooled_dim)
    if _positive_int(pooled_dim, "pooled_dim") != expected_pooled_dim:
        raise ValueError(
            f"MiniLM artifact pooled_dim is {pooled_dim}, but pooling mode "
            f"{pooling_mode!r} with hidden_dim={hidden_dim} requires "
            f"{expected_pooled_dim}"
        )

    normalize_embedding = config.get("normalize_embedding", True)
    if not isinstance(normalize_embedding, bool):
        raise TypeError("MiniLM artifact normalize_embedding must be a boolean")
    _validate_compatible_models(config.get("compatible_ardy_models"))
    _positive_int(config.get("max_length"), "max_length")
    if format_version == ARTIFACT_FORMAT_VERSION:
        if not isinstance(config.get("metadata"), dict):
            raise ValueError("MiniLM artifact metadata must be a JSON object")
        if not isinstance(config.get("artifact_files"), dict):
            raise ValueError("MiniLM artifact artifact_files must be a JSON object")


def _artifact_payload_paths(artifact_path: Path) -> dict[str, Path]:
    heads_path = artifact_path / HEADS_FILE
    if not heads_path.is_file():
        raise FileNotFoundError(f"MiniLM artifact condition heads not found: {heads_path}")
    backbone_path = artifact_path / BACKBONE_DIR
    if not backbone_path.is_dir():
        raise FileNotFoundError(f"MiniLM artifact backbone directory not found: {backbone_path}")

    paths = {HEADS_FILE: heads_path}
    for path in sorted(backbone_path.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"MiniLM artifact payload must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(artifact_path).as_posix()
            paths[relative] = path
    if len(paths) == 1:
        raise ValueError(f"MiniLM artifact backbone directory is empty: {backbone_path}")
    return paths


def _build_file_manifest(artifact_path: Path) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for relative, path in sorted(_artifact_payload_paths(artifact_path).items())
    }


def _validate_v2_integrity(artifact_path: Path, config: Mapping[str, Any]) -> None:
    manifest = config["artifact_files"]
    payload_paths = _artifact_payload_paths(artifact_path)
    if set(manifest) != set(payload_paths):
        missing = sorted(set(manifest) - set(payload_paths))
        untracked = sorted(set(payload_paths) - set(manifest))
        raise ValueError(
            f"MiniLM artifact file manifest does not match payload files: missing={missing}, untracked={untracked}"
        )

    for relative, path in sorted(payload_paths.items()):
        entry = manifest[relative]
        if not isinstance(entry, dict):
            raise TypeError(f"MiniLM artifact manifest entry {relative!r} must be an object")
        if set(entry) != {"sha256", "size_bytes"}:
            raise ValueError(
                f"MiniLM artifact manifest entry {relative!r} must contain exactly 'sha256' and 'size_bytes'"
            )
        expected_hash = entry["sha256"]
        expected_size = entry["size_bytes"]
        if not isinstance(expected_hash, str) or _SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError(f"MiniLM artifact manifest hash for {relative!r} is invalid")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"MiniLM artifact manifest size for {relative!r} is invalid")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"MiniLM artifact size mismatch for {relative!r}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"MiniLM artifact SHA-256 mismatch for {relative!r}: expected {expected_hash}, got {actual_hash}"
            )

    fingerprint = config.get("artifact_fingerprint")
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("MiniLM artifact artifact_fingerprint must be a SHA-256 hex digest")
    unsigned_config = dict(config)
    unsigned_config.pop("artifact_fingerprint", None)
    expected_fingerprint = hashlib.sha256(_canonical_json_bytes(unsigned_config)).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("MiniLM artifact fingerprint mismatch: configuration or file manifest has been modified")


def _validate_v1_integrity(artifact_path: Path, config: Mapping[str, Any]) -> None:
    """Validate the heads-only fingerprint emitted by the legacy v1 writer."""
    fingerprint = config.get("artifact_fingerprint")
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("Legacy MiniLM artifact artifact_fingerprint is invalid")
    heads_path = artifact_path / HEADS_FILE
    if not heads_path.is_file():
        raise FileNotFoundError(f"MiniLM artifact condition heads not found: {heads_path}")
    unsigned_config = dict(config)
    unsigned_config.pop("artifact_fingerprint", None)
    config_bytes = json.dumps(unsigned_config, indent=2, sort_keys=True).encode("utf-8")
    heads_hash = _sha256_file(heads_path)
    expected_fingerprint = hashlib.sha256(config_bytes + heads_hash.encode("ascii")).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError(
            "Legacy MiniLM artifact fingerprint mismatch: configuration or condition heads have been modified"
        )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the all-MiniLM attention mask."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def pool_token_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_mode: str,
) -> torch.Tensor:
    """Pool contextual token outputs into one sentence-level feature vector."""
    mean = mean_pool(last_hidden_state, attention_mask)
    if pooling_mode == "mean":
        return mean
    if pooling_mode != "mean_cls_max_std":
        raise ValueError(f"Unsupported pooling mode {pooling_mode!r}; expected one of {POOLING_MODES}.")

    mask = attention_mask.unsqueeze(-1)
    valid_tokens = mask.bool()
    cls = last_hidden_state[:, 0]
    maximum = last_hidden_state.masked_fill(
        ~valid_tokens,
        torch.finfo(last_hidden_state.dtype).min,
    ).amax(dim=1)
    has_valid_token = attention_mask.bool().any(dim=1, keepdim=True)
    maximum = torch.where(has_valid_token, maximum, torch.zeros_like(maximum))

    float_mask = mask.to(last_hidden_state.dtype)
    centered = last_hidden_state - mean.unsqueeze(1)
    variance = (centered.square() * float_mask).sum(dim=1)
    variance = variance / float_mask.sum(dim=1).clamp_min(1e-9)
    standard_deviation = variance.clamp_min(1e-12).sqrt()
    return torch.cat((mean, cls, maximum, standard_deviation), dim=-1)


class MotionConditionStudent(nn.Module):
    """Fine-tunable MiniLM backbone with shared adapter and root/body heads."""

    def __init__(
        self,
        backbone: nn.Module,
        adapter_dim: int = 768,
        condition_dim: int = 1024,
        normalize_embedding: bool = True,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        if pooling_mode not in POOLING_MODES:
            raise ValueError(f"Unsupported pooling mode {pooling_mode!r}; expected one of {POOLING_MODES}.")
        self.backbone = backbone
        self.hidden_dim = int(backbone.config.hidden_size)
        self.adapter_dim = int(adapter_dim)
        self.condition_dim = int(condition_dim)
        self.normalize_embedding = bool(normalize_embedding)
        self.pooling_mode = pooling_mode
        self.pooled_dim = _pooled_width(self.hidden_dim, pooling_mode)
        self.base_model_name_or_path: str | None = None

        self.adapter = nn.Sequential(
            nn.LayerNorm(self.pooled_dim),
            nn.Linear(self.pooled_dim, self.adapter_dim),
            nn.GELU(),
            nn.LayerNorm(self.adapter_dim),
        )
        self.root_head = nn.Linear(self.adapter_dim, self.condition_dim)
        self.body_head = nn.Linear(self.adapter_dim, self.condition_dim)

    @classmethod
    def from_base_model(
        cls,
        base_model_name_or_path: str,
        adapter_dim: int = 768,
        condition_dim: int = 1024,
        normalize_embedding: bool = True,
        pooling_mode: str = "mean",
    ) -> MotionConditionStudent:
        backbone = AutoModel.from_pretrained(base_model_name_or_path)
        model = cls(
            backbone=backbone,
            adapter_dim=adapter_dim,
            condition_dim=condition_dim,
            normalize_embedding=normalize_embedding,
            pooling_mode=pooling_mode,
        )
        model.base_model_name_or_path = str(base_model_name_or_path)
        return model

    def sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **tokenizer_outputs: torch.Tensor,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **tokenizer_outputs,
        }
        output = self.backbone(**model_inputs)
        pooled = pool_token_embeddings(
            output.last_hidden_state,
            attention_mask,
            self.pooling_mode,
        )
        if self.normalize_embedding:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **tokenizer_outputs: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **tokenizer_outputs,
        )
        shared = self.adapter(pooled)
        return torch.cat((self.root_head(shared), self.body_head(shared)), dim=-1)

    def save_artifact(
        self,
        output_dir: str | Path,
        tokenizer,
        metadata: dict[str, Any],
        *,
        base_model_name_or_path: str | None = None,
        compatible_ardy_models: list[str],
        max_length: int,
    ) -> Path:
        """Save a self-contained, integrity-checked inference artifact."""
        if not isinstance(metadata, dict):
            raise TypeError("artifact metadata must be a dictionary")
        base_model = _nonempty_string(
            base_model_name_or_path or self.base_model_name_or_path,
            "base_model",
        )
        compatible_models = _validate_compatible_models(compatible_ardy_models)
        max_length = _positive_int(max_length, "max_length")
        if self.condition_dim != ARDY_CONDITION_DIM:
            raise ValueError(
                f"ARDY MiniLM artifacts require condition_dim={ARDY_CONDITION_DIM}, got {self.condition_dim}"
            )

        output_path = Path(output_dir)
        backbone_path = output_path / BACKBONE_DIR
        output_path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(backbone_path, safe_serialization=True)
        tokenizer.save_pretrained(backbone_path)

        head_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }
        save_file(head_state, output_path / HEADS_FILE)

        config = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": base_model,
            "hidden_dim": self.hidden_dim,
            "pooled_dim": self.pooled_dim,
            "pooling_mode": self.pooling_mode,
            "adapter_dim": self.adapter_dim,
            "condition_dim": self.condition_dim,
            "output_dim": self.condition_dim * 2,
            "normalize_embedding": self.normalize_embedding,
            "compatible_ardy_models": compatible_models,
            "max_length": max_length,
            "metadata": metadata,
            "artifact_files": _build_file_manifest(output_path),
        }
        _validate_artifact_config(config)
        config["artifact_fingerprint"] = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
        config_path = output_path / ARTIFACT_CONFIG
        temporary_config_path = config_path.with_suffix(f"{config_path.suffix}.tmp")
        temporary_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_config_path.replace(config_path)
        _validate_v2_integrity(output_path, config)
        return output_path

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
    ) -> tuple[MotionConditionStudent, Any, dict[str, Any]]:
        artifact_path = Path(artifact_dir)
        config_path = artifact_path / ARTIFACT_CONFIG
        if not config_path.is_file():
            raise FileNotFoundError(
                f"MiniLM ARDY artifact config not found: {config_path}. "
                "Train an artifact first or set MINILM_TEXT_ENCODER_PATH."
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"MiniLM artifact config is invalid JSON: {config_path}") from error
        if not isinstance(config, dict):
            raise TypeError(f"MiniLM artifact config must contain a JSON object: {config_path}")
        _validate_artifact_config(config)
        if config["format_version"] == ARTIFACT_FORMAT_VERSION:
            _validate_v2_integrity(artifact_path, config)
        else:
            _validate_v1_integrity(artifact_path, config)

        backbone_path = artifact_path / BACKBONE_DIR
        backbone = AutoModel.from_pretrained(backbone_path)
        tokenizer = AutoTokenizer.from_pretrained(backbone_path)
        model = cls(
            backbone=backbone,
            adapter_dim=int(config["adapter_dim"]),
            condition_dim=int(config["condition_dim"]),
            normalize_embedding=bool(config.get("normalize_embedding", True)),
            pooling_mode=str(config.get("pooling_mode", "mean")),
        )
        model.base_model_name_or_path = str(config["base_model"])
        if model.hidden_dim != int(config["hidden_dim"]):
            raise ValueError(
                f"MiniLM artifact hidden_dim={config['hidden_dim']} does not match "
                f"the saved backbone hidden size {model.hidden_dim}"
            )
        incompatible = model.load_state_dict(
            load_file(artifact_path / HEADS_FILE),
            strict=False,
        )
        missing_non_backbone = [key for key in incompatible.missing_keys if not key.startswith("backbone.")]
        if missing_non_backbone or incompatible.unexpected_keys:
            raise RuntimeError(
                "Invalid MiniLM condition-head state: "
                f"missing={missing_non_backbone}, unexpected={incompatible.unexpected_keys}"
            )
        return model, tokenizer, config


class MiniLMArdyEncoder(nn.Module):
    """Inference wrapper matching ARDY's ``(tensor, lengths)`` encoder contract."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        expected_ardy_model: str | None = None,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        model, tokenizer, artifact_config = MotionConditionStudent.from_artifact(model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.artifact_config = artifact_config
        self.output_dim = int(artifact_config["output_dim"])
        configured_max_length = artifact_config.get("max_length", 256)
        self.max_length = _positive_int(
            configured_max_length if max_length is None else max_length,
            "max_length",
        )
        self.cache_namespace = (
            f"ardy-minilm-v{artifact_config['format_version']}:"
            f"{artifact_config.get('artifact_fingerprint', 'unknown')}:"
            f"{self.output_dim}"
        )

        if expected_ardy_model is not None:
            self.assert_compatible(expected_ardy_model)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = str(device)
        self.model.to(device=device, dtype=getattr(torch, dtype))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def assert_compatible(self, ardy_model_name: str) -> None:
        ardy_model_name = _nonempty_string(ardy_model_name, "expected_ardy_model")
        compatible = _validate_compatible_models(self.artifact_config.get("compatible_ardy_models"))
        if ardy_model_name not in compatible:
            raise ValueError(
                f"MiniLM artifact is trained for {compatible}, not {ardy_model_name!r}. "
                "Direct root/body heads are ARDY-checkpoint-specific."
            )

    def to(self, device=None, dtype=None, **kwargs):
        super().to(device=device, dtype=dtype, **kwargs)
        if device is not None:
            self._device = str(device)
        return self

    def get_device(self):
        return next(self.model.parameters()).device

    # ARDY mutates the returned feature tensor when constructing the
    # classifier-free unconditional condition. ``inference_mode`` tensors
    # reject that write once they leave this method, whereas ``no_grad``
    # preserves the same inference-only behavior without changing the tensor
    # contract expected by ``ArdyModel._encode_text``.
    @torch.no_grad()
    def forward(self, text: list[str] | str):
        is_string = isinstance(text, str)
        texts = [text] if is_string else list(text)
        if not texts:
            raise ValueError("text must contain at least one prompt")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.get_device()) for key, value in encoded.items()}
        conditions = self.model(**encoded)[:, None, :]
        lengths: list[int] | int = [1] * len(texts)
        if is_string:
            conditions = conditions[0]
            lengths = 1
        return conditions, lengths


__all__ = [
    "ARDY_CONDITION_DIM",
    "ARTIFACT_CONFIG",
    "ARTIFACT_FORMAT_VERSION",
    "BACKBONE_DIR",
    "HEADS_FILE",
    "LEGACY_ARTIFACT_FORMAT_VERSION",
    "POOLING_MODES",
    "MiniLMArdyEncoder",
    "MotionConditionStudent",
    "mean_pool",
    "pool_token_embeddings",
]
