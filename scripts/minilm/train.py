#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Train the all-MiniLM-L6-v2 ARDY root/body condition student.

The input directory is produced by ``cache_teacher.py``. Each shard contains
production LLM2Vec embeddings and the checkpoint-specific, bias-free
``[W_root @ e, W_body @ e]`` target. The resulting artifact is self-contained
for inference but remains specific to the ARDY checkpoint named in its
metadata.

Example:

    uv run python scripts/minilm/train.py \
      --cache-dir artifacts/teacher-core40-timeline \
      --output-dir artifacts/minilm-ardy-core40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ardy.minilm_teacher_cache import (
    load_teacher_cache,
    sha256_file,
    teacher_cache_fingerprint,
    validated_teacher_lineage,
)
from ardy.model.minilm_encoder import (
    ARDY_CONDITION_DIM,
    ARTIFACT_CONFIG,
    POOLING_MODES,
    MotionConditionStudent,
)
from ardy.model.registry import resolve_model_name

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BASE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_ARDY_MODEL = "ARDY-Core-RP-20FPS-Horizon40"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = frozenset((":16:8", ":4096:8"))
DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def training_runtime_versions() -> dict[str, str]:
    """Return the package versions that can affect student training."""
    versions = {
        package: version(package)
        for package in (
            "numpy",
            "safetensors",
            "tokenizers",
            "torch",
            "transformers",
        )
    }
    versions.update(
        {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cuda_runtime": torch.version.cuda or "none",
            "cudnn": str(torch.backends.cudnn.version() or "none"),
        }
    )
    return versions


def _saved_artifact_identity(output_dir: Path) -> tuple[str, int]:
    """Read the immutable fingerprint and payload size from a saved artifact."""

    config_path = output_dir / ARTIFACT_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"saved artifact config is invalid JSON: {config_path}") from error
    if not isinstance(config, dict):
        raise TypeError(f"saved artifact config must be an object: {config_path}")
    fingerprint = config.get("artifact_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("saved artifact config has no valid artifact_fingerprint")
    manifest = config.get("artifact_files")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("saved artifact config has no artifact_files manifest")
    payload_size = 0
    for relative_path, entry in manifest.items():
        if not isinstance(relative_path, str) or not isinstance(entry, dict):
            raise TypeError("saved artifact artifact_files is invalid")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"saved artifact size_bytes is invalid for {relative_path!r}"
            )
        payload_size += size
    return fingerprint, payload_size


@dataclass
class CachedExamples:
    texts: list[str]
    splits: list[str]
    targets: torch.Tensor


class ConditionDataset(Dataset):
    def __init__(self, examples: CachedExamples, split: str) -> None:
        indices = [index for index, value in enumerate(examples.splits) if value == split]
        if not indices:
            raise ValueError(f"No {split!r} examples found in teacher cache")
        self.texts = [examples.texts[index] for index in indices]
        self.targets = examples.targets[indices].float()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.targets[index]


class TokenizingCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        texts, targets = zip(*rows)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded, torch.stack(targets), list(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Teacher-cache directory")
    parser.add_argument("--output-dir", default="artifacts/minilm-ardy-core40")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_MODEL_REVISION,
        help="immutable all-MiniLM-L6-v2 commit",
    )
    parser.add_argument("--ardy-model", default=DEFAULT_ARDY_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=768)
    parser.add_argument(
        "--pooling-mode",
        choices=POOLING_MODES,
        default="mean",
        help="Sentence-level pooling over MiniLM token outputs",
    )
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--lr-schedule-epochs",
        type=int,
        default=None,
        help=(
            "cosine-schedule horizon in epochs; defaults to --epochs. Set the "
            "same horizon for shorter/longer runs to compare one shared "
            "training trajectory"
        ),
    )
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--relational-weight", type=float, default=0.02)
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--runtime-max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--device",
        default="auto",
        help="training device; auto selects CUDA when available, otherwise CPU",
    )
    parser.add_argument("--no-bf16", action="store_true", help="Disable CUDA BF16 autocast")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    workspace_config = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        DEFAULT_CUBLAS_WORKSPACE_CONFIG,
    )
    if workspace_config not in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be ':16:8' or ':4096:8' for "
            f"deterministic training, got {workspace_config!r}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_fresh_output_dir(output_dir: str | Path) -> Path:
    """Reject an existing output before any expensive training starts."""

    output_path = Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"--output-dir already exists: {output_path}. "
            "Use a new path so stale or partial artifact files cannot be mixed "
            "with this run."
        )
    return output_path


def cosine_lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    schedule_steps: int,
) -> float:
    """Return the deterministic warmup/cosine multiplier for one update."""

    if warmup_steps and step < warmup_steps:
        return max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(
        1,
        schedule_steps - warmup_steps,
    )
    return 0.5 * (
        1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
    )


def resolve_training_device(requested: str) -> torch.device:
    value = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if value == "auto":
        value = "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def cuda_supports_bf16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_bf16_supported()


def training_device_details(
    requested: str,
    resolved: torch.device,
) -> dict[str, object]:
    details: dict[str, object] = {
        "requested": requested,
        "resolved": str(resolved),
        "type": resolved.type,
    }
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        details.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return details


def load_cached_examples(
    cache_dir: str | Path,
) -> tuple[CachedExamples, dict, dict, str]:
    cache = load_teacher_cache(
        cache_dir,
        keep_teacher_embeddings=False,
        keep_targets=True,
    )
    targets = [shard.targets for shard in cache.shards]
    if any(target is None for target in targets):
        raise RuntimeError("strict teacher-cache loader did not retain targets")
    examples = CachedExamples(
        texts=[text for shard in cache.shards for text in shard.texts],
        splits=[split for shard in cache.shards for split in shard.splits],
        targets=torch.cat([target for target in targets if target is not None]),
    )
    if not torch.isfinite(examples.targets).all():
        raise ValueError("Teacher cache targets contain non-finite values")
    lineage = validated_teacher_lineage(cache.metadata)
    fingerprint = teacher_cache_fingerprint(cache)
    return examples, cache.metadata, lineage, fingerprint


def validate_training_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "epochs",
        "batch_size",
        "adapter_dim",
        "train_max_length",
        "runtime_max_length",
    )
    for field in positive_int_fields:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be a positive integer")
    if (
        isinstance(args.head_warmup_epochs, bool)
        or not isinstance(args.head_warmup_epochs, int)
        or not 0 <= args.head_warmup_epochs <= args.epochs
    ):
        raise ValueError("--head-warmup-epochs must be between zero and --epochs")
    lr_schedule_epochs = getattr(args, "lr_schedule_epochs", None)
    if lr_schedule_epochs is not None and (
        isinstance(lr_schedule_epochs, bool)
        or not isinstance(lr_schedule_epochs, int)
        or lr_schedule_epochs < args.epochs
    ):
        raise ValueError(
            "--lr-schedule-epochs must be at least --epochs when specified"
        )
    if isinstance(args.num_workers, bool) or not isinstance(args.num_workers, int) or args.num_workers < 0:
        raise ValueError("--num-workers must be a non-negative integer")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or not 0 <= args.seed < 2**32
    ):
        raise ValueError("--seed must be an integer between 0 and 4294967295")

    positive_float_fields = ("backbone_lr", "head_lr")
    for field in positive_float_fields:
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be finite and positive")
    nonnegative_float_fields = (
        "weight_decay",
        "cosine_weight",
        "relational_weight",
    )
    for field in nonnegative_float_fields:
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.warmup_ratio) or not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be finite and between zero and one")
    if not isinstance(args.base_model, str) or not args.base_model.strip():
        raise ValueError("--base-model must be a non-empty string")
    if (
        not isinstance(args.base_model_revision, str)
        or len(args.base_model_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in args.base_model_revision
        )
    ):
        raise ValueError(
            "--base-model-revision must be a 40-character lowercase commit SHA"
        )
    if not isinstance(args.ardy_model, str) or not args.ardy_model.strip():
        raise ValueError("--ardy-model must be a non-empty string")
    for field in ("cache_dir", "output_dir"):
        value = getattr(args, field)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"--{field.replace('_', '-')} must be a non-empty path")
    if args.pooling_mode not in POOLING_MODES:
        raise ValueError(f"--pooling-mode must be one of {POOLING_MODES}")
    if args.device != "auto":
        torch.device(args.device)


def resolve_and_validate_teacher_checkpoint(
    ardy_model: str,
    teacher_metadata: dict,
) -> tuple[str, Path]:
    resolved_model = resolve_model_name(ardy_model)
    checkpoint_value = teacher_metadata.get("checkpoint_path")
    checkpoint_hash = teacher_metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Teacher metadata checkpoint_path must be a non-empty string")
    checkpoint_path = Path(checkpoint_value).resolve()
    if checkpoint_path.parent.name != resolved_model:
        raise ValueError(
            f"--ardy-model resolves to {resolved_model!r}, but teacher targets were "
            f"projected with checkpoint folder {checkpoint_path.parent.name!r}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_path}")
    actual_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != actual_hash:
        raise ValueError(
            "Teacher metadata checkpoint SHA-256 does not match the checkpoint file: "
            f"metadata={checkpoint_hash}, actual={actual_hash}"
        )
    return resolved_model, checkpoint_path


def make_loader(
    dataset: ConditionDataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TokenizingCollator(tokenizer, max_length),
        generator=generator,
    )


def move_tokens(tokens: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def relational_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    prediction = F.normalize(prediction, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return F.smooth_l1_loss(prediction @ prediction.T, target @ target.T)


def distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_scale: torch.Tensor,
    cosine_weight: float,
    relational_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    expected_dim = 2 * ARDY_CONDITION_DIM
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != expected_dim:
        raise ValueError(
            f"distillation tensors must share shape [N, {expected_dim}], got "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    if target_scale.shape != (expected_dim,):
        raise ValueError(f"target_scale must have shape [{expected_dim}], got {tuple(target_scale.shape)}")
    scaled_error = (prediction - target) / target_scale
    regression = F.smooth_l1_loss(
        scaled_error,
        torch.zeros_like(scaled_error),
        beta=0.5,
    )
    root_cosine = (
        1.0
        - F.cosine_similarity(
            prediction[:, :ARDY_CONDITION_DIM],
            target[:, :ARDY_CONDITION_DIM],
            dim=-1,
        ).mean()
    )
    body_cosine = (
        1.0
        - F.cosine_similarity(
            prediction[:, ARDY_CONDITION_DIM:],
            target[:, ARDY_CONDITION_DIM:],
            dim=-1,
        ).mean()
    )
    cosine = 0.5 * (root_cosine + body_cosine)
    relation = relational_loss(prediction, target)
    total = regression + cosine_weight * cosine + relational_weight * relation
    return total, {
        "loss": float(total.detach()),
        "regression": float(regression.detach()),
        "cosine_loss": float(cosine.detach()),
        "root_cosine_loss": float(root_cosine.detach()),
        "body_cosine_loss": float(body_cosine.detach()),
        "relational_loss": float(relation.detach()),
    }


def condition_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, part in (
        ("root", slice(0, ARDY_CONDITION_DIM)),
        ("body", slice(ARDY_CONDITION_DIM, 2 * ARDY_CONDITION_DIM)),
    ):
        pred_part = prediction[:, part]
        target_part = target[:, part]
        result[f"{name}_cosine"] = float(F.cosine_similarity(pred_part, target_part, dim=-1).mean())
        result[f"{name}_rmse"] = float(torch.sqrt(F.mse_loss(pred_part, target_part)))
        result[f"{name}_mae"] = float(F.l1_loss(pred_part, target_part))
        target_rms = torch.sqrt(torch.mean(target_part.square())).clamp_min(1e-9)
        result[f"{name}_nrmse"] = result[f"{name}_rmse"] / float(target_rms)
    return result


@torch.inference_mode()
def evaluate(
    model: MotionConditionStudent,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for tokens, target, _texts in loader:
        tokens = move_tokens(tokens, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = model(**tokens)
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    return condition_metrics(torch.cat(predictions), torch.cat(targets))


def average_logs(logs: Iterable[dict[str, float]]) -> dict[str, float]:
    logs = list(logs)
    return {key: sum(item[key] for item in logs) / len(logs) for key in logs[0]}


def train(args: argparse.Namespace) -> dict:
    validate_training_args(args)
    output_dir = require_fresh_output_dir(args.output_dir)
    seed_everything(args.seed)
    device = resolve_training_device(args.device)
    use_bf16 = (
        device.type == "cuda"
        and not args.no_bf16
        and cuda_supports_bf16(device)
    )
    device_details = training_device_details(args.device, device)
    runtime_versions = training_runtime_versions()

    (
        examples,
        teacher_metadata,
        teacher_lineage,
        teacher_fingerprint,
    ) = load_cached_examples(args.cache_dir)
    resolved_ardy_model, teacher_checkpoint = resolve_and_validate_teacher_checkpoint(
        args.ardy_model,
        teacher_metadata,
    )
    split_counts = Counter(examples.splits)
    train_dataset = ConditionDataset(examples, "train")
    val_dataset = ConditionDataset(examples, "val")
    print(
        f"Loaded teacher cache: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={split_counts['test']} (held out)",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_model_revision,
    )
    model = MotionConditionStudent.from_base_model(
        args.base_model,
        revision=args.base_model_revision,
        adapter_dim=args.adapter_dim,
        condition_dim=ARDY_CONDITION_DIM,
        normalize_embedding=True,
        pooling_mode=args.pooling_mode,
    ).to(device)

    train_loader = make_loader(
        train_dataset,
        tokenizer,
        args.batch_size,
        args.train_max_length,
        True,
        args.num_workers,
        args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        tokenizer,
        args.batch_size,
        args.train_max_length,
        False,
        args.num_workers,
        args.seed,
    )

    target_std = train_dataset.targets.std(dim=0, unbiased=False)
    if not torch.isfinite(target_std).all():
        raise ValueError("Training target scale contains non-finite values")
    std_floor = float(target_std.median()) * 0.05
    target_scale = target_std.clamp_min(max(std_floor, 1e-6)).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = [
        *model.adapter.parameters(),
        *model.root_head.parameters(),
        *model.body_head.parameters(),
    ]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_updates = max(1, args.epochs * len(train_loader))
    lr_schedule_epochs = getattr(args, "lr_schedule_epochs", None) or args.epochs
    lr_schedule_steps = max(1, lr_schedule_epochs * len(train_loader))
    warmup_steps = int(lr_schedule_steps * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr_multiplier(
            step,
            warmup_steps=warmup_steps,
            schedule_steps=lr_schedule_steps,
        ),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    history = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, float] | None = None
    started = time.perf_counter()

    for epoch in range(args.epochs):
        backbone_trainable = epoch >= args.head_warmup_epochs
        for parameter in backbone_params:
            parameter.requires_grad = backbone_trainable

        model.train()
        epoch_logs = []
        epoch_start = time.perf_counter()
        for step, (tokens, target, _texts) in enumerate(train_loader, start=1):
            tokens = move_tokens(tokens, device)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                prediction = model(**tokens)
                loss, log = distillation_loss(
                    prediction.float(),
                    target.float(),
                    target_scale,
                    args.cosine_weight,
                    args.relational_weight,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_logs.append(log)
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} loss={log['loss']:.5f}",
                    flush=True,
                )

        train_log = average_logs(epoch_logs)
        val_metrics = evaluate(model, val_loader, device, use_bf16)
        score = 0.5 * (val_metrics["root_cosine"] + val_metrics["body_cosine"])
        epoch_result = {
            "epoch": epoch + 1,
            "backbone_trainable": backbone_trainable,
            "seconds": time.perf_counter() - epoch_start,
            "train": train_log,
            "validation": val_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation_metrics = dict(val_metrics)

    assert best_state is not None
    assert best_validation_metrics is not None
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = best_validation_metrics
    elapsed = time.perf_counter() - started

    metadata = {
        "target_definition": "[W_root @ LLM2Vec(prompt), W_body @ LLM2Vec(prompt)] (bias excluded)",
        "teacher_cache": Path(args.cache_dir).name,
        "teacher_cache_lineage": teacher_lineage,
        "teacher_cache_fingerprint": teacher_fingerprint,
        "teacher_checkpoint": teacher_checkpoint.name,
        "training": {
            "seed": args.seed,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "test_examples": split_counts["test"],
            "epochs": args.epochs,
            "head_warmup_epochs": args.head_warmup_epochs,
            "batch_size": args.batch_size,
            "adapter_dim": args.adapter_dim,
            "pooling_mode": args.pooling_mode,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_schedule_epochs": lr_schedule_epochs,
            "lr_schedule_steps": lr_schedule_steps,
            "warmup_steps": warmup_steps,
            "cosine_weight": args.cosine_weight,
            "relational_weight": args.relational_weight,
            "train_max_length": args.train_max_length,
            "runtime_max_length": args.runtime_max_length,
            "num_workers": args.num_workers,
            "device": str(device),
            "device_details": device_details,
            "bf16_autocast": use_bf16,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "train_batches_per_epoch": len(train_loader),
            "optimizer_updates": optimizer_updates,
            "base_model": args.base_model,
            "base_model_revision": args.base_model_revision,
            "ardy_model": resolved_ardy_model,
            "selected_epoch": best_epoch,
            "best_validation_score": best_score,
            "runtime_versions": runtime_versions,
        },
        "selection": {
            "split": "val",
            "metric": "mean(root_cosine, body_cosine)",
            "test_evaluated": False,
        },
        "validation_metrics": validation_metrics,
        "provenance_notice": (
            "Derived using ARDY checkpoint projections and LLM2Vec teacher outputs. "
            "Review NVIDIA Open Model, Meta Llama, LLM2Vec, and source-corpus licenses "
            "before redistribution."
        ),
    }
    artifact_path = model.save_artifact(
        output_dir,
        tokenizer,
        metadata,
        base_model_name_or_path=args.base_model,
        compatible_ardy_models=[resolved_ardy_model],
        max_length=args.runtime_max_length,
    )
    artifact_fingerprint, artifact_payload_size_bytes = (
        _saved_artifact_identity(Path(artifact_path))
    )
    training_report = {
        "schema_version": 1,
        "artifact": output_dir.name,
        "artifact_fingerprint": artifact_fingerprint,
        "artifact_payload_size_bytes": artifact_payload_size_bytes,
        "base_model": args.base_model,
        "ardy_model": resolved_ardy_model,
        "configuration": metadata["training"],
        "best_validation_score": best_score,
        "best_epoch": best_epoch,
        "teacher_cache_fingerprint": teacher_fingerprint,
        "runtime_versions": runtime_versions,
        "selection": {
            "split": "val",
            "metric": "mean(root_cosine, body_cosine)",
            "test_evaluated": False,
        },
        "validation": validation_metrics,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(training_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(training_report, indent=2, sort_keys=True), flush=True)
    return training_report


if __name__ == "__main__":
    train(parse_args())
