#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Cache ARDY's production LLM2Vec teacher and bias-free Core40 targets.

The teacher is the production :class:`ardy.model.LLM2VecEncoder`, configured
with bfloat16 weights and its mandatory internal batch size of one.  For every
JSONL prompt this script saves:

* the raw 4096-dimensional LLM2Vec embedding; and
* ``concat(W_root @ e, W_body @ e)`` (2048 dimensions), with no biases.

Projection matrices are read directly from the Core40 ``denoiser.safetensors``.
Completed shards are atomic and validated on restart, so an interrupted run can
be resumed by invoking the same command again.

Example:

    uv run python scripts/minilm/cache_teacher.py \
        --input artifacts/minilm/prompts.jsonl \
        --output-dir artifacts/minilm/teacher \
        --checkpoint checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors

Each ``teacher-NNNNN.pt`` is a dictionary containing ``texts``, ``splits``,
``teacher_embeddings`` (float32 ``[N, 4096]``), and ``targets`` (float32
``[N, 2048]``).  ``metadata.json`` records identities, hashes, counts, dtypes,
elapsed wall time, and the ordered shard list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from torch import Tensor

from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

DEFAULT_INPUT = Path("artifacts/minilm/prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/minilm/teacher")
DEFAULT_CHECKPOINT = Path("checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors")
DEFAULT_BASE_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
DEFAULT_PEFT_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
DEFAULT_ROOT_KEY = "denoiser.backbone.root_model.embed_text.weight"
DEFAULT_BODY_KEY = "denoiser.backbone.body_model.embed_text.weight"
METADATA_FILENAME = "metadata.json"
SHARD_PATTERN = re.compile(r"^teacher-(\d{5})\.pt$")
FORMAT_VERSION = 1
TEACHER_DIM = 4096
BRANCH_DIM = 1024
TARGET_DIM = 2 * BRANCH_DIM
VALID_SPLITS = frozenset(("train", "val", "test"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the full SHA-256 of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _snapshot_revision(path: str | os.PathLike[str]) -> str | None:
    """Extract a Hugging Face snapshot commit from a cached/local path."""

    # Cache files are symlinks into ``blobs/``; inspect the non-resolved path
    # first or the ``snapshots/<commit>`` component is lost.
    candidates = (Path(path).absolute(), Path(path).resolve())
    for candidate in candidates:
        parts = candidate.parts
        try:
            snapshot_index = parts.index("snapshots")
        except ValueError:
            continue
        if snapshot_index + 1 < len(parts):
            return parts[snapshot_index + 1]
    return None


def _model_revision_hint(model_name_or_path: str, marker_filename: str) -> str | None:
    """Best-effort resolved revision for provenance (not a resume identity key)."""

    marker_path = _model_marker_path(model_name_or_path, marker_filename)
    if marker_path is not None:
        revision = _snapshot_revision(marker_path)
        if revision is not None:
            return revision
        resolved = marker_path.resolve()
        if marker_path.is_file():
            return f"local:{resolved}:{sha256_file(marker_path)}"
        return f"local:{resolved}"
    return None


def _model_marker_path(
    model_name_or_path: str,
    marker_filename: str,
) -> Path | None:
    """Locate a local or cached model metadata file without network access."""

    effective_name = model_name_or_path
    text_encoders_dir = os.environ.get("TEXT_ENCODERS_DIR")
    if text_encoders_dir:
        effective_name = str(Path(text_encoders_dir) / model_name_or_path)

    local_path = Path(effective_name)
    if local_path.exists():
        marker_path = local_path / marker_filename
        return marker_path if marker_path.is_file() else local_path

    cached = try_to_load_from_cache(
        repo_id=model_name_or_path,
        filename=marker_filename,
        cache_dir=os.environ.get("HUGGINGFACE_CACHE_DIR"),
    )
    if isinstance(cached, str):
        return Path(cached)
    return None


def _adapter_foundation_model(model_name_or_path: str) -> str | None:
    """Read the foundation model named by a PEFT adapter, when available."""

    adapter_config = _model_marker_path(model_name_or_path, "adapter_config.json")
    if adapter_config is None or not adapter_config.is_file():
        return None
    try:
        value = json.loads(adapter_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    foundation = value.get("base_model_name_or_path") if isinstance(value, dict) else None
    return foundation if isinstance(foundation, str) and foundation else None


def _runtime_provenance(
    *,
    requested_device: str,
    base_model: str,
    peft_model: str,
) -> dict[str, Any]:
    env_device = os.environ.get("TEXT_ENCODER_DEVICE")
    effective_device = env_device or requested_device
    if effective_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = str(torch.device(effective_device))
    device_details: dict[str, Any] = {
        "cli_requested": requested_device,
        "env_override": env_device,
        "requested": effective_device,
        "resolved": resolved_device,
    }
    device_type = torch.device(resolved_device).type
    if device_type == "cuda" and torch.cuda.is_available():
        device_index = torch.device(resolved_device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        device_details.update(
            {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "capability": list(torch.cuda.get_device_capability(device_index)),
            }
        )
    foundation_model = _adapter_foundation_model(base_model)
    return {
        "device": device_details,
        "model_revisions": {
            "foundation_model": foundation_model,
            "foundation": (
                _model_revision_hint(foundation_model, "config.json") if foundation_model is not None else None
            ),
            "base": _model_revision_hint(base_model, "config.json"),
            "peft": _model_revision_hint(peft_model, "adapter_config.json"),
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "safetensors": _package_version("safetensors"),
        },
    }


def _device_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Canonical fields that can affect bfloat16 teacher numerics."""

    resolved = value.get("resolved")
    try:
        parsed = torch.device(str(resolved))
        device_type = parsed.type
        index = value.get("index", parsed.index)
    except (RuntimeError, ValueError):
        device_type = resolved
        index = value.get("index")
    return (
        device_type,
        index,
        value.get("name"),
        tuple(value.get("capability", ())),
    )


def _select_resume_provenance(
    previous_metadata: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    has_shards: bool,
) -> dict[str, Any]:
    """Validate and preserve provenance once any teacher shard exists."""

    if previous_metadata is None or not has_shards:
        return dict(current)

    keys = ("device", "model_revisions", "versions")
    if not all(isinstance(previous_metadata.get(key), dict) for key in keys):
        # Version-1 caches created before provenance tracking remain usable.
        # Bind subsequent resumes to this first observed environment, while
        # disclosing that the attribution of pre-existing shards is inferred.
        migrated = dict(current)
        migrated["provenance_status"] = "migrated_from_legacy_unverified"
        return migrated

    previous_device = previous_metadata["device"]
    current_device = current["device"]
    if _device_signature(previous_device) != _device_signature(current_device):
        raise ValueError(
            f"teacher-cache device provenance mismatch: existing={previous_device!r}, current={current_device!r}"
        )

    previous_versions = previous_metadata["versions"]
    current_versions = current["versions"]
    version_mismatches = [
        name
        for name, old_value in previous_versions.items()
        if old_value not in (None, "unknown") and current_versions.get(name) not in (None, "unknown", old_value)
    ]
    if version_mismatches:
        raise ValueError(
            "teacher-cache runtime version mismatch for "
            f"{version_mismatches}: existing={previous_versions!r}, "
            f"current={current_versions!r}"
        )

    previous_revisions = previous_metadata["model_revisions"]
    current_revisions = current["model_revisions"]
    previous_foundation = previous_revisions.get("foundation_model")
    current_foundation = current_revisions.get("foundation_model")
    if previous_foundation is not None and current_foundation is not None and previous_foundation != current_foundation:
        raise ValueError(
            f"teacher-cache foundation model mismatch: existing={previous_foundation!r}, current={current_foundation!r}"
        )
    revision_mismatches = [
        name
        for name, old_value in previous_revisions.items()
        if name != "foundation_model" and old_value is not None and current_revisions.get(name) not in (None, old_value)
    ]
    if revision_mismatches:
        raise ValueError(
            "teacher-cache model revision mismatch for "
            f"{revision_mismatches}: existing={previous_revisions!r}, "
            f"current={current_revisions!r}"
        )

    # Preserve the provenance attributed when the first shard was written.
    # Newly introduced fields remain explicitly unknown for an older cache.
    preserved_revisions = dict(previous_revisions)
    preserved_revisions.setdefault(
        "foundation_model",
        current_revisions.get("foundation_model"),
    )
    preserved_revisions.setdefault("foundation", None)
    return {
        "device": dict(previous_device),
        "model_revisions": preserved_revisions,
        "versions": dict(previous_versions),
        "provenance_status": previous_metadata.get("provenance_status", "recorded"),
    }


def read_prompts(path: Path) -> tuple[list[str], list[str]]:
    """Read and validate the prompt JSONL while preserving its order."""

    texts: list[str] = []
    splits: list[str] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            text = record.get("text")
            split = record.get("split")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: 'text' must be a non-empty string")
            if split not in VALID_SPLITS:
                raise ValueError(f"{path}:{line_number}: 'split' must be one of {sorted(VALID_SPLITS)}")
            texts.append(text)
            splits.append(split)
    if not texts:
        raise ValueError(f"{path} contains no prompts")
    return texts, splits


def inspect_projection_shapes(
    checkpoint: Path,
    root_key: str,
    body_key: str,
) -> None:
    """Validate projection keys and shapes without materialising the tensors."""

    expected_shape = [BRANCH_DIM, TEACHER_DIM]
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        available = set(tensors.keys())
        for key in (root_key, body_key):
            if key not in available:
                raise KeyError(f"projection key {key!r} is absent from {checkpoint}")
            shape = list(tensors.get_slice(key).get_shape())
            if shape != expected_shape:
                raise ValueError(f"{key!r} has shape {shape}; expected {expected_shape}")


def load_projection_weights(
    checkpoint: Path,
    root_key: str,
    body_key: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Load only the two required safetensors matrices as float32."""

    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        root_weight = tensors.get_tensor(root_key)
        body_weight = tensors.get_tensor(body_key)
    return (
        root_weight.to(device=device, dtype=torch.float32),
        body_weight.to(device=device, dtype=torch.float32),
    )


def _metadata_identity(
    *,
    input_path: Path,
    input_sha256: str,
    count: int,
    split_counts: Mapping[str, int],
    base_model: str,
    peft_model: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    root_key: str,
    body_key: str,
    shard_size: int,
) -> dict[str, Any]:
    model_name = f"{base_model} + {peft_model}"
    return {
        "format_version": FORMAT_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "model_name": model_name,
        "base_model_name_or_path": base_model,
        "peft_model_name_or_path": peft_model,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "target_keys": [root_key, body_key],
        "target_order": ["root", "body"],
        "bias_applied": False,
        "count": count,
        "split_counts": dict(split_counts),
        "shard_size": shard_size,
        "dtype": {
            "teacher_model": "bfloat16",
            "teacher_embeddings": "float32",
            "projection_weights": "float32",
            "targets": "float32",
        },
        "teacher_batch_size": 1,
        "teacher_dim": TEACHER_DIM,
        "target_dim": TARGET_DIM,
    }


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return metadata


def _check_metadata_identity(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mismatches = [key for key, expected_value in expected.items() if metadata.get(key) != expected_value]
    if mismatches:
        details = ", ".join(f"{key}={metadata.get(key)!r} (expected {expected[key]!r})" for key in mismatches)
        raise ValueError(
            "existing teacher cache does not match this invocation; use a new "
            f"--output-dir. Mismatched metadata: {details}"
        )


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(value), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with torch versions predating ``weights_only``.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a dictionary")
    return value


def _validate_shard(
    path: Path,
    *,
    expected_texts: Sequence[str],
    expected_splits: Sequence[str],
) -> None:
    shard = _load_shard(path)
    expected_keys = {"texts", "splits", "teacher_embeddings", "targets"}
    if set(shard) != expected_keys:
        raise ValueError(f"{path} has keys {sorted(shard)}; expected {sorted(expected_keys)}")
    if shard["texts"] != list(expected_texts):
        raise ValueError(f"{path} prompt text/order does not match the input JSONL")
    if shard["splits"] != list(expected_splits):
        raise ValueError(f"{path} split/order does not match the input JSONL")
    embeddings = shard["teacher_embeddings"]
    targets = shard["targets"]
    expected_count = len(expected_texts)
    if not isinstance(embeddings, Tensor) or embeddings.shape != (
        expected_count,
        TEACHER_DIM,
    ):
        raise ValueError(f"{path} teacher_embeddings must have shape [{expected_count}, {TEACHER_DIM}]")
    if not isinstance(targets, Tensor) or targets.shape != (
        expected_count,
        TARGET_DIM,
    ):
        raise ValueError(f"{path} targets must have shape [{expected_count}, {TARGET_DIM}]")
    if embeddings.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError(f"{path} tensors must both be float32")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(targets).all():
        raise ValueError(f"{path} contains non-finite teacher values")


def discover_and_validate_shards(
    output_dir: Path,
    *,
    texts: Sequence[str],
    splits: Sequence[str],
    shard_size: int,
) -> list[str]:
    """Return the contiguous, validated shard list currently on disk."""

    indexed_paths: list[tuple[int, Path]] = []
    for path in output_dir.glob("teacher-*.pt"):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected teacher shard filename: {path}")
        indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort()

    indices = [index for index, _path in indexed_paths]
    if indices != list(range(len(indices))):
        raise ValueError(f"teacher shards must be contiguous from 00000; found indices {indices}")

    maximum_shards = (len(texts) + shard_size - 1) // shard_size
    if len(indexed_paths) > maximum_shards:
        raise ValueError(f"found {len(indexed_paths)} shards but input permits only {maximum_shards}")

    names: list[str] = []
    for index, path in indexed_paths:
        start = index * shard_size
        end = min(start + shard_size, len(texts))
        _validate_shard(
            path,
            expected_texts=texts[start:end],
            expected_splits=splits[start:end],
        )
        names.append(path.name)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache production ARDY LLM2Vec embeddings and bias-free Core40 "
            "root/body projection targets in resumable PyTorch shards."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="prepared prompt JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="cache directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Core40 denoiser.safetensors",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="LLM2Vec base model")
    parser.add_argument("--peft-model", default=DEFAULT_PEFT_MODEL, help="LLM2Vec PEFT model")
    parser.add_argument("--root-key", default=DEFAULT_ROOT_KEY, help="root projection tensor key")
    parser.add_argument("--body-key", default=DEFAULT_BODY_KEY, help="body projection tensor key")
    parser.add_argument("--device", default="auto", help="teacher/projection device")
    parser.add_argument("--shard-size", type=int, default=256, help="prompts per .pt shard")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="validate and continue an existing compatible cache",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_start = time.perf_counter()
    if args.shard_size <= 0:
        print("error: --shard-size must be positive", file=sys.stderr)
        return 2

    try:
        texts, splits = read_prompts(args.input)
        input_sha256 = sha256_file(args.input)
        checkpoint_sha256 = sha256_file(args.checkpoint)
        inspect_projection_shapes(args.checkpoint, args.root_key, args.body_key)
        split_counts = Counter(splits)
        identity = _metadata_identity(
            input_path=args.input,
            input_sha256=input_sha256,
            count=len(texts),
            split_counts={split: split_counts.get(split, 0) for split in sorted(VALID_SPLITS)},
            base_model=args.base_model,
            peft_model=args.peft_model,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            root_key=args.root_key,
            body_key=args.body_key,
            shard_size=args.shard_size,
        )
        current_provenance = _runtime_provenance(
            requested_device=args.device,
            base_model=args.base_model,
            peft_model=args.peft_model,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = args.output_dir / METADATA_FILENAME
        previous_metadata = _read_metadata(metadata_path)
        if previous_metadata is not None:
            if not args.resume:
                raise ValueError(f"{metadata_path} already exists and --no-resume was requested")
            _check_metadata_identity(previous_metadata, identity)
        elif any(args.output_dir.glob("teacher-*.pt")):
            raise ValueError(
                f"{args.output_dir} contains teacher shards but no {METADATA_FILENAME}; "
                "their model/checkpoint identity cannot be verified, so use a new "
                "--output-dir"
            )

        # Writing an initial manifest closes the tiny crash window between the
        # first atomic shard and the first metadata update.
        prior_elapsed = float(previous_metadata.get("elapsed_seconds", 0.0)) if previous_metadata is not None else 0.0
        shard_names = discover_and_validate_shards(
            args.output_dir,
            texts=texts,
            splits=splits,
            shard_size=args.shard_size,
        )
        shard_sha256 = {name: sha256_file(args.output_dir / name) for name in shard_names}
        runtime_provenance = _select_resume_provenance(
            previous_metadata,
            current_provenance,
            has_shards=bool(shard_names),
        )
        if previous_metadata is not None:
            declared_shards = previous_metadata.get("shards", [])
            if not isinstance(declared_shards, list) or not all(isinstance(name, str) for name in declared_shards):
                raise ValueError(f"{metadata_path}: 'shards' must be a list of filenames")
            if shard_names[: len(declared_shards)] != declared_shards:
                raise ValueError(f"{metadata_path}: declared shard list does not match files on disk")
            declared_hashes = previous_metadata.get("shard_sha256")
            if declared_hashes is not None:
                if not isinstance(declared_hashes, dict):
                    raise ValueError(f"{metadata_path}: 'shard_sha256' must be an object")
                hash_mismatches = [
                    name
                    for name, expected_hash in declared_hashes.items()
                    if name not in shard_sha256 or shard_sha256[name] != expected_hash
                ]
                if hash_mismatches:
                    raise ValueError(f"{metadata_path}: shard SHA-256 mismatch for {hash_mismatches}")

        completed_count = min(len(shard_names) * args.shard_size, len(texts))
        metadata = {
            **identity,
            **runtime_provenance,
            "completed_count": completed_count,
            "elapsed_seconds": prior_elapsed + (time.perf_counter() - session_start),
            "status": "complete" if completed_count == len(texts) else "in_progress",
            "shards": shard_names,
            "shard_sha256": shard_sha256,
        }
        _atomic_json_dump(metadata, metadata_path)

        if completed_count == len(texts):
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "count": len(texts),
                        "shards": len(shard_names),
                        "output_dir": str(args.output_dir),
                    },
                    indent=2,
                )
            )
            return 0

        encoder = LLM2VecEncoder(
            base_model_name_or_path=args.base_model,
            peft_model_name_or_path=args.peft_model,
            dtype="bfloat16",
            llm_dim=TEACHER_DIM,
            device=args.device,
        )
        # A clean Hugging Face cache has no revision hints until model loading
        # resolves/downloads the snapshots. Refresh now and validate again
        # before mixing any new teacher output with resumed shards.
        refreshed_provenance = _runtime_provenance(
            requested_device=args.device,
            base_model=args.base_model,
            peft_model=args.peft_model,
        )
        runtime_provenance = _select_resume_provenance(
            previous_metadata,
            refreshed_provenance,
            has_shards=bool(shard_names),
        )
        metadata = {
            **identity,
            **runtime_provenance,
            "completed_count": completed_count,
            "elapsed_seconds": prior_elapsed + (time.perf_counter() - session_start),
            "status": "in_progress",
            "shards": shard_names,
            "shard_sha256": shard_sha256,
        }
        _atomic_json_dump(metadata, metadata_path)

        projection_device = torch.device(encoder.get_device())
        root_weight, body_weight = load_projection_weights(
            args.checkpoint,
            args.root_key,
            args.body_key,
            projection_device,
        )

        first_shard = len(shard_names)
        total_shards = (len(texts) + args.shard_size - 1) // args.shard_size
        for shard_index in range(first_shard, total_shards):
            start = shard_index * args.shard_size
            end = min(start + args.shard_size, len(texts))
            shard_texts = texts[start:end]
            shard_splits = splits[start:end]

            # LLM2VecEncoder hard-codes encode(batch_size=1); passing a list
            # here only lets its wrapper collect one complete output shard.
            encoded, lengths = encoder(shard_texts)
            if any(length != 1 for length in lengths):
                raise RuntimeError(f"teacher returned unexpected lengths: {lengths}")
            if encoded.ndim != 3 or encoded.shape[1:] != (1, TEACHER_DIM):
                raise RuntimeError(f"teacher returned shape {tuple(encoded.shape)}; expected [N, 1, {TEACHER_DIM}]")
            embeddings_device = encoded[:, 0, :].to(
                device=projection_device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                # No embed_text biases: the student learns only W @ e.
                root_target = torch.nn.functional.linear(embeddings_device, root_weight)
                body_target = torch.nn.functional.linear(embeddings_device, body_weight)
                targets = torch.cat((root_target, body_target), dim=-1)

            embeddings_cpu = embeddings_device.detach().to("cpu").contiguous()
            targets_cpu = targets.detach().to("cpu").contiguous()
            if not torch.isfinite(embeddings_cpu).all() or not torch.isfinite(targets_cpu).all():
                raise RuntimeError(f"non-finite teacher value encountered in shard {shard_index}")

            shard_name = f"teacher-{shard_index:05d}.pt"
            _atomic_torch_save(
                {
                    "texts": list(shard_texts),
                    "splits": list(shard_splits),
                    "teacher_embeddings": embeddings_cpu,
                    "targets": targets_cpu,
                },
                args.output_dir / shard_name,
            )
            shard_names.append(shard_name)
            shard_sha256[shard_name] = sha256_file(args.output_dir / shard_name)
            completed_count = end
            metadata = {
                **identity,
                **runtime_provenance,
                "completed_count": completed_count,
                "elapsed_seconds": prior_elapsed + (time.perf_counter() - session_start),
                "status": "complete" if completed_count == len(texts) else "in_progress",
                "shards": shard_names,
                "shard_sha256": shard_sha256,
            }
            _atomic_json_dump(metadata, metadata_path)
            print(
                f"[{shard_index + 1}/{total_shards}] cached {completed_count}/{len(texts)} prompts",
                flush=True,
            )

            del encoded, embeddings_device, root_target, body_target, targets
            del embeddings_cpu, targets_cpu

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "complete",
                "count": len(texts),
                "shards": len(shard_names),
                "output_dir": str(args.output_dir),
                "elapsed_seconds": metadata["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
