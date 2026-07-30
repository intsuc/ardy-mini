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
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import Tensor

from ardy.minilm_teacher_cache import (
    TEACHER_CACHE_FORMAT_VERSION,
    TIMELINE_PROMPT_MAX_CHARACTERS,
    TIMELINE_PROMPT_SOURCES,
    normalize_timeline_prompt,
    prompt_provenance_sha256,
    timeline_prompt_deduplication_key,
    validate_prompt_provenance,
    validated_teacher_lineage,
)
from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

DEFAULT_INPUT = Path("artifacts/minilm/prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/minilm/teacher")
DEFAULT_CHECKPOINT = Path("checkpoints/ARDY-Core-RP-20FPS-Horizon40/denoiser.safetensors")
DEFAULT_BASE_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
DEFAULT_PEFT_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
DEFAULT_FOUNDATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_FOUNDATION_MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
DEFAULT_BASE_MODEL_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
DEFAULT_PEFT_MODEL_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"
DEFAULT_ROOT_KEY = "denoiser.backbone.root_model.embed_text.weight"
DEFAULT_BODY_KEY = "denoiser.backbone.body_model.embed_text.weight"
METADATA_FILENAME = "metadata.json"
SHARD_PATTERN = re.compile(r"^teacher-(\d{5})\.pt$")
FORMAT_VERSION = TEACHER_CACHE_FORMAT_VERSION
TEACHER_DIM = 4096
BRANCH_DIM = 1024
TARGET_DIM = 2 * BRANCH_DIM
VALID_SPLITS = frozenset(("train", "val", "test"))
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def resolve_pinned_snapshot(
    repo_id: str,
    revision: str,
    *,
    allow_patterns: Sequence[str],
    local_files_only: bool = False,
) -> Path:
    """Resolve an immutable Hub commit and verify the returned snapshot path.

    ``local_files_only`` is used by benchmarks so network acquisition cannot
    contaminate model-load timing.
    """

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("model repo ID must be a non-empty string")
    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"model revision must be a 40-character lowercase commit SHA, got {revision!r}")
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=os.environ.get("HUGGINGFACE_CACHE_DIR"),
            allow_patterns=list(allow_patterns),
            local_files_only=local_files_only,
        )
    ).absolute()
    if path.parent.name != "snapshots" or path.name != revision:
        raise RuntimeError(f"{repo_id}@{revision} resolved to unexpected snapshot path {path}")
    return path


def _runtime_provenance(
    *,
    requested_device: str,
    foundation_model: str,
    foundation_model_revision: str,
    base_model: str,
    base_model_revision: str,
    peft_model: str,
    peft_model_revision: str,
) -> dict[str, Any]:
    env_device = os.environ.get("TEXT_ENCODER_DEVICE")
    effective_device = env_device if env_device and requested_device in (None, "auto") else requested_device
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
    return {
        "device": device_details,
        "model_revisions": {
            "foundation_model": foundation_model,
            "foundation": foundation_model_revision,
            "base": base_model_revision,
            "peft": peft_model_revision,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "safetensors": _package_version("safetensors"),
        },
        "provenance_status": "recorded",
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
        raise ValueError("existing teacher cache is missing required runtime provenance")

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
    preserved_revisions = dict(previous_revisions)
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
    seen_prompt_keys: set[str] = set()
    group_splits: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            expected_keys = {"text", "split", "group", "source"}
            if set(record) != expected_keys:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly {sorted(expected_keys)}, got {sorted(record)}"
                )
            text = record.get("text")
            split = record.get("split")
            group = record.get("group")
            source = record.get("source")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: 'text' must be a non-empty string")
            if text != normalize_timeline_prompt(text):
                raise ValueError(f"{path}:{line_number}: 'text' is not in canonical NFKC/whitespace form")
            if len(text) > TIMELINE_PROMPT_MAX_CHARACTERS:
                raise ValueError(f"{path}:{line_number}: 'text' exceeds {TIMELINE_PROMPT_MAX_CHARACTERS} characters")
            if not isinstance(split, str) or split not in VALID_SPLITS:
                raise ValueError(f"{path}:{line_number}: 'split' must be one of {sorted(VALID_SPLITS)}")
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{path}:{line_number}: 'group' must be a non-empty string")
            if group != normalize_timeline_prompt(group).casefold():
                raise ValueError(f"{path}:{line_number}: 'group' is not in canonical NFKC/casefold/whitespace form")
            if not isinstance(source, str) or source not in TIMELINE_PROMPT_SOURCES:
                raise ValueError(f"{path}:{line_number}: 'source' must be one of {list(TIMELINE_PROMPT_SOURCES)}")

            prompt_key = timeline_prompt_deduplication_key(text)
            if prompt_key in seen_prompt_keys:
                raise ValueError(f"{path}:{line_number}: duplicate canonical prompt text {text!r}")
            seen_prompt_keys.add(prompt_key)
            previous_split = group_splits.setdefault(group, split)
            if previous_split != split:
                raise ValueError(
                    f"{path}:{line_number}: group {group!r} appears in both {previous_split!r} and {split!r}"
                )
            texts.append(text)
            splits.append(split)
    if not texts:
        raise ValueError(f"{path} contains no prompts")
    return texts, splits


def read_prompt_provenance(
    path: Path,
    *,
    input_path: Path,
    input_sha256: str,
    count: int,
    split_counts: Mapping[str, int],
) -> tuple[dict[str, Any], str]:
    """Read and validate the provenance sidecar bound to a prompt manifest."""

    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"prompt provenance sidecar not found: {path}; generate it with prepare_prompts.py or pass --input-metadata"
        ) from error
    input_metadata_sha256 = hashlib.sha256(encoded).hexdigest()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid prompt provenance JSON") from error
    validate_prompt_provenance(
        value,
        expected_manifest_sha256=input_sha256,
        expected_manifest_filename=input_path.name,
        expected_count=count,
        expected_split_counts=split_counts,
    )
    canonical_sha256 = prompt_provenance_sha256(value)
    if input_metadata_sha256 != canonical_sha256:
        raise ValueError(
            f"{path}: prompt provenance JSON is not in the canonical encoding "
            f"emitted by prepare_prompts.py (canonical SHA-256 {canonical_sha256}, "
            f"actual {input_metadata_sha256})"
        )
    return value, input_metadata_sha256


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
    input_metadata_sha256: str,
    corpus_provenance: Mapping[str, Any],
    count: int,
    split_counts: Mapping[str, int],
    foundation_model: str,
    foundation_model_revision: str,
    base_model: str,
    base_model_revision: str,
    peft_model: str,
    peft_model_revision: str,
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
        "input_metadata_sha256": input_metadata_sha256,
        "corpus_provenance": dict(corpus_provenance),
        "model_name": model_name,
        "foundation_model_name_or_path": foundation_model,
        "foundation_model_revision": foundation_model_revision,
        "base_model_name_or_path": base_model,
        "base_model_revision": base_model_revision,
        "peft_model_name_or_path": peft_model,
        "peft_model_revision": peft_model_revision,
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
    parser.add_argument(
        "--input-metadata",
        type=Path,
        default=None,
        help="prompt provenance sidecar; defaults to INPUT with .metadata.json suffix",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="cache directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Core40 denoiser.safetensors",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="LLM2Vec base model")
    parser.add_argument("--peft-model", default=DEFAULT_PEFT_MODEL, help="LLM2Vec PEFT model")
    parser.add_argument(
        "--foundation-model",
        default=DEFAULT_FOUNDATION_MODEL,
        help="LLM2Vec foundation model Hub repository",
    )
    parser.add_argument(
        "--foundation-model-revision",
        default=DEFAULT_FOUNDATION_MODEL_REVISION,
        help="immutable foundation-model commit",
    )
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_MODEL_REVISION,
        help="immutable MNTP-adapter commit",
    )
    parser.add_argument(
        "--peft-model-revision",
        default=DEFAULT_PEFT_MODEL_REVISION,
        help="immutable supervised-adapter commit",
    )
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
        model_ids = {
            "--foundation-model": args.foundation_model,
            "--base-model": args.base_model,
            "--peft-model": args.peft_model,
        }
        for flag, repo_id in model_ids.items():
            if not isinstance(repo_id, str) or not repo_id.strip():
                raise ValueError(f"{flag} must be a non-empty Hub repository ID")
        revisions = {
            "--foundation-model-revision": args.foundation_model_revision,
            "--base-model-revision": args.base_model_revision,
            "--peft-model-revision": args.peft_model_revision,
        }
        for flag, revision in revisions.items():
            if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
                raise ValueError(f"{flag} must be a 40-character lowercase commit SHA")

        texts, splits = read_prompts(args.input)
        input_sha256 = sha256_file(args.input)
        split_counts = Counter(splits)
        validated_split_counts = {split: split_counts.get(split, 0) for split in sorted(VALID_SPLITS)}
        input_metadata_path = args.input_metadata or args.input.with_suffix(".metadata.json")
        corpus_provenance, input_metadata_sha256 = read_prompt_provenance(
            input_metadata_path,
            input_path=args.input,
            input_sha256=input_sha256,
            count=len(texts),
            split_counts=validated_split_counts,
        )
        checkpoint_sha256 = sha256_file(args.checkpoint)
        inspect_projection_shapes(args.checkpoint, DEFAULT_ROOT_KEY, DEFAULT_BODY_KEY)
        identity = _metadata_identity(
            input_path=args.input,
            input_sha256=input_sha256,
            input_metadata_sha256=input_metadata_sha256,
            corpus_provenance=corpus_provenance,
            count=len(texts),
            split_counts=validated_split_counts,
            foundation_model=args.foundation_model,
            foundation_model_revision=args.foundation_model_revision,
            base_model=args.base_model,
            base_model_revision=args.base_model_revision,
            peft_model=args.peft_model,
            peft_model_revision=args.peft_model_revision,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            root_key=DEFAULT_ROOT_KEY,
            body_key=DEFAULT_BODY_KEY,
            shard_size=args.shard_size,
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

        prior_elapsed = float(previous_metadata.get("elapsed_seconds", 0.0)) if previous_metadata is not None else 0.0
        shard_names = discover_and_validate_shards(
            args.output_dir,
            texts=texts,
            splits=splits,
            shard_size=args.shard_size,
        )
        shard_sha256 = {name: sha256_file(args.output_dir / name) for name in shard_names}
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
        if previous_metadata is not None and previous_metadata.get("status") == "complete":
            # A complete cache is immutable output, not a resumable generation
            # session. Validate its recorded lineage and files without comparing
            # them with, or rewriting them from, this machine's runtime.
            validated_teacher_lineage(previous_metadata)
            if completed_count != len(texts):
                raise ValueError(
                    f"{metadata_path}: teacher cache is marked complete but only "
                    f"{completed_count}/{len(texts)} examples are present"
                )
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

        current_provenance = _runtime_provenance(
            requested_device=args.device,
            foundation_model=args.foundation_model,
            foundation_model_revision=args.foundation_model_revision,
            base_model=args.base_model,
            base_model_revision=args.base_model_revision,
            peft_model=args.peft_model,
            peft_model_revision=args.peft_model_revision,
        )
        runtime_provenance = _select_resume_provenance(
            previous_metadata,
            current_provenance,
            has_shards=bool(shard_names),
        )

        # Writing an initial manifest closes the tiny crash window between the
        # first atomic shard and the first metadata update.
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

        foundation_snapshot = resolve_pinned_snapshot(
            args.foundation_model,
            args.foundation_model_revision,
            allow_patterns=(
                "config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
            ),
        )
        base_snapshot = resolve_pinned_snapshot(
            args.base_model,
            args.base_model_revision,
            allow_patterns=(
                "adapter_config.json",
                "adapter_model.safetensors",
                "config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        )
        peft_snapshot = resolve_pinned_snapshot(
            args.peft_model,
            args.peft_model_revision,
            allow_patterns=("adapter_config.json", "adapter_model.safetensors"),
        )
        encoder = LLM2VecEncoder(
            base_model_name_or_path=str(base_snapshot),
            peft_model_name_or_path=str(peft_snapshot),
            foundation_model_name_or_path=str(foundation_snapshot),
            dtype="bfloat16",
            llm_dim=TEACHER_DIM,
            device=args.device,
        )
        refreshed_provenance = _runtime_provenance(
            requested_device=args.device,
            foundation_model=args.foundation_model,
            foundation_model_revision=args.foundation_model_revision,
            base_model=args.base_model,
            base_model_revision=args.base_model_revision,
            peft_model=args.peft_model,
            peft_model_revision=args.peft_model_revision,
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
            DEFAULT_ROOT_KEY,
            DEFAULT_BODY_KEY,
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
