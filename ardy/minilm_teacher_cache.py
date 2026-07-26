# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Strict validation for MiniLM distillation teacher caches."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

VALID_SPLITS = ("train", "val", "test")
EXPECTED_TEACHER_DIM = 4096
EXPECTED_TARGET_DIM = 2048
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHARD_PATTERN = re.compile(r"^teacher-\d{5}\.pt$")


@dataclass(frozen=True)
class TeacherShard:
    path: Path
    texts: list[str]
    splits: list[str]
    teacher_embeddings: torch.Tensor | None
    targets: torch.Tensor | None


@dataclass(frozen=True)
class TeacherCache:
    metadata_path: Path
    metadata: dict[str, Any]
    shards: list[TeacherShard]

    @property
    def shard_paths(self) -> list[Path]:
        return [shard.path for shard in self.shards]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"teacher-cache {field} must be a {qualifier} integer")
    return value


def _resolve_metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    if path.is_dir():
        path = path / "metadata.json"
    elif path.name != "metadata.json":
        raise ValueError("strict teacher-cache validation requires a cache directory or metadata.json")
    if not path.is_file():
        raise FileNotFoundError(f"teacher-cache metadata not found: {path}")
    return path.resolve()


def _load_metadata(cache_path: str | Path) -> tuple[Path, dict[str, Any]]:
    metadata_path = _resolve_metadata_path(cache_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata_path}: invalid JSON") from error
    if not isinstance(metadata, dict):
        raise TypeError(f"{metadata_path}: metadata must be a JSON object")
    return metadata_path, metadata


def _validate_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    expected_teacher_dim: int,
    expected_target_dim: int,
) -> tuple[list[str], dict[str, int], int, int]:
    required = {
        "format_version",
        "status",
        "count",
        "completed_count",
        "split_counts",
        "shard_size",
        "shards",
        "shard_sha256",
        "teacher_dim",
        "target_dim",
        "target_order",
        "bias_applied",
        "dtype",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"{metadata_path}: missing required metadata fields {missing}")
    format_version = _integer(
        metadata["format_version"],
        "format_version",
        positive=True,
    )
    if format_version != 1:
        raise ValueError(f"{metadata_path}: unsupported teacher-cache format_version {format_version}")
    if metadata["status"] != "complete":
        raise ValueError(f"{metadata_path}: teacher cache is incomplete (status={metadata['status']!r})")

    count = _integer(metadata["count"], "count", positive=True)
    completed_count = _integer(metadata["completed_count"], "completed_count")
    if completed_count != count:
        raise ValueError(f"{metadata_path}: teacher cache is incomplete ({completed_count}/{count} examples cached)")
    shard_size = _integer(metadata["shard_size"], "shard_size", positive=True)

    teacher_dim = _integer(metadata["teacher_dim"], "teacher_dim", positive=True)
    target_dim = _integer(metadata["target_dim"], "target_dim", positive=True)
    if teacher_dim != expected_teacher_dim:
        raise ValueError(f"{metadata_path}: teacher_dim is {teacher_dim}, expected {expected_teacher_dim}")
    if target_dim != expected_target_dim:
        raise ValueError(f"{metadata_path}: target_dim is {target_dim}, expected {expected_target_dim}")
    if metadata["target_order"] != ["root", "body"]:
        raise ValueError(f"{metadata_path}: target_order must be ['root', 'body'], got {metadata['target_order']!r}")
    if metadata["bias_applied"] is not False:
        raise ValueError(f"{metadata_path}: bias_applied must be false")

    dtype = metadata["dtype"]
    if not isinstance(dtype, dict):
        raise TypeError(f"{metadata_path}: dtype must be an object")
    for field in ("teacher_embeddings", "targets"):
        if dtype.get(field) != "float32":
            raise ValueError(f"{metadata_path}: dtype.{field} must be 'float32'")

    split_counts = metadata["split_counts"]
    if not isinstance(split_counts, dict) or set(split_counts) != set(VALID_SPLITS):
        raise ValueError(f"{metadata_path}: split_counts must contain exactly {list(VALID_SPLITS)}")
    validated_split_counts = {split: _integer(split_counts[split], f"split_counts.{split}") for split in VALID_SPLITS}
    if sum(validated_split_counts.values()) != count:
        raise ValueError(
            f"{metadata_path}: split_counts sum to {sum(validated_split_counts.values())}, expected {count}"
        )

    shard_names = metadata["shards"]
    if (
        not isinstance(shard_names, list)
        or not shard_names
        or any(not isinstance(name, str) or not _SHARD_PATTERN.fullmatch(name) for name in shard_names)
    ):
        raise ValueError(f"{metadata_path}: shards must be a non-empty list of teacher-NNNNN.pt filenames")
    if len(set(shard_names)) != len(shard_names):
        raise ValueError(f"{metadata_path}: shards contains duplicate filenames")
    expected_names = [f"teacher-{index:05d}.pt" for index in range(len(shard_names))]
    if shard_names != expected_names:
        raise ValueError(f"{metadata_path}: shards must be contiguous and ordered from teacher-00000.pt")
    expected_shard_count = (count + shard_size - 1) // shard_size
    if len(shard_names) != expected_shard_count:
        raise ValueError(
            f"{metadata_path}: {len(shard_names)} shards listed, expected "
            f"{expected_shard_count} for count={count}, shard_size={shard_size}"
        )

    shard_hashes = metadata["shard_sha256"]
    if not isinstance(shard_hashes, dict) or set(shard_hashes) != set(shard_names):
        raise ValueError(f"{metadata_path}: shard_sha256 keys must exactly match the shard manifest")
    invalid_hashes = [
        name
        for name, value in shard_hashes.items()
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
    ]
    if invalid_hashes:
        raise ValueError(f"{metadata_path}: invalid shard SHA-256 values for {invalid_hashes}")

    checkpoint_path = metadata["checkpoint_path"]
    checkpoint_hash = metadata["checkpoint_sha256"]
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(f"{metadata_path}: checkpoint_path must be a non-empty string")
    if not isinstance(checkpoint_hash, str) or _SHA256_PATTERN.fullmatch(checkpoint_hash) is None:
        raise ValueError(f"{metadata_path}: checkpoint_sha256 is invalid")
    return shard_names, validated_split_counts, count, shard_size


def _load_matrix(
    value: Any,
    *,
    path: Path,
    key: str,
    rows: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: {key} must be a tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{path}: {key} must be float32, got {value.dtype}")
    if value.shape != (rows, width):
        raise ValueError(f"{path}: {key} must have shape [{rows}, {width}], got {tuple(value.shape)}")
    value = value.detach().cpu()
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: {key} contains non-finite values")
    return value


def _load_shard(
    path: Path,
    *,
    expected_rows: int,
    teacher_dim: int,
    target_dim: int,
    keep_teacher_embeddings: bool,
    keep_targets: bool,
) -> TeacherShard:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"{path}: failed to safely load teacher shard") from error
    required = {"texts", "splits", "teacher_embeddings", "targets"}
    if not isinstance(payload, dict) or set(payload) != required:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{path}: expected exactly shard keys {sorted(required)}, got {keys}")

    texts = payload["texts"]
    splits = payload["splits"]
    if (
        not isinstance(texts, list)
        or len(texts) != expected_rows
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ValueError(f"{path}: texts must contain exactly {expected_rows} non-empty strings")
    if (
        not isinstance(splits, list)
        or len(splits) != expected_rows
        or any(split not in VALID_SPLITS for split in splits)
    ):
        raise ValueError(f"{path}: splits must contain exactly {expected_rows} values from {list(VALID_SPLITS)}")

    embeddings = _load_matrix(
        payload["teacher_embeddings"],
        path=path,
        key="teacher_embeddings",
        rows=expected_rows,
        width=teacher_dim,
    )
    targets = _load_matrix(
        payload["targets"],
        path=path,
        key="targets",
        rows=expected_rows,
        width=target_dim,
    )
    return TeacherShard(
        path=path,
        texts=list(texts),
        splits=list(splits),
        teacher_embeddings=embeddings if keep_teacher_embeddings else None,
        targets=targets if keep_targets else None,
    )


def load_teacher_cache(
    cache_path: str | Path,
    *,
    expected_teacher_dim: int = EXPECTED_TEACHER_DIM,
    expected_target_dim: int = EXPECTED_TARGET_DIM,
    keep_teacher_embeddings: bool = False,
    keep_targets: bool = True,
) -> TeacherCache:
    """Load a complete teacher cache after validating its full manifest."""
    metadata_path, metadata = _load_metadata(cache_path)
    shard_names, expected_split_counts, count, shard_size = _validate_metadata(
        metadata_path,
        metadata,
        expected_teacher_dim=expected_teacher_dim,
        expected_target_dim=expected_target_dim,
    )
    cache_dir = metadata_path.parent
    discovered_names = sorted(path.name for path in cache_dir.glob("teacher-*.pt"))
    if discovered_names != shard_names:
        raise ValueError(
            f"{metadata_path}: on-disk teacher shards do not exactly match manifest; "
            f"disk={discovered_names}, manifest={shard_names}"
        )

    shards: list[TeacherShard] = []
    observed_splits: Counter[str] = Counter()
    for index, shard_name in enumerate(shard_names):
        shard_path = (cache_dir / shard_name).resolve()
        if shard_path.parent != cache_dir:
            raise ValueError(f"{metadata_path}: shard path escapes cache directory")
        if not shard_path.is_file():
            raise FileNotFoundError(f"teacher-cache shard not found: {shard_path}")
        actual_hash = sha256_file(shard_path)
        expected_hash = metadata["shard_sha256"][shard_name]
        if actual_hash != expected_hash:
            raise ValueError(f"{shard_path}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}")
        expected_rows = min(shard_size, count - index * shard_size)
        shard = _load_shard(
            shard_path,
            expected_rows=expected_rows,
            teacher_dim=expected_teacher_dim,
            target_dim=expected_target_dim,
            keep_teacher_embeddings=keep_teacher_embeddings,
            keep_targets=keep_targets,
        )
        observed_splits.update(shard.splits)
        shards.append(shard)

    loaded_count = sum(len(shard.texts) for shard in shards)
    if loaded_count != count:
        raise ValueError(f"{metadata_path}: loaded {loaded_count} rows, metadata declares {count}")
    observed = {split: observed_splits.get(split, 0) for split in VALID_SPLITS}
    if observed != expected_split_counts:
        raise ValueError(
            f"{metadata_path}: observed split counts {observed} do not match metadata {expected_split_counts}"
        )
    return TeacherCache(metadata_path=metadata_path, metadata=metadata, shards=shards)


def teacher_cache_fingerprint(cache: TeacherCache) -> str:
    identity = {
        "format_version": cache.metadata["format_version"],
        "input_sha256": cache.metadata.get("input_sha256"),
        "checkpoint_sha256": cache.metadata["checkpoint_sha256"],
        "shard_sha256": cache.metadata["shard_sha256"],
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EXPECTED_TARGET_DIM",
    "EXPECTED_TEACHER_DIM",
    "VALID_SPLITS",
    "TeacherCache",
    "TeacherShard",
    "load_teacher_cache",
    "sha256_file",
    "teacher_cache_fingerprint",
]
