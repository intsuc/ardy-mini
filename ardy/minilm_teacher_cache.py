# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Strict validation for MiniLM distillation teacher caches."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

VALID_SPLITS = ("train", "val", "test")
EXPECTED_TEACHER_DIM = 4096
EXPECTED_TARGET_DIM = 2048
TEACHER_CACHE_FORMAT_VERSION = 3
EXPECTED_TARGET_KEYS = (
    "denoiser.backbone.root_model.embed_text.weight",
    "denoiser.backbone.body_model.embed_text.weight",
)
EXPECTED_TARGET_ORDER = ("root", "body")
EXPECTED_DTYPE_CONTRACT = {
    "teacher_model": "bfloat16",
    "teacher_embeddings": "float32",
    "projection_weights": "float32",
    "targets": "float32",
}
MODEL_REVISION_KEYS = (
    "foundation_model",
    "foundation",
    "base",
    "peft",
)
VERSION_KEYS = ("torch", "transformers", "peft", "safetensors")
PROMPT_PROVENANCE_FORMAT = "ardy-minilm-prompt-provenance"
PROMPT_PROVENANCE_FORMAT_VERSION = 1
TIMELINE_DATASET_REPO = "nvidia/SEED-Timeline-Annotations"
TIMELINE_DATASET_REVISION = "b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f"
TIMELINE_DATASET_FILENAME = "timelines.jsonl"
TIMELINE_DATASET_SIZE_BYTES = 80_373_523
TIMELINE_DATASET_SHA256 = "379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8"
TIMELINE_DATASET_URL = f"https://huggingface.co/datasets/{TIMELINE_DATASET_REPO}"
TIMELINE_PROMPT_SOURCES = ("overview_description", "events.description")
TIMELINE_PROMPT_NORMALIZATION = "Unicode NFKC, trim, collapse whitespace"
TIMELINE_PROMPT_DEDUPLICATION = (
    "global case- and punctuation-insensitive alphanumeric-token match "
    "after normalization; overview_description preferred"
)
TIMELINE_PROMPT_MAX_CHARACTERS = 512
TIMELINE_PROMPT_GROUPING = (
    "NFKC/casefold filename; require <take>__A<actor>[_M], remove actor, "
    "mirror, and terminal three-digit non-angle take suffixes; union "
    "propagated_from_filename families"
)
TIMELINE_SPLIT_HASH_NAMESPACE = "ardy-minilm-nvidia-timeline-split-v3"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHARD_PATTERN = re.compile(r"^teacher-\d{5}\.pt$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


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


def normalize_timeline_prompt(value: str | None) -> str:
    """Apply the normalization used by the pinned Timeline prompt producer."""

    if value is None:
        return ""
    return _WHITESPACE_PATTERN.sub(
        " ",
        unicodedata.normalize("NFKC", value),
    ).strip()


def timeline_prompt_deduplication_key(text: str) -> str:
    """Return the canonical prompt identity used for global deduplication."""

    return " ".join("".join(character if character.isalnum() else " " for character in text.casefold()).split())


def _provenance_object(value: Any, field: str, expected_keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"prompt provenance {field} must be a JSON object")
    if set(value) != expected_keys:
        raise ValueError(f"prompt provenance {field} must contain exactly {sorted(expected_keys)}, got {sorted(value)}")
    return value


def _provenance_integer(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"prompt provenance {field} must be a {qualifier} integer")
    return value


def validate_prompt_provenance(
    value: Any,
    *,
    expected_manifest_sha256: str,
    expected_manifest_filename: str | None = None,
    expected_count: int | None = None,
    expected_split_counts: Mapping[str, int] | None = None,
) -> None:
    """Validate the pinned NVIDIA Timeline prompt-manifest provenance schema."""

    provenance = _provenance_object(
        value,
        "root",
        {
            "format",
            "format_version",
            "dataset",
            "preparation",
            "counts",
            "manifest",
        },
    )
    if provenance["format"] != PROMPT_PROVENANCE_FORMAT:
        raise ValueError(f"prompt provenance format must be {PROMPT_PROVENANCE_FORMAT!r}, got {provenance['format']!r}")
    format_version = _provenance_integer(
        provenance["format_version"],
        "format_version",
        positive=True,
    )
    if format_version != PROMPT_PROVENANCE_FORMAT_VERSION:
        raise ValueError(
            "unsupported prompt provenance format_version "
            f"{format_version}; expected {PROMPT_PROVENANCE_FORMAT_VERSION}"
        )

    dataset = _provenance_object(
        provenance["dataset"],
        "dataset",
        {
            "repo_id",
            "revision",
            "filename",
            "sha256",
            "size_bytes",
            "resolved_from",
            "owner",
            "license",
            "url",
        },
    )
    expected_dataset = {
        "repo_id": TIMELINE_DATASET_REPO,
        "revision": TIMELINE_DATASET_REVISION,
        "filename": TIMELINE_DATASET_FILENAME,
        "sha256": TIMELINE_DATASET_SHA256,
        "size_bytes": TIMELINE_DATASET_SIZE_BYTES,
        "owner": "NVIDIA",
        "license": "CC BY 4.0",
        "url": TIMELINE_DATASET_URL,
    }
    mismatches = {
        field: (dataset.get(field), expected)
        for field, expected in expected_dataset.items()
        if dataset.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"prompt provenance dataset identity mismatch: {mismatches}")
    if dataset["resolved_from"] not in {"hugging_face_hub", "local_input"}:
        raise ValueError("prompt provenance dataset.resolved_from must be 'hugging_face_hub' or 'local_input'")

    preparation = _provenance_object(
        provenance["preparation"],
        "preparation",
        {
            "sources",
            "normalization",
            "deduplication",
            "max_prompt_characters",
            "grouping",
            "split_hash_namespace",
            "seed",
            "split_ratios",
            "sample_size",
        },
    )
    if preparation["sources"] != list(TIMELINE_PROMPT_SOURCES):
        raise ValueError(f"prompt provenance preparation.sources must be {list(TIMELINE_PROMPT_SOURCES)!r}")
    expected_preparation_policies = {
        "normalization": TIMELINE_PROMPT_NORMALIZATION,
        "deduplication": TIMELINE_PROMPT_DEDUPLICATION,
        "grouping": TIMELINE_PROMPT_GROUPING,
    }
    policy_mismatches = {
        field: (preparation[field], expected)
        for field, expected in expected_preparation_policies.items()
        if preparation[field] != expected
    }
    if policy_mismatches:
        raise ValueError(f"prompt provenance preparation policy mismatch: {policy_mismatches}")
    if preparation["split_hash_namespace"] != TIMELINE_SPLIT_HASH_NAMESPACE:
        raise ValueError("prompt provenance preparation.split_hash_namespace is unsupported")
    max_prompt_characters = _provenance_integer(
        preparation["max_prompt_characters"],
        "preparation.max_prompt_characters",
        positive=True,
    )
    if max_prompt_characters != TIMELINE_PROMPT_MAX_CHARACTERS:
        raise ValueError(
            "prompt provenance preparation.max_prompt_characters must be "
            f"{TIMELINE_PROMPT_MAX_CHARACTERS}, got {max_prompt_characters}"
        )
    if isinstance(preparation["seed"], bool) or not isinstance(preparation["seed"], int):
        raise TypeError("prompt provenance preparation.seed must be an integer")
    sample_size = _provenance_integer(
        preparation["sample_size"],
        "preparation.sample_size",
    )
    split_ratios = _provenance_object(
        preparation["split_ratios"],
        "preparation.split_ratios",
        set(VALID_SPLITS),
    )
    ratio_values: list[float] = []
    for split in VALID_SPLITS:
        ratio = split_ratios[split]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError(f"prompt provenance preparation.split_ratios.{split} must be a finite non-negative number")
        ratio_values.append(float(ratio))
    if not math.isclose(sum(ratio_values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("prompt provenance preparation.split_ratios must sum to 1")

    count_keys = {
        "timeline_rows",
        "recording_families",
        "recording_components",
        "missing_propagation_references",
        "raw_overview_descriptions",
        "raw_event_descriptions",
        "dropped_prompt_too_long",
        "unique_descriptions",
        "unique_before_sampling",
        "written",
        "groups_written",
        "splits",
        "split_groups",
        "sources",
    }
    counts = _provenance_object(provenance["counts"], "counts", count_keys)
    positive_count_fields = {
        "timeline_rows",
        "recording_families",
        "recording_components",
        "raw_overview_descriptions",
        "raw_event_descriptions",
        "unique_descriptions",
        "unique_before_sampling",
        "written",
        "groups_written",
    }
    for field in count_keys - {"splits", "split_groups", "sources"}:
        _provenance_integer(
            counts[field],
            f"counts.{field}",
            positive=field in positive_count_fields,
        )
    split_counts = _provenance_object(
        counts["splits"],
        "counts.splits",
        set(VALID_SPLITS),
    )
    split_group_counts = _provenance_object(
        counts["split_groups"],
        "counts.split_groups",
        set(VALID_SPLITS),
    )
    source_counts = _provenance_object(
        counts["sources"],
        "counts.sources",
        set(TIMELINE_PROMPT_SOURCES),
    )
    validated_splits = {
        split: _provenance_integer(
            split_counts[split],
            f"counts.splits.{split}",
            positive=True,
        )
        for split in VALID_SPLITS
    }
    validated_split_groups = {
        split: _provenance_integer(
            split_group_counts[split],
            f"counts.split_groups.{split}",
            positive=True,
        )
        for split in VALID_SPLITS
    }
    validated_sources = {
        source: _provenance_integer(
            source_counts[source],
            f"counts.sources.{source}",
        )
        for source in TIMELINE_PROMPT_SOURCES
    }
    written = counts["written"]
    if sum(validated_splits.values()) != written:
        raise ValueError("prompt provenance counts.splits must sum to counts.written")
    if sum(validated_split_groups.values()) != counts["groups_written"]:
        raise ValueError("prompt provenance counts.split_groups must sum to counts.groups_written")
    if sum(validated_sources.values()) != written:
        raise ValueError("prompt provenance counts.sources must sum to counts.written")
    if counts["unique_descriptions"] != counts["unique_before_sampling"]:
        raise ValueError("prompt provenance counts.unique_descriptions must equal counts.unique_before_sampling")
    expected_written = (
        counts["unique_before_sampling"] if sample_size == 0 else min(sample_size, counts["unique_before_sampling"])
    )
    if written != expected_written:
        raise ValueError(f"prompt provenance counts.written is {written}, expected {expected_written}")
    if expected_count is not None and written != expected_count:
        raise ValueError(
            f"prompt provenance counts.written is {written}, but the prompt manifest contains {expected_count}"
        )
    if expected_split_counts is not None and validated_splits != dict(expected_split_counts):
        raise ValueError(
            f"prompt provenance split counts {validated_splits} do not match "
            f"the prompt manifest {dict(expected_split_counts)}"
        )

    manifest = _provenance_object(
        provenance["manifest"],
        "manifest",
        {"filename", "sha256"},
    )
    filename = manifest["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("prompt provenance manifest.filename must be a plain filename")
    manifest_sha256 = manifest["sha256"]
    if not isinstance(manifest_sha256, str) or _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("prompt provenance manifest.sha256 is invalid")
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "prompt provenance manifest SHA-256 mismatch: "
            f"sidecar={manifest_sha256}, prompt manifest={expected_manifest_sha256}"
        )
    if expected_manifest_filename is not None and filename != expected_manifest_filename:
        raise ValueError(
            "prompt provenance manifest filename mismatch: "
            f"sidecar={filename!r}, prompt manifest={expected_manifest_filename!r}"
        )


def prompt_provenance_sha256(value: Mapping[str, Any]) -> str:
    """Hash provenance using the canonical encoding emitted by prepare_prompts."""

    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"teacher-cache {field} must be a {qualifier} integer")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"teacher-cache {field} must be a non-empty string")
    return value


def _revision(value: Any, field: str) -> str:
    revision = _nonempty_string(value, field)
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"teacher-cache {field} must be a resolved 40-character lowercase hexadecimal revision")
    return revision


def _known_string(value: Any, field: str) -> str:
    result = _nonempty_string(value, field)
    if result == "unknown":
        raise ValueError(f"teacher-cache {field} must be known, got 'unknown'")
    return result


def _validate_device(metadata_path: Path, value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{metadata_path}: device must be an object")
    base_keys = {"cli_requested", "env_override", "requested", "resolved"}
    resolved = value.get("resolved")
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(f"{metadata_path}: device.resolved must be a non-empty string")
    try:
        resolved_device = torch.device(resolved)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"{metadata_path}: device.resolved is invalid: {resolved!r}") from error
    expected_keys = base_keys | {"index", "name", "capability"} if resolved_device.type == "cuda" else base_keys
    if set(value) != expected_keys:
        raise ValueError(f"{metadata_path}: device must contain exactly {sorted(expected_keys)}, got {sorted(value)}")
    cli_requested = value["cli_requested"]
    if cli_requested is not None and (not isinstance(cli_requested, str) or not cli_requested):
        raise ValueError(f"{metadata_path}: device.cli_requested must be null or a non-empty string")
    _nonempty_string(value["requested"], "device.requested")
    requested = value["requested"]
    if requested != "auto":
        try:
            torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"{metadata_path}: device.requested is invalid: {requested!r}") from error
    env_override = value["env_override"]
    if env_override is not None and (not isinstance(env_override, str) or not env_override):
        raise ValueError(f"{metadata_path}: device.env_override must be null or a non-empty string")
    expected_requested = env_override if env_override and cli_requested in (None, "auto") else cli_requested
    if value["requested"] != expected_requested:
        raise ValueError(
            f"{metadata_path}: device.requested must equal the environment "
            f"override or CLI request; expected {expected_requested!r}, "
            f"got {value['requested']!r}"
        )
    if requested != "auto" and resolved != requested:
        raise ValueError(
            f"{metadata_path}: device.resolved must equal the explicit request; "
            f"expected {requested!r}, got {resolved!r}"
        )
    if resolved_device.type == "cuda":
        index = value["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{metadata_path}: device.index must be a non-negative integer")
        if resolved_device.index is not None and index != resolved_device.index:
            raise ValueError(f"{metadata_path}: device.index={index} is inconsistent with device.resolved={resolved!r}")
        _nonempty_string(value["name"], "device.name")
        capability = value["capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(component, bool) or not isinstance(component, int) or component < 0
                for component in capability
            )
        ):
            raise ValueError(f"{metadata_path}: device.capability must contain two non-negative integers")


def _validate_teacher_identity(metadata_path: Path, metadata: dict[str, Any]) -> None:
    model_name = _nonempty_string(metadata["model_name"], "model_name")
    foundation_model = _nonempty_string(
        metadata["foundation_model_name_or_path"],
        "foundation_model_name_or_path",
    )
    base_model = _nonempty_string(
        metadata["base_model_name_or_path"],
        "base_model_name_or_path",
    )
    peft_model = _nonempty_string(
        metadata["peft_model_name_or_path"],
        "peft_model_name_or_path",
    )
    expected_model_name = f"{base_model} + {peft_model}"
    if model_name != expected_model_name:
        raise ValueError(
            f"{metadata_path}: model_name is inconsistent with the base/PEFT "
            f"models; expected {expected_model_name!r}, got {model_name!r}"
        )

    revisions = metadata["model_revisions"]
    if not isinstance(revisions, dict) or set(revisions) != set(MODEL_REVISION_KEYS):
        raise ValueError(f"{metadata_path}: model_revisions must contain exactly {list(MODEL_REVISION_KEYS)}")
    expected_revisions = {
        "foundation_model": foundation_model,
        "foundation": _revision(
            metadata["foundation_model_revision"],
            "foundation_model_revision",
        ),
        "base": _revision(metadata["base_model_revision"], "base_model_revision"),
        "peft": _revision(metadata["peft_model_revision"], "peft_model_revision"),
    }
    for field, expected in expected_revisions.items():
        if revisions[field] != expected:
            raise ValueError(
                f"{metadata_path}: model_revisions.{field} is inconsistent with "
                f"its top-level identity; expected {expected!r}, "
                f"got {revisions[field]!r}"
            )

    versions = metadata["versions"]
    if not isinstance(versions, dict) or set(versions) != set(VERSION_KEYS):
        raise ValueError(f"{metadata_path}: versions must contain exactly {list(VERSION_KEYS)}")
    for package in VERSION_KEYS:
        _known_string(versions[package], f"versions.{package}")

    if metadata["provenance_status"] != "recorded":
        raise ValueError(
            f"{metadata_path}: provenance_status must be 'recorded', got {metadata['provenance_status']!r}"
        )
    _validate_device(metadata_path, metadata["device"])


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
        "input_path",
        "input_sha256",
        "input_metadata_sha256",
        "corpus_provenance",
        "model_name",
        "foundation_model_name_or_path",
        "foundation_model_revision",
        "base_model_name_or_path",
        "base_model_revision",
        "peft_model_name_or_path",
        "peft_model_revision",
        "model_revisions",
        "versions",
        "device",
        "provenance_status",
        "status",
        "count",
        "completed_count",
        "split_counts",
        "shard_size",
        "shards",
        "shard_sha256",
        "teacher_dim",
        "target_dim",
        "target_keys",
        "target_order",
        "bias_applied",
        "dtype",
        "teacher_batch_size",
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
    if format_version != TEACHER_CACHE_FORMAT_VERSION:
        raise ValueError(f"{metadata_path}: unsupported teacher-cache format_version {format_version}")
    _validate_teacher_identity(metadata_path, metadata)
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
    if metadata["target_keys"] != list(EXPECTED_TARGET_KEYS):
        raise ValueError(
            f"{metadata_path}: target_keys must be {list(EXPECTED_TARGET_KEYS)!r}, got {metadata['target_keys']!r}"
        )
    if metadata["target_order"] != list(EXPECTED_TARGET_ORDER):
        raise ValueError(
            f"{metadata_path}: target_order must be {list(EXPECTED_TARGET_ORDER)!r}, got {metadata['target_order']!r}"
        )
    if metadata["bias_applied"] is not False:
        raise ValueError(f"{metadata_path}: bias_applied must be false")
    teacher_batch_size = metadata["teacher_batch_size"]
    if isinstance(teacher_batch_size, bool) or not isinstance(teacher_batch_size, int) or teacher_batch_size != 1:
        raise ValueError(f"{metadata_path}: teacher_batch_size must be 1")

    dtype = metadata["dtype"]
    if not isinstance(dtype, dict):
        raise TypeError(f"{metadata_path}: dtype must be an object")
    if dtype != EXPECTED_DTYPE_CONTRACT:
        raise ValueError(f"{metadata_path}: dtype must be exactly {EXPECTED_DTYPE_CONTRACT!r}, got {dtype!r}")

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
    input_path = metadata["input_path"]
    input_hash = metadata["input_sha256"]
    input_metadata_hash = metadata["input_metadata_sha256"]
    if not isinstance(input_path, str) or not input_path:
        raise ValueError(f"{metadata_path}: input_path must be a non-empty string")
    if not isinstance(input_hash, str) or _SHA256_PATTERN.fullmatch(input_hash) is None:
        raise ValueError(f"{metadata_path}: input_sha256 is invalid")
    if not isinstance(input_metadata_hash, str) or _SHA256_PATTERN.fullmatch(input_metadata_hash) is None:
        raise ValueError(f"{metadata_path}: input_metadata_sha256 is invalid")
    try:
        validate_prompt_provenance(
            metadata["corpus_provenance"],
            expected_manifest_sha256=input_hash,
            expected_manifest_filename=Path(input_path).name,
            expected_count=count,
            expected_split_counts=validated_split_counts,
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"{metadata_path}: invalid corpus_provenance: {error}") from error
    canonical_metadata_hash = prompt_provenance_sha256(metadata["corpus_provenance"])
    if input_metadata_hash != canonical_metadata_hash:
        raise ValueError(
            f"{metadata_path}: input_metadata_sha256 does not match corpus_provenance; "
            f"expected {canonical_metadata_hash}, got {input_metadata_hash}"
        )
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


_TEACHER_LINEAGE_FIELDS = (
    "format_version",
    "input_sha256",
    "input_metadata_sha256",
    "corpus_provenance",
    "model_name",
    "foundation_model_name_or_path",
    "foundation_model_revision",
    "base_model_name_or_path",
    "base_model_revision",
    "peft_model_name_or_path",
    "peft_model_revision",
    "model_revisions",
    "versions",
    "device",
    "provenance_status",
    "teacher_batch_size",
    "checkpoint_sha256",
    "target_keys",
    "target_order",
    "bias_applied",
    "teacher_dim",
    "target_dim",
    "dtype",
    "count",
    "split_counts",
    "shard_size",
    "shards",
    "shard_sha256",
)


def validated_teacher_lineage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable, canonical lineage from validated cache metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("teacher-cache metadata must be an object")
    metadata_copy = dict(metadata)
    _validate_metadata(
        Path("metadata.json"),
        metadata_copy,
        expected_teacher_dim=EXPECTED_TEACHER_DIM,
        expected_target_dim=EXPECTED_TARGET_DIM,
    )
    missing = [field for field in _TEACHER_LINEAGE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"teacher-cache metadata is missing lineage fields {missing}")
    lineage = {field: metadata[field] for field in _TEACHER_LINEAGE_FIELDS}
    # A JSON round-trip both detaches nested mutable values and rejects values
    # that cannot be represented in a portable artifact configuration.
    try:
        return json.loads(
            json.dumps(
                lineage,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise TypeError("teacher-cache lineage is not JSON serializable") from error


def teacher_cache_fingerprint(cache: TeacherCache) -> str:
    """Hash every portable field that identifies the cache's numeric lineage."""

    identity = validated_teacher_lineage(cache.metadata)
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact_teacher_cache_fingerprint(
    artifact_config: Mapping[str, Any],
    cache: TeacherCache,
) -> str:
    """Require a student artifact produced from exactly ``cache``.

    The artifact stores this portable digest under
    ``metadata.teacher_cache_fingerprint``. Local cache/checkpoint paths are
    deliberately absent from the digest, while the complete corpus, teacher,
    projection, runtime-provenance, and shard identities are included.
    """

    metadata = artifact_config.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("MiniLM artifact metadata must be an object")
    declared_lineage = metadata.get("teacher_cache_lineage")
    if not isinstance(declared_lineage, Mapping):
        raise TypeError("MiniLM artifact metadata.teacher_cache_lineage must be an object")
    actual_lineage = validated_teacher_lineage(cache.metadata)
    if dict(declared_lineage) != actual_lineage:
        raise ValueError("MiniLM artifact was trained from a different teacher-cache lineage")
    declared = metadata.get("teacher_cache_fingerprint")
    if not isinstance(declared, str) or _SHA256_PATTERN.fullmatch(declared) is None:
        raise ValueError("MiniLM artifact metadata.teacher_cache_fingerprint must be a SHA-256 hex digest")
    actual = teacher_cache_fingerprint(cache)
    if declared != actual:
        raise ValueError(
            f"MiniLM artifact was trained from a different teacher cache: artifact={declared}, loaded={actual}"
        )
    return actual


__all__ = [
    "EXPECTED_TARGET_DIM",
    "EXPECTED_TARGET_KEYS",
    "EXPECTED_TARGET_ORDER",
    "EXPECTED_TEACHER_DIM",
    "PROMPT_PROVENANCE_FORMAT",
    "PROMPT_PROVENANCE_FORMAT_VERSION",
    "TEACHER_CACHE_FORMAT_VERSION",
    "TIMELINE_DATASET_FILENAME",
    "TIMELINE_DATASET_REPO",
    "TIMELINE_DATASET_REVISION",
    "TIMELINE_DATASET_SHA256",
    "TIMELINE_DATASET_SIZE_BYTES",
    "TIMELINE_PROMPT_DEDUPLICATION",
    "TIMELINE_PROMPT_GROUPING",
    "TIMELINE_PROMPT_MAX_CHARACTERS",
    "TIMELINE_PROMPT_NORMALIZATION",
    "TIMELINE_PROMPT_SOURCES",
    "TIMELINE_SPLIT_HASH_NAMESPACE",
    "VALID_SPLITS",
    "TeacherCache",
    "TeacherShard",
    "load_teacher_cache",
    "normalize_timeline_prompt",
    "prompt_provenance_sha256",
    "sha256_file",
    "teacher_cache_fingerprint",
    "timeline_prompt_deduplication_key",
    "validate_artifact_teacher_cache_fingerprint",
    "validate_prompt_provenance",
    "validated_teacher_lineage",
]
