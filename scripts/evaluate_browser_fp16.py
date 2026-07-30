# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Compare FP32 and mixed-FP16 browser model files on fixed rollouts."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from ardy.browser.precision import (
    MIXED_FP16_POLICIES,
    MIXED_FP16_POLICY_VERSION,
    validate_fp32_source_graph,
    validate_no_external_data,
    validate_no_storage_only_fp16_casts,
    validate_production_policy_coverage,
)
from ardy.minilm_teacher_cache import (
    prompt_provenance_sha256,
    validate_prompt_provenance,
)


@dataclass(frozen=True)
class ModelFilesRuntime:
    directory: Path
    manifest: dict[str, Any]
    text_encoder: Any
    denoiser: Any
    decoder: Any


@dataclass
class RolloutState:
    """Browser-equivalent autoregressive state for one model."""

    global_hybrid: np.ndarray
    initial_translation: np.ndarray
    initial_heading: float


@dataclass(frozen=True)
class PreparedHistory:
    """One history window after browser world-to-local recentering."""

    history: np.ndarray | None
    history_tokens: int
    history_frames: int
    global_translation: np.ndarray
    first_heading_angle: float


@dataclass(frozen=True)
class RolloutResult:
    """Generated windows, their concatenation, and continuity diagnostics."""

    windows: list[dict[str, np.ndarray]]
    accumulated: dict[str, np.ndarray]
    continuity: list[dict[str, Any]]


@dataclass(frozen=True)
class InputFileIdentity:
    """Stable identity captured before an evaluator input is consumed."""

    filename: str
    size_bytes: int
    sha256: str
    stat_signature: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ModelFilesIdentity:
    """Portable identity and measured transfer/storage totals for one model."""

    id: str
    revision: str
    manifest_sha256: str
    transport_size_bytes: int
    raw_size_bytes: int

    def to_report(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "transport_size_bytes": self.transport_size_bytes,
            "raw_size_bytes": self.raw_size_bytes,
        }


@dataclass(frozen=True)
class PreparedModelFiles:
    """Verified model manifest and decompressed assets in evaluator-owned storage."""

    directory: Path
    manifest: dict[str, Any]
    identity: ModelFilesIdentity


@dataclass(frozen=True)
class ReferenceGraphContract:
    """Lightweight FP32 facts retained after releasing an ONNX protobuf."""

    initializer_stats: dict[str, Any]
    io_contract: tuple[
        tuple[tuple[str, bytes], ...],
        tuple[tuple[str, bytes], ...],
    ]
    node_count: int


_MODEL_FILES_FORMAT = "ardy-browser-model-files"
_MODEL_FILES_SCHEMA_VERSION = 1
_MODEL_MANIFEST_FILE = "model.json.gz"
_RUNTIME_CONTRACT_REVISION = 3
_REQUIRED_WEBGPU_FEATURES = ["shader-f16"]
_EXPECTED_GRAPH_NAMES = ("text_encoder", "denoiser", "decoder")
_EXPECTED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
_EXPECTED_GRAPH_BINDINGS = {
    "text_encoder": {
        "inputs": {
            "inputIds": "input_ids",
            "attentionMask": "attention_mask",
            "tokenTypeIds": "token_type_ids",
        },
        "outputs": {"textConditions": "text_conditions"},
    },
    "denoiser": {
        "inputs": {
            "cfgWeight": "cfg_weight",
            "x": "x",
            "historyLength": "history_len",
            "generationLength": "generation_len",
            "historyMask": "history_mask",
            "generationMask": "generation_mask",
            "historyTokenMask": "history_token_mask",
            "generationTokenMask": "generation_token_mask",
            "textConditions": "text_conditions",
            "timestep": "timestep",
            "firstHeadingAngle": "first_heading_angle",
        },
        "outputs": {"predX0": "pred_x0"},
    },
    "decoder": {
        "inputs": {
            "hybridTokens": "hybrid_tokens",
            "motionPadMask": "motion_pad_mask",
            "globalTranslation": "global_translation",
        },
        "outputs": {
            "normalizedMotion": "normalized_motion",
            "posedJoints": "posed_joints",
            "localRotations": "local_rotations",
            "globalRotations": "global_rotations",
            "rootPositions": "root_positions",
            "footContacts": "foot_contacts",
            "globalRootHeading": "global_root_heading",
        },
    },
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_MAX_MODEL_FILES = 10_000
_MAX_MODEL_FILE_BYTES = 0x7FFF_FFFF
_MAX_MODEL_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_MODEL_PATH_BYTES = 4_096
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class PortableRandom:
    """Python port of web/src/runtime/random.ts."""

    _UINT32_MASK = 0xFFFF_FFFF
    _UINT32_RANGE = 0x1_0000_0000

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & self._UINT32_MASK
        self.spare_normal: float | None = None

    def next_uint32(self) -> int:
        self.state = (self.state + 0x6D2B_79F5) & self._UINT32_MASK
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & self._UINT32_MASK
        mixed = ((value ^ (value >> 7)) * (value | 61)) & self._UINT32_MASK
        value = (value ^ ((value + mixed) & self._UINT32_MASK)) & self._UINT32_MASK
        return (value ^ (value >> 14)) & self._UINT32_MASK

    def next_float(self) -> float:
        return (self.next_uint32() + 1) / (self._UINT32_RANGE + 1)

    def next_normal(self) -> float:
        if self.spare_normal is not None:
            result = self.spare_normal
            self.spare_normal = None
            return result
        while True:
            x = 2 * self.next_float() - 1
            y = 2 * self.next_float() - 1
            radius_squared = x * x + y * y
            if math.ulp(1.0) < radius_squared < 1:
                scale = math.sqrt((-2 * math.log(radius_squared)) / radius_squared)
                self.spare_normal = y * scale
                return x * scale

    def normal_array(self, shape: tuple[int, ...]) -> np.ndarray:
        result = np.empty(math.prod(shape), dtype=np.float32)
        for index in range(result.size):
            result[index] = self.next_normal()
        return result.reshape(shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired, deterministic browser-style DDIM rollouts through an "
            "FP32 reference model directory and a mixed-FP16 candidate model directory."
        )
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL records containing a `text` field.")
    parser.add_argument(
        "--prompt-metadata",
        type=Path,
        default=None,
        help=(
            "canonical NVIDIA Timeline prompt-provenance sidecar; inferred from "
            "PROMPTS when present and required by --public-output"
        ),
    )
    parser.add_argument("--split", default="test", help="Optional JSONL split to select; use `all` for every record.")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument(
        "--seeds",
        default="12031,987654,20260729",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument("--cfg-weight", type=float, default=3.5)
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Number of consecutive 40-frame browser windows per prompt/seed.",
    )
    parser.add_argument(
        "--initial-translation",
        default="1.25,0.2,-0.75",
        help="Comma-separated initial world x,y,z translation.",
    )
    parser.add_argument(
        "--initial-heading",
        type=float,
        default=0.35,
        help="Initial heading in radians; non-zero by default to exercise continuation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "local detailed report; includes worst-case prompt text and should "
            "remain in an ignored artifacts directory"
        ),
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=None,
        help=(
            "optional Git-safe aggregate report with prompt provenance and "
            "hashes, but no prompt text or absolute paths"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _capture_file_identity(path: Path) -> InputFileIdentity:
    """Hash one stable regular file and retain enough stat data for a recheck."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Evaluator input is not a regular file: {path}")
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    before_signature = _stat_signature(before)
    after_signature = _stat_signature(after)
    if before_signature != after_signature:
        raise RuntimeError(f"Evaluator input changed while it was being hashed: {path}")
    return InputFileIdentity(
        filename=path.name,
        size_bytes=after.st_size,
        sha256=digest,
        stat_signature=after_signature,
    )


def _verify_file_identity(path: Path, identity: InputFileIdentity) -> None:
    """Fail when an input changed between its prehash and its consumption."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Evaluator input changed after its identity was captured: {path}")
    current = path.stat()
    if _stat_signature(current) != identity.stat_signature or _sha256(path) != identity.sha256:
        raise RuntimeError(f"Evaluator input changed after its identity was captured: {path}")


def _read_stable_bytes(path: Path) -> tuple[bytes, InputFileIdentity]:
    """Read and identify the exact same file bytes."""

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Evaluator input is not a regular file: {path}")
    before = path.stat()
    encoded = path.read_bytes()
    after = path.stat()
    before_signature = _stat_signature(before)
    after_signature = _stat_signature(after)
    if before_signature != after_signature or len(encoded) != after.st_size:
        raise RuntimeError(f"Evaluator input changed while it was being read: {path}")
    return encoded, InputFileIdentity(
        filename=path.name,
        size_bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        stat_signature=after_signature,
    )


def _canonical_model_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path.")
    if (
        "\\" in value
        or _WINDOWS_DRIVE_PATTERN.match(value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or len(value.encode("utf-8")) > _MAX_MODEL_PATH_BYTES
    ):
        raise ValueError(f"{label} is not a safe canonical POSIX path: {value!r}.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is not a safe canonical POSIX path: {value!r}.")
    return value


def _regular_directory_files(directory: Path) -> set[str]:
    """Return canonical relative files while rejecting links and special entries."""

    if directory.is_symlink() or not directory.is_dir():
        raise NotADirectoryError(f"Model files input must be a directory: {directory}")
    files: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"Model files directory must not contain symbolic links: {relative}.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Model files directory contains a non-regular entry: {relative}.")
        canonical = _canonical_model_path(relative, "model files entry")
        if canonical in files:
            raise ValueError(f"Duplicate model files entry: {canonical!r}.")
        files.add(canonical)
    return files


def _session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Run this script with `uv run --extra browser`.") from error

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 8
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _create_runtime(directory: Path, manifest: dict[str, Any]) -> ModelFilesRuntime:
    """Create ORT sessions only after both models have passed static validation."""

    graphs = manifest["graphs"]
    return ModelFilesRuntime(
        directory=directory,
        manifest=manifest,
        text_encoder=_session(directory / graphs["text_encoder"]["model"]),
        denoiser=_session(directory / graphs["denoiser"]["model"]),
        decoder=_session(directory / graphs["decoder"]["model"]),
    )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _nonnegative_integer(value: Any, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer.")
    return value


def _sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _bounded_gzip_bytes(
    path: Path,
    *,
    limit: int,
    label: str,
    expected_identity: InputFileIdentity,
) -> bytes:
    """Read one stable gzip stream without allowing unbounded expansion."""

    decoded = bytearray()
    try:
        with gzip.open(path, mode="rb") as input_file:
            while chunk := input_file.read(min(1024 * 1024, limit + 1 - len(decoded))):
                decoded.extend(chunk)
                if len(decoded) > limit:
                    raise ValueError(f"{label} expands beyond {limit} bytes.")
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError(f"{label} is not a valid gzip stream.") from error
    _verify_file_identity(path, expected_identity)
    return bytes(decoded)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{_MODEL_MANIFEST_FILE} contains duplicate key {key!r}.")
        result[key] = value
    return result


def _read_model_manifest(
    source_directory: Path,
) -> tuple[dict[str, Any], InputFileIdentity, str, int]:
    manifest_path = source_directory / _MODEL_MANIFEST_FILE
    manifest_transport_identity = _capture_file_identity(manifest_path)
    if manifest_transport_identity.size_bytes > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"{_MODEL_MANIFEST_FILE} exceeds the compressed manifest size limit."
        )
    encoded = _bounded_gzip_bytes(
        manifest_path,
        limit=_MAX_MANIFEST_BYTES,
        label=_MODEL_MANIFEST_FILE,
        expected_identity=manifest_transport_identity,
    )
    try:
        manifest = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{_MODEL_MANIFEST_FILE} does not contain valid UTF-8 JSON.") from error
    checked_manifest = _object(manifest, "manifest")
    _validate_json_finite(checked_manifest, "manifest")
    return (
        checked_manifest,
        manifest_transport_identity,
        hashlib.sha256(encoded).hexdigest(),
        len(encoded),
    )


def _validate_model_file_descriptions(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], set[str], int, int]:
    if manifest.get("format") != _MODEL_FILES_FORMAT:
        raise ValueError(f"manifest.format must be {_MODEL_FILES_FORMAT!r}.")
    if manifest.get("schema_version") != _MODEL_FILES_SCHEMA_VERSION:
        raise ValueError(
            f"manifest.schema_version must be {_MODEL_FILES_SCHEMA_VERSION}."
        )
    model = _object(manifest.get("model"), "manifest.model")
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("manifest.model.id must be a non-empty string.")
    _sha256_value(model.get("revision"), "manifest.model.revision")

    raw_files = _object(manifest.get("files"), "manifest.files")
    if not raw_files or len(raw_files) > _MAX_MODEL_FILES:
        raise ValueError(
            f"manifest.files must declare between 1 and {_MAX_MODEL_FILES} files."
        )

    records: dict[str, dict[str, Any]] = {}
    transport_paths: set[str] = set()
    transport_total = 0
    raw_total = 0
    for raw_path, raw_record in raw_files.items():
        relative_path = _canonical_model_path(raw_path, "manifest.files key")
        if relative_path == _MODEL_MANIFEST_FILE:
            raise ValueError(f"manifest.files must not declare {_MODEL_MANIFEST_FILE}.")
        record = _object(raw_record, f"manifest.files.{relative_path}")
        raw_size = _nonnegative_integer(
            record.get("size_bytes"),
            f"manifest.files.{relative_path}.size_bytes",
        )
        if raw_size > _MAX_MODEL_FILE_BYTES:
            raise ValueError(
                f"manifest.files.{relative_path}.size_bytes exceeds the per-file limit."
            )
        _sha256_value(
            record.get("sha256"),
            f"manifest.files.{relative_path}.sha256",
        )
        if "media_type" in record and (
            not isinstance(record["media_type"], str) or not record["media_type"]
        ):
            raise ValueError(
                f"manifest.files.{relative_path}.media_type must be a non-empty string."
            )

        transport = _object(
            record.get("transport"),
            f"manifest.files.{relative_path}.transport",
        )
        transport_path = _canonical_model_path(
            transport.get("path"),
            f"manifest.files.{relative_path}.transport.path",
        )
        if transport_path == _MODEL_MANIFEST_FILE:
            raise ValueError(
                f"manifest.files.{relative_path}.transport.path must not be "
                f"{_MODEL_MANIFEST_FILE}."
            )
        if transport_path in transport_paths:
            raise ValueError(f"Duplicate model transport path: {transport_path!r}.")
        transport_paths.add(transport_path)
        if transport.get("compression") != "gzip":
            raise ValueError(
                f"manifest.files.{relative_path}.transport.compression must be 'gzip'."
            )
        transport_size = _nonnegative_integer(
            transport.get("size_bytes"),
            f"manifest.files.{relative_path}.transport.size_bytes",
            positive=True,
        )
        if transport_size > _MAX_MODEL_FILE_BYTES:
            raise ValueError(
                f"manifest.files.{relative_path}.transport.size_bytes exceeds "
                "the per-file limit."
            )
        _sha256_value(
            transport.get("sha256"),
            f"manifest.files.{relative_path}.transport.sha256",
        )
        transport_total += transport_size
        raw_total += raw_size
        if (
            transport_total > _MAX_MODEL_TOTAL_BYTES
            or raw_total > _MAX_MODEL_TOTAL_BYTES
        ):
            raise ValueError(
                f"Model files exceed the {_MAX_MODEL_TOTAL_BYTES}-byte total limit."
            )
        records[relative_path] = record
    return records, transport_paths, transport_total, raw_total


def _decompress_model_file(
    source: Path,
    destination: Path,
    *,
    record: dict[str, Any],
    relative_path: str,
) -> None:
    transport = record["transport"]
    identity = _capture_file_identity(source)
    if identity.size_bytes != transport["size_bytes"]:
        raise ValueError(f"Compressed size mismatch for {relative_path}.")
    if identity.sha256 != transport["sha256"]:
        raise ValueError(f"Compressed SHA-256 mismatch for {relative_path}.")

    expected_raw_size = record["size_bytes"]
    digest = hashlib.sha256()
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(source, mode="rb") as input_file, destination.open("xb") as output:
            while chunk := input_file.read(8 * 1024 * 1024):
                written += len(chunk)
                if written > expected_raw_size:
                    raise ValueError(
                        f"Decompressed size exceeds the manifest value for {relative_path}."
                    )
                digest.update(chunk)
                output.write(chunk)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Compressed model file is not valid gzip data: {relative_path}."
        ) from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    _verify_file_identity(source, identity)
    if written != expected_raw_size:
        raise ValueError(f"Decompressed size mismatch for {relative_path}.")
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"Decompressed SHA-256 mismatch for {relative_path}.")


def _prepare_model_files(
    source_directory: Path,
    output_directory: Path,
) -> PreparedModelFiles:
    """Validate compressed inputs and materialize verified raw assets."""

    if source_directory.is_symlink() or not source_directory.is_dir():
        raise NotADirectoryError(
            f"Model files input must be a directory: {source_directory}"
        )
    source_resolved = source_directory.resolve()
    output_resolved = output_directory.resolve()
    if (
        source_resolved == output_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError("Model input and evaluator output directories must not overlap.")
    if output_directory.exists():
        if output_directory.is_symlink() or not output_directory.is_dir():
            raise NotADirectoryError(
                f"Evaluator model output must be a directory: {output_directory}"
            )
        if any(output_directory.iterdir()):
            raise FileExistsError(
                f"Evaluator model output must be empty: {output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)

    (
        manifest,
        manifest_transport_identity,
        manifest_sha256,
        manifest_raw_size,
    ) = _read_model_manifest(source_directory)
    records, transport_paths, transport_total, raw_total = (
        _validate_model_file_descriptions(manifest)
    )
    actual_files = _regular_directory_files(source_directory)
    expected_files = {_MODEL_MANIFEST_FILE, *transport_paths}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(
            "Model directory entries do not match manifest transports "
            f"(missing={missing}, extra={extra})."
        )

    for relative_path, record in sorted(records.items()):
        transport_path = record["transport"]["path"]
        source = source_directory / Path(*PurePosixPath(transport_path).parts)
        destination = output_directory / Path(*PurePosixPath(relative_path).parts)
        _decompress_model_file(
            source,
            destination,
            record=record,
            relative_path=relative_path,
        )

    model = manifest["model"]
    return PreparedModelFiles(
        directory=output_directory,
        manifest=manifest,
        identity=ModelFilesIdentity(
            id=model["id"],
            revision=model["revision"],
            manifest_sha256=manifest_sha256,
            transport_size_bytes=(
                manifest_transport_identity.size_bytes + transport_total
            ),
            raw_size_bytes=manifest_raw_size + raw_total,
        ),
    )


def _fraction(value: Any, label: str, numerator: int, denominator: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite fraction.")
    expected = numerator / denominator
    if not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} does not match the declared byte reduction.")
    return float(value)


def _initializer_precision_stats(model: onnx.ModelProto) -> dict[str, Any]:
    element_bytes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }
    count_by_dtype: Counter[str] = Counter()
    bytes_by_dtype: Counter[str] = Counter()
    for initializer in model.graph.initializer:
        dtype = TensorProto.DataType.Name(initializer.data_type).lower()
        element_count = math.prod(initializer.dims)
        count_by_dtype[dtype] += 1
        bytes_by_dtype[dtype] += element_count * element_bytes.get(
            initializer.data_type,
            len(initializer.raw_data),
        )
    return {
        "count_by_dtype": dict(sorted(count_by_dtype.items())),
        "bytes_by_dtype": dict(sorted(bytes_by_dtype.items())),
        "total_count": sum(count_by_dtype.values()),
        "total_bytes": sum(bytes_by_dtype.values()),
    }


def _graph_io_types(values) -> dict[str, str]:
    return {
        value.name: TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower()
        for value in values
    }


def _graph_io_contract(model: onnx.ModelProto) -> tuple[tuple[tuple[str, bytes], ...], tuple[tuple[str, bytes], ...]]:
    return (
        tuple(
            (value.name, value.type.SerializeToString(deterministic=True))
            for value in model.graph.input
        ),
        tuple(
            (value.name, value.type.SerializeToString(deterministic=True))
            for value in model.graph.output
        ),
    )


def _load_checked_onnx(path: Path, graph_name: str) -> onnx.ModelProto:
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as error:
        raise ValueError(f"Unable to load {graph_name} ONNX graph: {path.name}.") from error
    validate_no_external_data(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def _validate_initializer_summary(
    value: Any,
    expected: dict[str, Any],
    label: str,
) -> None:
    summary = _object(value, label)
    if summary != expected:
        raise ValueError(f"{label} does not match the ONNX initializer payload.")


def _validate_manifest_files(
    directory: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    files = _object(manifest.get("files"), "manifest.files")
    if _MODEL_MANIFEST_FILE in files:
        raise ValueError(f"manifest.files must not declare {_MODEL_MANIFEST_FILE}.")
    resolved: dict[str, Path] = {}
    for raw_path, raw_record in files.items():
        relative_path = _canonical_model_path(raw_path, "manifest.files key")
        record = _object(raw_record, f"manifest.files.{relative_path}")
        expected_keys = {"sha256", "size_bytes", "transport"}
        if not expected_keys.issubset(record):
            raise ValueError(
                f"manifest.files.{relative_path} must declare sha256, size_bytes, "
                "and transport."
            )
        expected_size = _nonnegative_integer(
            record["size_bytes"],
            f"manifest.files.{relative_path}.size_bytes",
        )
        expected_hash = _sha256_value(
            record["sha256"],
            f"manifest.files.{relative_path}.sha256",
        )
        path = directory / Path(*PurePosixPath(relative_path).parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Declared model file is missing: {relative_path}.")
        if path.stat().st_size != expected_size:
            raise ValueError(f"Model file size mismatch for {relative_path}.")
        if _sha256(path) != expected_hash:
            raise ValueError(f"Model file SHA-256 mismatch for {relative_path}.")
        resolved[relative_path] = path
    actual_paths = _regular_directory_files(directory)
    expected_paths = set(resolved)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            "Decompressed model files do not match manifest.files "
            f"(missing={missing}, extra={extra})."
        )
    return resolved


def _validate_common_manifest(
    directory: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path]]:
    manifest = _object(manifest, "manifest")
    if manifest.get("format") != _MODEL_FILES_FORMAT:
        raise ValueError(f"manifest.format must be {_MODEL_FILES_FORMAT!r}.")
    if manifest.get("schema_version") != _MODEL_FILES_SCHEMA_VERSION:
        raise ValueError(
            f"manifest.schema_version must be {_MODEL_FILES_SCHEMA_VERSION}."
        )
    files = _validate_manifest_files(directory, manifest)

    tokenizer = _object(manifest.get("tokenizer"), "manifest.tokenizer")
    tokenizer_directory = _canonical_model_path(
        tokenizer.get("directory"),
        "manifest.tokenizer.directory",
    )
    _nonnegative_integer(
        tokenizer.get("max_length"),
        "manifest.tokenizer.max_length",
        positive=True,
    )
    required_tokenizer_paths = {
        f"{tokenizer_directory}/{filename}" for filename in _EXPECTED_TOKENIZER_FILES
    }
    if not required_tokenizer_paths.issubset(files):
        raise ValueError(
            "Browser model is missing one or more required tokenizer payloads."
        )

    graphs = _object(manifest.get("graphs"), "manifest.graphs")
    if set(graphs) != set(_EXPECTED_GRAPH_NAMES):
        raise ValueError(
            f"manifest.graphs must contain exactly {list(_EXPECTED_GRAPH_NAMES)}."
        )
    graph_paths: dict[str, Path] = {}
    graph_asset_names: set[str] = set()
    for graph_name in _EXPECTED_GRAPH_NAMES:
        graph = _object(graphs[graph_name], f"manifest.graphs.{graph_name}")
        if "external_data" in graph and graph["external_data"] != []:
            raise ValueError(
                f"manifest.graphs.{graph_name} must not declare external ONNX data."
            )
        model_path = _canonical_model_path(
            graph.get("model"),
            f"manifest.graphs.{graph_name}.model",
        )
        if model_path not in files:
            raise ValueError(
                f"manifest.graphs.{graph_name}.model is not declared in manifest.files."
            )
        graph_paths[graph_name] = files[model_path]
        graph_asset_names.add(model_path)
        expected_bindings = _EXPECTED_GRAPH_BINDINGS[graph_name]
        for binding_kind in ("inputs", "outputs"):
            bindings = _object(
                graph.get(binding_kind),
                f"manifest.graphs.{graph_name}.{binding_kind}",
            )
            if bindings != expected_bindings[binding_kind]:
                raise ValueError(
                    f"manifest.graphs.{graph_name}.{binding_kind} does not "
                    "match the fixed browser evaluator contract."
                )
    if set(files) != required_tokenizer_paths | graph_asset_names:
        raise ValueError("manifest.files contains unreferenced browser model assets.")

    runtime = _object(manifest.get("runtime"), "manifest.runtime")
    if runtime.get("contract_revision") != _RUNTIME_CONTRACT_REVISION:
        raise ValueError(
            f"manifest.runtime.contract_revision must be {_RUNTIME_CONTRACT_REVISION}."
        )
    if runtime.get("required_webgpu_features") != _REQUIRED_WEBGPU_FEATURES:
        raise ValueError(
            "manifest.runtime.required_webgpu_features must be ['shader-f16']."
        )
    if runtime.get("text_only") is not True:
        raise ValueError("manifest.runtime.text_only must be true.")

    dimensions = _object(manifest.get("dimensions"), "manifest.dimensions")
    required_dimensions = (
        "fps",
        "num_frames_per_token",
        "max_tokens",
        "max_frames",
        "generation_tokens",
        "generation_frames",
        "history_tokens",
        "history_frames",
        "root_features_per_frame",
        "nframe_root_dim",
        "latent_dim",
        "hybrid_dim",
        "motion_dim",
        "body_dim",
        "text_condition_dim",
        "num_joints",
    )
    dimension_values = {
        key: _nonnegative_integer(
            dimensions.get(key),
            f"manifest.dimensions.{key}",
            positive=True,
        )
        for key in required_dimensions
    }
    if dimension_values["max_frames"] != (
        dimension_values["max_tokens"] * dimension_values["num_frames_per_token"]
    ):
        raise ValueError("manifest.dimensions.max_frames is inconsistent.")
    if dimension_values["generation_frames"] != (
        dimension_values["generation_tokens"]
        * dimension_values["num_frames_per_token"]
    ):
        raise ValueError("manifest.dimensions.generation_frames is inconsistent.")
    if dimension_values["history_tokens"] != (
        dimension_values["max_tokens"] - dimension_values["generation_tokens"]
    ):
        raise ValueError("manifest.dimensions.history_tokens is inconsistent.")
    if dimension_values["history_frames"] != (
        dimension_values["history_tokens"]
        * dimension_values["num_frames_per_token"]
    ):
        raise ValueError("manifest.dimensions.history_frames is inconsistent.")
    if dimension_values["nframe_root_dim"] != (
        dimension_values["root_features_per_frame"]
        * dimension_values["num_frames_per_token"]
    ):
        raise ValueError("manifest.dimensions.nframe_root_dim is inconsistent.")
    if dimension_values["hybrid_dim"] != (
        dimension_values["nframe_root_dim"] + dimension_values["latent_dim"]
    ):
        raise ValueError("manifest.dimensions.hybrid_dim is inconsistent.")
    if dimension_values["text_condition_dim"] != 2048:
        raise ValueError("manifest.dimensions.text_condition_dim must be 2048.")

    generation = _object(manifest.get("generation"), "manifest.generation")
    if generation.get("min_frames") != dimension_values["generation_frames"]:
        raise ValueError("manifest.generation.min_frames is inconsistent.")
    _nonnegative_integer(
        generation.get("max_frames"),
        "manifest.generation.max_frames",
        positive=True,
    )
    if generation.get("denoising_steps") != 10:
        raise ValueError("manifest.generation.denoising_steps must be 10.")
    return manifest, files, graph_paths


def _validate_fp32_precision(
    manifest: dict[str, Any],
    graph_paths: dict[str, Path],
) -> dict[str, ReferenceGraphContract]:
    precision = _object(manifest.get("precision"), "reference.precision")
    if precision.get("format") != "fp32":
        raise ValueError("Reference model must declare precision.format='fp32'.")
    if precision.get("public_io_dtype") != "float32":
        raise ValueError("Reference precision.public_io_dtype must be 'float32'.")
    if precision.get("required_webgpu_features") != []:
        raise ValueError("Reference precision.required_webgpu_features must be empty.")
    toolchain = _object(precision.get("toolchain"), "reference.precision.toolchain")
    for package in ("torch", "onnx", "onnxruntime"):
        if not isinstance(toolchain.get(package), str) or not toolchain[package]:
            raise ValueError(
                f"reference.precision.toolchain.{package} must be a non-empty string."
            )
    summaries = _object(precision.get("graphs"), "reference.precision.graphs")
    if set(summaries) != set(_EXPECTED_GRAPH_NAMES):
        raise ValueError("Reference precision.graphs must contain exactly three graphs.")

    contracts: dict[str, ReferenceGraphContract] = {}
    total_bytes = 0
    for graph_name in _EXPECTED_GRAPH_NAMES:
        graph_path = graph_paths[graph_name]
        summary = _object(
            summaries[graph_name],
            f"reference.precision.graphs.{graph_name}",
        )
        if summary.get("model") != manifest["graphs"][graph_name]["model"]:
            raise ValueError(
                f"Reference precision graph path differs for {graph_name}."
            )
        expected_size = graph_path.stat().st_size
        if summary.get("size_bytes") != expected_size:
            raise ValueError(f"Reference precision size differs for {graph_name}.")
        if summary.get("sha256") != _sha256(graph_path):
            raise ValueError(f"Reference precision SHA-256 differs for {graph_name}.")
        model = _load_checked_onnx(graph_path, graph_name)
        validate_fp32_source_graph(model, graph_name)
        validate_production_policy_coverage(model, graph_name)
        contracts[graph_name] = ReferenceGraphContract(
            initializer_stats=_initializer_precision_stats(model),
            io_contract=_graph_io_contract(model),
            node_count=len(model.graph.node),
        )
        total_bytes += expected_size
    if precision.get("onnx_bytes") != total_bytes:
        raise ValueError("Reference precision.onnx_bytes does not match graph sizes.")
    return contracts


def _validate_mixed_precision(
    manifest: dict[str, Any],
    graph_paths: dict[str, Path],
    reference_contracts: dict[str, ReferenceGraphContract],
    reference_graph_paths: dict[str, Path],
) -> None:
    precision = _object(manifest.get("precision"), "candidate.precision")
    if precision.get("format") != "mixed-fp16":
        raise ValueError("Candidate model must declare precision.format='mixed-fp16'.")
    if precision.get("policy_version") != MIXED_FP16_POLICY_VERSION:
        raise ValueError(
            f"Candidate precision.policy_version must be {MIXED_FP16_POLICY_VERSION}."
        )
    if precision.get("public_io_dtype") != "float32":
        raise ValueError("Candidate precision.public_io_dtype must be 'float32'.")
    if precision.get("required_webgpu_features") != _REQUIRED_WEBGPU_FEATURES:
        raise ValueError(
            "Candidate precision.required_webgpu_features must be ['shader-f16']."
        )
    toolchain = _object(precision.get("toolchain"), "candidate.precision.toolchain")
    for package in ("torch", "onnx", "onnxruntime"):
        if not isinstance(toolchain.get(package), str) or not toolchain[package]:
            raise ValueError(
                f"candidate.precision.toolchain.{package} must be a non-empty string."
            )
    summaries = _object(precision.get("graphs"), "candidate.precision.graphs")
    if set(summaries) != set(_EXPECTED_GRAPH_NAMES):
        raise ValueError("Candidate precision.graphs must contain exactly three graphs.")

    source_total = 0
    output_total = 0
    saved_total = 0
    for graph_name, policy in MIXED_FP16_POLICIES.items():
        summary = _object(
            summaries[graph_name],
            f"candidate.precision.graphs.{graph_name}",
        )
        label = f"candidate.precision.graphs.{graph_name}"
        if summary.get("schema_version") != 1:
            raise ValueError(f"{label}.schema_version must be 1.")
        if summary.get("graph_name") != graph_name:
            raise ValueError(f"{label}.graph_name must be {graph_name!r}.")
        if summary.get("policy_id") != policy.policy_id:
            raise ValueError(f"{label}.policy_id must be {policy.policy_id!r}.")
        if summary.get("conversion_mode") != policy.conversion_mode:
            raise ValueError(
                f"{label}.conversion_mode must be {policy.conversion_mode!r}."
            )

        source_path = reference_graph_paths[graph_name]
        output_path = graph_paths[graph_name]
        source_size = source_path.stat().st_size
        output_size = output_path.stat().st_size
        source_hash = _sha256(source_path)
        output_hash = _sha256(output_path)
        if summary.get("source_sha256") != source_hash:
            raise ValueError(f"{label}.source_sha256 differs from the FP32 graph.")
        if summary.get("output_sha256") != output_hash:
            raise ValueError(f"{label}.output_sha256 differs from the candidate graph.")
        if summary.get("source_size_bytes") != source_size:
            raise ValueError(f"{label}.source_size_bytes differs from the FP32 graph.")
        if summary.get("output_size_bytes") != output_size:
            raise ValueError(f"{label}.output_size_bytes differs from the candidate graph.")
        reduction = source_size - output_size
        if reduction < 0 or summary.get("size_reduction_bytes") != reduction:
            raise ValueError(f"{label}.size_reduction_bytes is inconsistent.")
        _fraction(
            summary.get("size_reduction_fraction"),
            f"{label}.size_reduction_fraction",
            reduction,
            source_size,
        )

        source_contract = reference_contracts[graph_name]
        output_model = _load_checked_onnx(output_path, graph_name)
        if source_contract.io_contract != _graph_io_contract(output_model):
            raise ValueError(f"Candidate {graph_name} changed the public ONNX I/O contract.")
        output_stats = _initializer_precision_stats(output_model)
        _validate_initializer_summary(
            summary.get("source_initializers"),
            source_contract.initializer_stats,
            f"{label}.source_initializers",
        )
        _validate_initializer_summary(
            summary.get("output_initializers"),
            output_stats,
            f"{label}.output_initializers",
        )
        if summary.get("source_node_count") != source_contract.node_count:
            raise ValueError(f"{label}.source_node_count is inconsistent.")
        if summary.get("output_node_count") != len(output_model.graph.node):
            raise ValueError(f"{label}.output_node_count is inconsistent.")
        cast_count = sum(node.op_type == "Cast" for node in output_model.graph.node)
        if summary.get("output_cast_node_count") != cast_count:
            raise ValueError(f"{label}.output_cast_node_count is inconsistent.")
        if summary.get("graph_inputs") != _graph_io_types(output_model.graph.input):
            raise ValueError(f"{label}.graph_inputs is inconsistent.")
        if summary.get("graph_outputs") != _graph_io_types(output_model.graph.output):
            raise ValueError(f"{label}.graph_outputs is inconsistent.")

        if policy.conversion_mode == "fp32-identity":
            if source_hash != output_hash or reduction != 0:
                raise ValueError(
                    f"Candidate {graph_name} must be byte-identical to its FP32 reference."
                )
            validate_fp32_source_graph(output_model, graph_name)
        else:
            if reduction == 0:
                raise ValueError(f"Candidate {graph_name} mixed conversion must reduce size.")
            if output_stats["count_by_dtype"].get("float16", 0) == 0:
                raise ValueError(
                    f"Candidate {graph_name} must contain FP16 initializers."
                )
            if output_stats["count_by_dtype"].get("bfloat16", 0):
                raise ValueError(
                    f"Candidate {graph_name} must not contain BF16 initializers."
                )
            validate_no_storage_only_fp16_casts(output_model, graph_name)
        source_total += source_size
        output_total += output_size
        saved_total += reduction

    if precision.get("source_onnx_bytes") != source_total:
        raise ValueError("Candidate precision.source_onnx_bytes is inconsistent.")
    if precision.get("mixed_onnx_bytes") != output_total:
        raise ValueError("Candidate precision.mixed_onnx_bytes is inconsistent.")
    if precision.get("saved_onnx_bytes") != saved_total:
        raise ValueError("Candidate precision.saved_onnx_bytes is inconsistent.")
    _fraction(
        precision.get("saved_onnx_fraction"),
        "candidate.precision.saved_onnx_fraction",
        saved_total,
        source_total,
    )


def _validate_model_pair_before_sessions(
    reference: PreparedModelFiles,
    candidate: PreparedModelFiles,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate payloads and precision semantics before ONNX Runtime sees bytes."""

    reference_manifest, _, reference_graph_paths = _validate_common_manifest(
        reference.directory,
        reference.manifest,
    )
    candidate_manifest, _, candidate_graph_paths = _validate_common_manifest(
        candidate.directory,
        candidate.manifest,
    )
    reference_contracts = _validate_fp32_precision(
        reference_manifest,
        reference_graph_paths,
    )
    _validate_mixed_precision(
        candidate_manifest,
        candidate_graph_paths,
        reference_contracts,
        reference_graph_paths,
    )
    reference_placeholder = ModelFilesRuntime(
        directory=reference.directory,
        manifest=reference_manifest,
        text_encoder=None,
        denoiser=None,
        decoder=None,
    )
    candidate_placeholder = ModelFilesRuntime(
        directory=candidate.directory,
        manifest=candidate_manifest,
        text_encoder=None,
        denoiser=None,
        decoder=None,
    )
    precision_validation = _validate_compatible_models(
        reference_placeholder,
        candidate_placeholder,
    )
    return reference_manifest, candidate_manifest, precision_validation


def _without_precision_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic model contract shared by FP32 and mixed models."""
    contract = copy.deepcopy(manifest)
    for key in ("files", "precision", "verification"):
        contract.pop(key, None)
    return contract


def _first_difference(reference: Any, candidate: Any, path: str = "manifest") -> str | None:
    if type(reference) is not type(candidate):
        return f"{path} has different types ({type(reference).__name__} vs {type(candidate).__name__})"
    if isinstance(reference, dict):
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            return f"{path} keys differ (missing={missing}, extra={extra})"
        for key in sorted(reference):
            difference = _first_difference(reference[key], candidate[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            return f"{path} lengths differ ({len(reference)} vs {len(candidate)})"
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            difference = _first_difference(reference_item, candidate_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if reference != candidate:
        return f"{path} differs ({reference!r} vs {candidate!r})"
    return None


def _tokenizer_payloads(runtime: ModelFilesRuntime) -> dict[str, str]:
    directory = str(runtime.manifest["tokenizer"]["directory"])
    payloads: dict[str, str] = {}
    for relative_path in sorted(runtime.manifest["files"]):
        if relative_path.startswith(f"{directory}/"):
            payloads[relative_path] = _sha256(runtime.directory / relative_path)
    return payloads


def _validate_compatible_models(
    reference: ModelFilesRuntime,
    candidate: ModelFilesRuntime,
) -> dict[str, Any]:
    """Reject comparisons that change anything except ONNX precision."""
    reference_precision = reference.manifest.get("precision")
    if (
        not isinstance(reference_precision, dict)
        or reference_precision.get("format") != "fp32"
    ):
        raise ValueError("Reference model must declare precision.format='fp32'.")
    difference = _first_difference(
        _without_precision_metadata(reference.manifest),
        _without_precision_metadata(candidate.manifest),
    )
    if difference is not None:
        raise ValueError(f"Reference and candidate non-precision contracts differ: {difference}.")

    reference_tokenizer = _tokenizer_payloads(reference)
    candidate_tokenizer = _tokenizer_payloads(candidate)
    if reference_tokenizer != candidate_tokenizer:
        raise ValueError("Reference and candidate tokenizer payloads differ.")

    precision = candidate.manifest.get("precision")
    if not isinstance(precision, dict) or precision.get("format") != "mixed-fp16":
        raise ValueError("Candidate model must declare precision.format='mixed-fp16'.")
    if precision.get("policy_version") != MIXED_FP16_POLICY_VERSION:
        raise ValueError(
            "Candidate model must declare "
            f"precision.policy_version={MIXED_FP16_POLICY_VERSION}."
        )
    summaries = precision.get("graphs")
    if not isinstance(summaries, dict) or set(summaries) != set(MIXED_FP16_POLICIES):
        raise ValueError(
            "Candidate model precision.graphs must contain exactly the three "
            "production graphs."
        )

    reference_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    policy_ids: dict[str, str] = {}
    conversion_modes: dict[str, str] = {}
    for graph_name, policy in MIXED_FP16_POLICIES.items():
        summary = summaries.get(graph_name)
        if not isinstance(summary, dict):
            raise TypeError(f"Candidate precision summary for {graph_name} is missing.")
        if summary.get("policy_id") != policy.policy_id:
            raise ValueError(f"Candidate {graph_name} policy_id must be {policy.policy_id!r}.")
        if summary.get("conversion_mode") != policy.conversion_mode:
            raise ValueError(f"Candidate {graph_name} conversion_mode must be {policy.conversion_mode!r}.")

        reference_path = reference.directory / reference.manifest["graphs"][graph_name]["model"]
        candidate_path = candidate.directory / candidate.manifest["graphs"][graph_name]["model"]
        reference_hash = _sha256(reference_path)
        candidate_hash = _sha256(candidate_path)
        reference_hashes[graph_name] = reference_hash
        candidate_hashes[graph_name] = candidate_hash
        policy_ids[graph_name] = policy.policy_id
        conversion_modes[graph_name] = policy.conversion_mode
        if summary.get("source_sha256") != reference_hash:
            raise ValueError(f"Candidate {graph_name} source_sha256 does not match the FP32 reference graph.")
        if summary.get("output_sha256") != candidate_hash:
            raise ValueError(f"Candidate {graph_name} output_sha256 does not match the candidate graph.")
        if summary.get("source_size_bytes") != reference_path.stat().st_size:
            raise ValueError(f"Candidate {graph_name} source_size_bytes does not match the FP32 reference graph.")
        if summary.get("output_size_bytes") != candidate_path.stat().st_size:
            raise ValueError(f"Candidate {graph_name} output_size_bytes does not match the candidate graph.")
        if policy.conversion_mode == "fp32-identity" and reference_hash != candidate_hash:
            raise ValueError(f"Candidate {graph_name} must be byte-identical to the FP32 reference {graph_name}.")
        expected_reduction = reference_path.stat().st_size - candidate_path.stat().st_size
        if summary.get("size_reduction_bytes") != expected_reduction:
            raise ValueError(f"Candidate {graph_name} size_reduction_bytes does not match the paired graph sizes.")
        if policy.conversion_mode == "fp32-identity":
            if summary.get("size_reduction_bytes") != 0:
                raise ValueError(f"Candidate {graph_name} size_reduction_bytes must be zero.")
            output_initializers = summary.get("output_initializers")
            counts = output_initializers.get("count_by_dtype") if isinstance(output_initializers, dict) else None
            if not isinstance(counts, dict) or any(counts.get(dtype, 0) for dtype in ("float16", "bfloat16")):
                raise ValueError(f"Candidate {graph_name} must not contain reduced-precision initializers.")

    return {
        "candidate_precision_format": precision["format"],
        "candidate_policy_version": precision["policy_version"],
        "graph_policy_ids": policy_ids,
        "graph_conversion_modes": conversion_modes,
        "reference_graph_sha256": reference_hashes,
        "candidate_graph_sha256": candidate_hashes,
        "identity_graphs_byte_identical": sorted(
            graph_name
            for graph_name, policy in MIXED_FP16_POLICIES.items()
            if policy.conversion_mode == "fp32-identity"
        ),
    }


def _contract_fingerprint(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        _without_precision_metadata(manifest),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_prompt_records(path: Path, encoded: bytes) -> list[dict[str, Any]]:
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: prompt manifest is not valid UTF-8.") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL row.")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON.") from error
        if not isinstance(record, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object.")
        if not isinstance(record.get("text"), str) or not record["text"]:
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string.")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: prompt manifest is empty.")
    return records


def _select_prompts(
    path: Path,
    split: str,
    count: int,
    *,
    encoded: bytes | None = None,
) -> list[str]:
    if count <= 0:
        raise ValueError("--count must be positive.")
    if encoded is None:
        encoded, _ = _read_stable_bytes(path)
    records = _decode_prompt_records(path, encoded)
    eligible = [str(record["text"]) for record in records if split == "all" or record.get("split") == split]
    unique = list(dict.fromkeys(eligible))
    if len(unique) < count:
        raise ValueError(f"Only {len(unique)} unique prompts are available for split {split!r}; {count} requested.")
    indices = np.linspace(0, len(unique) - 1, count, dtype=np.int64)
    return [unique[int(index)] for index in indices]


def _prompt_manifest_identity(
    path: Path,
    metadata_path: Path | None,
    *,
    manifest_bytes: bytes | None = None,
    manifest_identity: InputFileIdentity | None = None,
    metadata_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return portable prompt-corpus identity, validating Timeline provenance."""

    if manifest_bytes is None:
        manifest_bytes, captured_identity = _read_stable_bytes(path)
        manifest_identity = captured_identity
    elif manifest_identity is None:
        manifest_identity = InputFileIdentity(
            filename=path.name,
            size_bytes=len(manifest_bytes),
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            stat_signature=(0, 0, len(manifest_bytes), 0, 0),
        )
    manifest_sha256 = manifest_identity.sha256
    identity: dict[str, Any] = {
        "filename": manifest_identity.filename,
        "size_bytes": manifest_identity.size_bytes,
        "sha256": manifest_sha256,
    }
    if metadata_path is None:
        return identity

    if metadata_bytes is None:
        metadata_bytes, _ = _read_stable_bytes(metadata_path)
    encoded_metadata = metadata_bytes
    try:
        provenance = json.loads(encoded_metadata)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{metadata_path}: invalid prompt provenance JSON.") from error

    records = _decode_prompt_records(path, manifest_bytes)
    split_counts = Counter(record.get("split") for record in records)
    expected_split_counts = {
        split: split_counts.get(split, 0) for split in ("train", "val", "test")
    }
    validate_prompt_provenance(
        provenance,
        expected_manifest_sha256=manifest_sha256,
        expected_manifest_filename=path.name,
        expected_count=len(records),
        expected_split_counts=expected_split_counts,
    )
    metadata_sha256 = hashlib.sha256(encoded_metadata).hexdigest()
    canonical_sha256 = prompt_provenance_sha256(provenance)
    if metadata_sha256 != canonical_sha256:
        raise ValueError(
            f"{metadata_path}: prompt provenance JSON is not canonically encoded "
            f"(canonical SHA-256 {canonical_sha256}, actual {metadata_sha256})."
        )
    identity["provenance_sidecar"] = {
        "filename": metadata_path.name,
        "size_bytes": len(encoded_metadata),
        "sha256": metadata_sha256,
        "content": provenance,
    }
    return identity


def _encode(runtime: ModelFilesRuntime, tokenizer, prompt: str) -> np.ndarray:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=runtime.manifest["tokenizer"]["max_length"],
        return_tensors="np",
    )
    encoded.setdefault("token_type_ids", np.zeros_like(encoded["input_ids"]))
    feeds = {name: encoded[name].astype(np.int64) for name in ("input_ids", "attention_mask", "token_type_ids")}
    conditions = runtime.text_encoder.run(None, feeds)[0].astype(np.float32)
    _require_finite_array(conditions, "text encoder output")
    return conditions


def _parse_initial_translation(value: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--initial-translation must contain three finite numbers.") from error
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError("--initial-translation must contain three finite numbers.")
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("--initial-translation values must be representable as float32.")
    return result


def _root_offset(
    dimensions: dict[str, int],
    frame: int,
    feature: int,
) -> tuple[int, int, int]:
    frames_per_token = dimensions["num_frames_per_token"]
    token = frame // frames_per_token
    frame_in_token = frame % frames_per_token
    return 0, token, frame_in_token * dimensions["root_features_per_frame"] + feature


def _raw_root(
    hybrid: np.ndarray,
    manifest: dict[str, Any],
    frame: int,
    feature: int,
) -> float:
    recenter = manifest["recenter"]
    normalized = float(hybrid[_root_offset(manifest["dimensions"], frame, feature)])
    return normalized * float(recenter["root_std"][feature]) + float(recenter["root_mean"][feature])


def _prepare_history(
    state: RolloutState,
    manifest: dict[str, Any],
) -> PreparedHistory:
    """Mirror BrowserArdyGenerationSession.#prepareHistory for a full window."""
    dimensions = manifest["dimensions"]
    recenter = manifest["recenter"]
    frames_per_token = dimensions["num_frames_per_token"]
    available_tokens = state.global_hybrid.shape[1]
    requested_tokens = min(dimensions["history_tokens"], available_tokens)
    if requested_tokens == 0:
        return PreparedHistory(
            history=None,
            history_tokens=0,
            history_frames=0,
            global_translation=state.initial_translation.copy(),
            first_heading_angle=state.initial_heading,
        )

    history = state.global_hybrid[:, available_tokens - requested_tokens :].copy()
    history_frames = requested_tokens * frames_per_token
    last_frame = history_frames - 1
    position_x, position_y, position_z = recenter["position_indices"]
    heading_cos, heading_sin = recenter["heading_indices"]
    center_x = _raw_root(history, manifest, last_frame, position_x)
    center_z = _raw_root(history, manifest, last_frame, position_z)
    first_cos = _raw_root(history, manifest, 0, heading_cos)
    first_sin = _raw_root(history, manifest, 0, heading_sin)

    for frame in range(history_frames):
        for feature, center in (
            (position_x, center_x),
            (position_y, float(state.initial_translation[1])),
            (position_z, center_z),
        ):
            translated = _raw_root(history, manifest, frame, feature) - center
            history[_root_offset(dimensions, frame, feature)] = np.float32(
                (translated - float(recenter["root_mean"][feature])) / float(recenter["root_std"][feature])
            )

    return PreparedHistory(
        history=history,
        history_tokens=requested_tokens,
        history_frames=history_frames,
        global_translation=np.asarray(
            [center_x, float(state.initial_translation[1]), center_z],
            dtype=np.float32,
        ),
        first_heading_angle=math.atan2(first_sin, first_cos),
    )


def _make_window(
    dimensions: dict[str, int],
    prepared: PreparedHistory,
    noise: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    generation_tokens = dimensions["generation_tokens"]
    generation_frames = dimensions["generation_frames"]
    generation_token_offset = prepared.history_tokens
    sample = np.zeros(
        (1, dimensions["max_tokens"], dimensions["hybrid_dim"]),
        dtype=np.float32,
    )
    if prepared.history is not None:
        sample[:, : prepared.history_tokens] = prepared.history
    sample[:, generation_token_offset : generation_token_offset + generation_tokens] = noise

    history_mask = np.zeros((1, dimensions["max_frames"]), dtype=np.float32)
    generation_mask = np.zeros_like(history_mask)
    history_mask[:, : prepared.history_frames] = 1
    generation_mask[:, prepared.history_frames : prepared.history_frames + generation_frames] = 1
    history_token_mask = np.zeros((1, dimensions["max_tokens"]), dtype=np.float32)
    generation_token_mask = np.zeros_like(history_token_mask)
    history_token_mask[:, : prepared.history_tokens] = 1
    generation_token_mask[:, generation_token_offset : generation_token_offset + generation_tokens] = 1
    return sample, {
        "history_mask": history_mask,
        "generation_mask": generation_mask,
        "history_token_mask": history_token_mask,
        "generation_token_mask": generation_token_mask,
    }


def _denoise(
    runtime: ModelFilesRuntime,
    sample: np.ndarray,
    text_conditions: np.ndarray,
    cfg_weight: float,
    prepared: PreparedHistory,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    dimensions = runtime.manifest["dimensions"]
    diffusion = runtime.manifest["diffusion"]
    sample = sample.copy()
    history_frames = prepared.history_frames
    generation_frames = dimensions["generation_frames"]
    generation_tokens = dimensions["generation_tokens"]
    generation_token_offset = prepared.history_tokens
    timestep = np.zeros(1, dtype=np.int64)
    feeds = {
        "cfg_weight": np.asarray([cfg_weight], dtype=np.float32),
        "x": sample,
        "history_len": np.asarray([history_frames], dtype=np.int64),
        "generation_len": np.asarray([generation_frames], dtype=np.int64),
        **masks,
        "text_conditions": text_conditions,
        "timestep": timestep,
        "first_heading_angle": np.asarray([prepared.first_heading_angle], dtype=np.float32),
    }

    for step in diffusion["timesteps"]:
        timestep[0] = step
        prediction = runtime.denoiser.run(None, feeds)[0]
        _require_finite_array(prediction, f"denoiser output at timestep {step}")
        alpha = float(diffusion["alphas_cumprod"][step])
        alpha_previous = float(diffusion["alphas_cumprod_prev"][step])
        sqrt_alpha = math.sqrt(alpha)
        sqrt_reciprocal_minus_one = math.sqrt(1 / alpha - 1)
        sqrt_previous = math.sqrt(alpha_previous)
        sqrt_one_minus_previous = math.sqrt(1 - alpha_previous)
        generation_slice = slice(generation_token_offset, generation_token_offset + generation_tokens)
        current = sample[:, generation_slice].astype(np.float64)
        predicted = prediction[:, generation_slice].astype(np.float64)
        if sqrt_reciprocal_minus_one == 0:
            epsilon = np.zeros_like(current)
        else:
            epsilon = (current / sqrt_alpha - predicted) / sqrt_reciprocal_minus_one
        sample[:, generation_slice] = (predicted * sqrt_previous + epsilon * sqrt_one_minus_previous).astype(np.float32)
        _require_finite_array(sample[:, generation_slice], f"DDIM state at timestep {step}")
    return sample


def _recenter_and_requantize(
    sample: np.ndarray,
    manifest: dict[str, Any],
    prepared: PreparedHistory,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mirror recenterAndRequantize for initial and continuation windows."""
    dimensions = manifest["dimensions"]
    recenter = manifest["recenter"]
    quantization = manifest["latent_quantization"]
    result = sample.copy()
    valid_tokens = prepared.history_tokens + dimensions["generation_tokens"]
    frames_per_token = dimensions["num_frames_per_token"]
    valid_frames = valid_tokens * frames_per_token
    root_token_dim = dimensions["nframe_root_dim"]
    position_x, _, position_z = recenter["position_indices"]
    heading_cos, heading_sin = recenter["heading_indices"]

    center_frame = valid_frames - 1
    center = np.asarray(
        [
            _raw_root(result, manifest, center_frame, position_x),
            0.0,
            _raw_root(result, manifest, center_frame, position_z),
        ],
        dtype=np.float32,
    )
    for frame in range(valid_frames):
        for feature, center_value in ((position_x, center[0]), (position_z, center[2])):
            translated = _raw_root(result, manifest, frame, feature) - float(center_value)
            result[_root_offset(dimensions, frame, feature)] = np.float32(
                (translated - float(recenter["root_mean"][feature])) / float(recenter["root_std"][feature])
            )

    latent = result[:, :valid_tokens, root_token_dim:]
    means = np.asarray(quantization["mean"], dtype=np.float64)
    stds = np.asarray(quantization["std"], dtype=np.float64)
    levels = np.asarray(quantization["levels"], dtype=np.int64)
    raw = latent.astype(np.float64) * stds + means
    half_width = levels // 2
    discrete = np.rint(np.clip(raw, -1, 1) * half_width) / half_width
    result[:, :valid_tokens, root_token_dim:] = ((discrete - means) / stds).astype(np.float32)
    next_history_start = max(0, valid_tokens - dimensions["history_tokens"])
    first_history_frame = next_history_start * frames_per_token
    first_cos = _raw_root(result, manifest, first_history_frame, heading_cos)
    first_sin = _raw_root(result, manifest, first_history_frame, heading_sin)
    world_translation = (prepared.global_translation.astype(np.float64) + center.astype(np.float64)).astype(np.float32)
    return result, world_translation, math.atan2(first_sin, first_cos)


def _decode(
    runtime: ModelFilesRuntime,
    sample: np.ndarray,
    global_translation: np.ndarray,
    valid_tokens: int,
) -> dict[str, np.ndarray]:
    dimensions = runtime.manifest["dimensions"]
    motion_mask = np.zeros((1, dimensions["max_frames"]), dtype=np.float32)
    motion_mask[:, : valid_tokens * dimensions["num_frames_per_token"]] = 1
    outputs = runtime.decoder.run(
        None,
        {
            "hybrid_tokens": sample,
            "motion_pad_mask": motion_mask,
            "global_translation": global_translation[None],
        },
    )
    result = {
        output.name: value
        for output, value in zip(runtime.decoder.get_outputs(), outputs)
    }
    for name, value in result.items():
        _require_finite_array(value, f"decoder output {name}")
    return result


def _slice_generated_outputs(
    decoded: dict[str, np.ndarray],
    prepared: PreparedHistory,
    dimensions: dict[str, int],
) -> dict[str, np.ndarray]:
    start = prepared.history_frames
    stop = start + dimensions["generation_frames"]
    return {name: value[:, start:stop].copy() for name, value in decoded.items()}


def _build_global_hybrid_tokens(
    decoded: dict[str, np.ndarray],
    decoder_hybrid: np.ndarray,
    prepared: PreparedHistory,
    dimensions: dict[str, int],
) -> np.ndarray:
    """Mirror buildGlobalHybridTokens for one complete generation window."""
    generation_tokens = dimensions["generation_tokens"]
    frames_per_token = dimensions["num_frames_per_token"]
    hybrid_dim = dimensions["hybrid_dim"]
    root_frame_dim = dimensions["root_features_per_frame"]
    root_token_dim = dimensions["nframe_root_dim"]
    motion = decoded["normalized_motion"]
    result = np.zeros((1, generation_tokens, hybrid_dim), dtype=np.float32)
    for token in range(generation_tokens):
        for frame_in_token in range(frames_per_token):
            source_frame = prepared.history_frames + token * frames_per_token + frame_in_token
            destination_start = frame_in_token * root_frame_dim
            result[0, token, destination_start : destination_start + root_frame_dim] = motion[
                0,
                source_frame,
                :root_frame_dim,
            ]
        source_token = prepared.history_tokens + token
        result[0, token, root_token_dim:] = decoder_hybrid[0, source_token, root_token_dim:]
    return result


def _concat_window_outputs(windows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not windows:
        raise ValueError("At least one rollout window is required.")
    names = set(windows[0])
    if any(set(window) != names for window in windows[1:]):
        raise ValueError("Decoder output contracts changed between rollout windows.")
    return {name: np.concatenate([window[name] for window in windows], axis=1) for name in sorted(names)}


def _rollout(
    runtime: ModelFilesRuntime,
    text_conditions: np.ndarray,
    noises: list[np.ndarray],
    cfg_weight: float,
    initial_translation: np.ndarray,
    initial_heading: float,
) -> RolloutResult:
    dimensions = runtime.manifest["dimensions"]
    state = RolloutState(
        global_hybrid=np.zeros((1, 0, dimensions["hybrid_dim"]), dtype=np.float32),
        initial_translation=initial_translation.copy(),
        initial_heading=initial_heading,
    )
    windows: list[dict[str, np.ndarray]] = []
    continuity: list[dict[str, Any]] = []
    for window_index, noise in enumerate(noises):
        prepared = _prepare_history(state, runtime.manifest)
        sample, masks = _make_window(dimensions, prepared, noise)
        sample = _denoise(
            runtime,
            sample,
            text_conditions,
            cfg_weight,
            prepared,
            masks,
        )
        decoder_hybrid, world_translation, next_heading = _recenter_and_requantize(
            sample,
            runtime.manifest,
            prepared,
        )
        valid_tokens = prepared.history_tokens + dimensions["generation_tokens"]
        decoded = _decode(
            runtime,
            decoder_hybrid,
            world_translation,
            valid_tokens,
        )
        generated = _slice_generated_outputs(decoded, prepared, dimensions)
        windows.append(generated)
        new_tokens = _build_global_hybrid_tokens(
            decoded,
            decoder_hybrid,
            prepared,
            dimensions,
        )
        state.global_hybrid = np.concatenate((state.global_hybrid, new_tokens), axis=1)
        continuity.append(
            {
                "window_index": window_index,
                "history_frames": prepared.history_frames,
                "history_tokens": prepared.history_tokens,
                "generation_token_offset": prepared.history_tokens,
                "first_heading_angle": prepared.first_heading_angle,
                "prepared_world_translation": prepared.global_translation.astype(float).tolist(),
                "decoded_world_translation": world_translation.astype(float).tolist(),
                "next_history_heading": next_heading,
            }
        )
    return RolloutResult(
        windows=windows,
        accumulated=_concat_window_outputs(windows),
        continuity=continuity,
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require_finite_array(array, "metric summary")
    return {
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _rotation_error_degrees(reference: np.ndarray, candidate: np.ndarray) -> float:
    relative = np.matmul(np.swapaxes(reference, -1, -2), candidate)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return float(np.degrees(np.mean(np.arccos(cosine))))


def _case_motion_metrics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    fps: int,
) -> dict[str, float]:
    for runtime_name, outputs in (("reference", reference), ("candidate", candidate)):
        for output_name, value in outputs.items():
            _require_finite_array(value, f"{runtime_name} motion output {output_name}")
    motion_ref = reference["normalized_motion"].astype(np.float32)
    motion_new = candidate["normalized_motion"].astype(np.float32)
    joints_ref = reference["posed_joints"].astype(np.float32)
    joints_new = candidate["posed_joints"].astype(np.float32)
    roots_ref = reference["root_positions"].astype(np.float32)
    roots_new = candidate["root_positions"].astype(np.float32)
    difference = motion_new - motion_ref
    denominator = max(float(np.linalg.norm(motion_ref) * np.linalg.norm(motion_new)), 1e-12)
    root_distance = np.linalg.norm(roots_new - roots_ref, axis=-1)
    aligned_ref = joints_ref - roots_ref[:, :, None]
    aligned_new = joints_new - roots_new[:, :, None]
    velocity_difference = np.diff(joints_new, axis=1) - np.diff(joints_ref, axis=1)
    heading_ref = reference["global_root_heading"].astype(np.float32)
    heading_new = candidate["global_root_heading"].astype(np.float32)
    heading_ref /= np.maximum(np.linalg.norm(heading_ref, axis=-1, keepdims=True), 1e-12)
    heading_new /= np.maximum(np.linalg.norm(heading_new, axis=-1, keepdims=True), 1e-12)
    heading_cosine = np.clip(np.sum(heading_ref * heading_new, axis=-1), -1, 1)
    contacts_ref = reference["foot_contacts"].astype(bool)
    contacts_new = candidate["foot_contacts"].astype(bool)
    return {
        "normalized_motion_rmse": float(np.sqrt(np.mean(difference * difference))),
        "normalized_motion_cosine": float(np.clip(np.sum(motion_ref * motion_new) / denominator, -1, 1)),
        "mpjpe_m": float(np.mean(np.linalg.norm(joints_new - joints_ref, axis=-1))),
        "root_aligned_mpjpe_m": float(np.mean(np.linalg.norm(aligned_new - aligned_ref, axis=-1))),
        "root_ade_m": float(np.mean(root_distance)),
        "root_fde_m": float(np.mean(root_distance[:, -1])),
        "joint_velocity_error_mps": float(np.mean(np.linalg.norm(velocity_difference, axis=-1)) * fps),
        "global_rotation_error_degrees": _rotation_error_degrees(
            reference["global_rotations"],
            candidate["global_rotations"],
        ),
        "local_rotation_error_degrees": _rotation_error_degrees(
            reference["local_rotations"],
            candidate["local_rotations"],
        ),
        "heading_error_degrees": float(np.degrees(np.mean(np.arccos(heading_cosine)))),
        "contact_agreement": float(np.mean(contacts_ref == contacts_new)),
    }


def _motion_metrics(
    references: list[dict[str, np.ndarray]],
    candidates: list[dict[str, np.ndarray]],
    fps: int,
) -> dict[str, Any]:
    if len(references) != len(candidates) or not references:
        raise ValueError("Motion metric inputs must contain the same non-zero number of cases.")
    per_case = {name: [] for name in _case_motion_metrics(references[0], candidates[0], fps)}
    contact_reference: list[np.ndarray] = []
    contact_candidate: list[np.ndarray] = []

    for reference, candidate in zip(references, candidates):
        metrics = _case_motion_metrics(reference, candidate, fps)
        for name, value in metrics.items():
            per_case[name].append(value)
        contacts_ref = reference["foot_contacts"].astype(bool)
        contacts_new = candidate["foot_contacts"].astype(bool)
        contact_reference.append(contacts_ref)
        contact_candidate.append(contacts_new)

    contact_ref = np.concatenate([value.reshape(-1) for value in contact_reference])
    contact_new = np.concatenate([value.reshape(-1) for value in contact_candidate])
    true_positive = int(np.sum(contact_ref & contact_new))
    false_positive = int(np.sum(~contact_ref & contact_new))
    false_negative = int(np.sum(contact_ref & ~contact_new))
    f1_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    result = {name: _summary(values) for name, values in per_case.items()}
    result["contact_f1"] = 1.0 if f1_denominator == 0 else 2 * true_positive / f1_denominator
    result["contact_iou"] = 1.0 if iou_denominator == 0 else true_positive / iou_denominator
    return result


_LOWER_IS_WORSE = frozenset({"normalized_motion_cosine", "contact_agreement"})


def _worst_cases(
    cases: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]],
    fps: int,
) -> dict[str, dict[str, Any]]:
    if not cases:
        raise ValueError("Worst-case attribution requires at least one case.")
    attributed = [
        (metadata, _case_motion_metrics(reference, candidate, fps)) for metadata, reference, candidate in cases
    ]
    result: dict[str, dict[str, Any]] = {}
    for metric in attributed[0][1]:
        selected = (
            min(attributed, key=lambda item: item[1][metric])
            if metric in _LOWER_IS_WORSE
            else max(
                attributed,
                key=lambda item: item[1][metric],
            )
        )
        metadata, values = selected
        result[metric] = {
            "value": values[metric],
            **metadata,
        }
    return result


_PUBLIC_AGGREGATE_FIELDS = (
    "method",
    "runtime_environment",
    "prompt_manifest",
    "models",
    "contract_validation",
    "text_conditions",
    "motion_fidelity",
    "motion_fidelity_by_window",
    "continuation_coverage",
    "cpu_timing",
)
_PROMPT_TEXT_KEYS = frozenset(("prompt", "prompts", "prompt_text", "prompt_texts"))


def _require_finite_array(value: np.ndarray, label: str) -> None:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")


def _validate_json_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_finite(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite.")


def _validate_public_report(value: Any, path: str = "report") -> None:
    """Fail closed if a public aggregate contains prompt text or a local path."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in _PROMPT_TEXT_KEYS:
                raise ValueError(f"{path}.{key} must not contain prompt text.")
            _validate_public_report(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_public_report(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith(("file://", "~/", "~\\"))
    ):
        raise ValueError(f"{path} must not contain an absolute local path.")


def _build_public_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build the Git-tracked aggregate by explicit allowlist."""

    missing = [
        field
        for field in ("schema_version", *_PUBLIC_AGGREGATE_FIELDS)
        if field not in report
    ]
    if missing:
        raise ValueError(f"Detailed report is missing public aggregate fields: {missing}.")
    prompt_manifest = report["prompt_manifest"]
    if (
        not isinstance(prompt_manifest, dict)
        or not isinstance(prompt_manifest.get("provenance_sidecar"), dict)
    ):
        raise TypeError(
            "A public report requires a validated prompt-provenance sidecar."
        )
    public_report = {
        "format": "ardy-browser-fp16-aggregate",
        "format_version": 2,
        "source_report_schema_version": report["schema_version"],
        **{
            field: copy.deepcopy(report[field])
            for field in _PUBLIC_AGGREGATE_FIELDS
        },
    }
    _validate_json_finite(public_report)
    _validate_public_report(public_report)
    return public_report


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    _validate_json_finite(report)
    encoded = json.dumps(
        report,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _write_reports(
    report: dict[str, Any],
    *,
    detailed_output: Path,
    public_output: Path | None,
) -> dict[str, Any]:
    """Write local diagnostics and, when requested, a Git-safe aggregate."""

    if (
        public_output is not None
        and detailed_output.resolve() == public_output.resolve()
    ):
        raise ValueError("--output and --public-output must be different files.")
    public_report = (
        _build_public_report(report) if public_output is not None else None
    )
    _write_json_report(detailed_output, report)
    if public_output is None or public_report is None:
        return report
    _write_json_report(public_output, public_report)
    return public_report


def _runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("numpy", "onnx", "onnxruntime", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
        "onnxruntime": {
            "execution_provider": "CPUExecutionProvider",
            "graph_optimizations": "disabled",
            "intra_op_num_threads": 8,
        },
    }


def main() -> None:
    args = _parse_args()
    if (
        args.public_output is not None
        and args.output.resolve() == args.public_output.resolve()
    ):
        raise ValueError("--output and --public-output must be different files.")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    if not math.isfinite(args.cfg_weight) or not 0 <= args.cfg_weight <= 100:
        raise ValueError("--cfg-weight must be finite and between 0 and 100.")
    if args.windows <= 0:
        raise ValueError("--windows must be positive.")
    initial_translation = _parse_initial_translation(args.initial_translation)
    if not math.isfinite(args.initial_heading):
        raise ValueError("--initial-heading must be finite.")
    initial_heading = math.atan2(math.sin(args.initial_heading), math.cos(args.initial_heading))
    prompt_bytes, prompt_file_identity = _read_stable_bytes(args.prompts)
    prompts = _select_prompts(
        args.prompts,
        args.split,
        args.count,
        encoded=prompt_bytes,
    )
    prompt_metadata_path = args.prompt_metadata
    if prompt_metadata_path is None:
        inferred_metadata_path = args.prompts.with_suffix(".metadata.json")
        if inferred_metadata_path.is_file():
            prompt_metadata_path = inferred_metadata_path
    if args.public_output is not None and prompt_metadata_path is None:
        raise ValueError(
            "--public-output requires --prompt-metadata or a sibling "
            "PROMPTS.metadata.json provenance sidecar."
        )
    prompt_metadata_bytes = None
    if prompt_metadata_path is not None:
        prompt_metadata_bytes, _ = _read_stable_bytes(prompt_metadata_path)
    prompt_manifest = _prompt_manifest_identity(
        args.prompts,
        prompt_metadata_path,
        manifest_bytes=prompt_bytes,
        manifest_identity=prompt_file_identity,
        metadata_bytes=prompt_metadata_bytes,
    )
    reference_input = args.reference_dir.resolve()
    candidate_input = args.candidate_dir.resolve()
    if reference_input == candidate_input:
        raise ValueError("--reference-dir and --candidate-dir must be different directories.")
    if (
        reference_input in candidate_input.parents
        or candidate_input in reference_input.parents
    ):
        raise ValueError(
            "--reference-dir and --candidate-dir must not contain one another."
        )
    protected_inputs = {
        args.prompts.resolve(),
        *(
            (prompt_metadata_path.resolve(),)
            if prompt_metadata_path is not None
            else ()
        ),
    }
    for option, output_path in (
        ("--output", args.output),
        ("--public-output", args.public_output),
    ):
        if output_path is None:
            continue
        resolved_output = output_path.resolve()
        if resolved_output in protected_inputs or any(
            model_dir == resolved_output or model_dir in resolved_output.parents
            for model_dir in (reference_input, candidate_input)
        ):
            raise ValueError(f"{option} must not overwrite an evaluator input.")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Run this script with `uv run --extra browser`.") from error

    with tempfile.TemporaryDirectory(prefix="ardy-fp16-eval-") as temporary:
        temporary_dir = Path(temporary)
        reference_dir = temporary_dir / "reference"
        candidate_dir = temporary_dir / "candidate"
        reference_files = _prepare_model_files(
            args.reference_dir,
            reference_dir,
        )
        candidate_files = _prepare_model_files(
            args.candidate_dir,
            candidate_dir,
        )
        (
            reference_manifest,
            candidate_manifest,
            precision_validation,
        ) = _validate_model_pair_before_sessions(
            reference_files,
            candidate_files,
        )
        reference = _create_runtime(reference_dir, reference_manifest)
        candidate = _create_runtime(candidate_dir, candidate_manifest)
        dimensions = reference.manifest["dimensions"]
        accumulated_frames = args.windows * dimensions["generation_frames"]
        if accumulated_frames > reference.manifest["generation"]["max_frames"]:
            raise ValueError(
                f"--windows={args.windows} requests {accumulated_frames} frames, "
                f"exceeding the browser limit of {reference.manifest['generation']['max_frames']}."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            reference_dir / reference.manifest["tokenizer"]["directory"],
            local_files_only=True,
        )

        reference_conditions = [_encode(reference, tokenizer, prompt) for prompt in prompts]
        candidate_conditions = [_encode(candidate, tokenizer, prompt) for prompt in prompts]
        condition_reference = np.concatenate([value.reshape(1, -1) for value in reference_conditions])
        condition_candidate = np.concatenate([value.reshape(1, -1) for value in candidate_conditions])
        condition_difference = condition_candidate - condition_reference
        condition_cosines = np.clip(
            np.sum(condition_reference * condition_candidate, axis=1)
            / np.maximum(
                np.linalg.norm(condition_reference, axis=1) * np.linalg.norm(condition_candidate, axis=1),
                1e-12,
            ),
            -1,
            1,
        )

        reference_outputs: list[dict[str, np.ndarray]] = []
        candidate_outputs: list[dict[str, np.ndarray]] = []
        per_window_cases: list[list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]]] = [
            [] for _ in range(args.windows)
        ]
        accumulated_cases: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        continuity_records: list[dict[str, Any]] = []
        reference_elapsed = 0.0
        candidate_elapsed = 0.0
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                paired_seed = (seed + prompt_index * 1_000_003) & 0xFFFF_FFFF
                random = PortableRandom(paired_seed)
                noises = [
                    random.normal_array(
                        (
                            1,
                            dimensions["generation_tokens"],
                            dimensions["hybrid_dim"],
                        )
                    )
                    for _ in range(args.windows)
                ]
                started = time.perf_counter()
                reference_rollout = _rollout(
                    reference,
                    reference_conditions[prompt_index],
                    noises,
                    args.cfg_weight,
                    initial_translation,
                    initial_heading,
                )
                reference_elapsed += time.perf_counter() - started

                started = time.perf_counter()
                candidate_rollout = _rollout(
                    candidate,
                    candidate_conditions[prompt_index],
                    noises,
                    args.cfg_weight,
                    initial_translation,
                    initial_heading,
                )
                candidate_elapsed += time.perf_counter() - started
                reference_outputs.append(reference_rollout.accumulated)
                candidate_outputs.append(candidate_rollout.accumulated)

                case_metadata = {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "paired_seed": paired_seed,
                }
                for window_index, (reference_window, candidate_window) in enumerate(
                    zip(reference_rollout.windows, candidate_rollout.windows)
                ):
                    reference_continuity = reference_rollout.continuity[window_index]
                    candidate_continuity = candidate_rollout.continuity[window_index]
                    window_metadata = {
                        **case_metadata,
                        "window_index": window_index,
                        "reference_first_heading_angle": reference_continuity["first_heading_angle"],
                        "candidate_first_heading_angle": candidate_continuity["first_heading_angle"],
                        "reference_world_translation": reference_continuity["decoded_world_translation"],
                        "candidate_world_translation": candidate_continuity["decoded_world_translation"],
                    }
                    per_window_cases[window_index].append((window_metadata, reference_window, candidate_window))
                    continuity_records.extend(
                        (
                            {"runtime": "reference", **case_metadata, **reference_continuity},
                            {"runtime": "candidate", **case_metadata, **candidate_continuity},
                        )
                    )
                accumulated_cases.append(
                    (
                        {
                            **case_metadata,
                            "window_index": args.windows - 1,
                            "window_count": args.windows,
                            "scope": "accumulated",
                        },
                        reference_rollout.accumulated,
                        candidate_rollout.accumulated,
                    )
                )

        prompt_digest = hashlib.sha256(("\n".join(prompts) + "\n").encode("utf-8")).hexdigest()
        fps = dimensions["fps"]
        per_window_metrics = [
            {
                "window_index": window_index,
                "history_frames": 0 if window_index == 0 else dimensions["history_frames"],
                "generation_token_offset": 0 if window_index == 0 else dimensions["history_tokens"],
                "motion_fidelity": _motion_metrics(
                    [case[1] for case in cases],
                    [case[2] for case in cases],
                    fps,
                ),
            }
            for window_index, cases in enumerate(per_window_cases)
        ]
        absolute_headings = [abs(float(record["first_heading_angle"])) for record in continuity_records]
        translation_norms = [
            float(np.linalg.norm(np.asarray(record["decoded_world_translation"], dtype=np.float64)))
            for record in continuity_records
        ]
        report = {
            "schema_version": 3,
            "method": {
                "reference": "FP32 ONNX Runtime CPU with graph optimizations disabled",
                "candidate": "mixed-FP16 ONNX Runtime CPU with graph optimizations disabled",
                "rollout": (
                    f"{args.windows} consecutive browser windows, 10-step deterministic DDIM per window, "
                    "40-frame history, recenter, FSQ requantize, heading, and accumulated world translation"
                ),
                "prompt_count": len(prompts),
                "prompt_split": args.split,
                "prompt_sha256": prompt_digest,
                "prompt_selection": (
                    "first-occurrence text deduplication in manifest order, "
                    "then evenly spaced numpy.linspace integer indices"
                ),
                "seeds": seeds,
                "case_count": len(prompts) * len(seeds),
                "cfg_weight": args.cfg_weight,
                "window_count": args.windows,
                "frames_per_case": accumulated_frames,
                "noise_generator": "browser PortableRandom (Mulberry32 + Marsaglia polar)",
                "initial_translation": initial_translation.astype(float).tolist(),
                "initial_heading": initial_heading,
            },
            "runtime_environment": _runtime_environment(),
            "prompt_manifest": prompt_manifest,
            "models": {
                "reference": reference_files.identity.to_report(),
                "candidate": candidate_files.identity.to_report(),
                "transport_saved_bytes": (
                    reference_files.identity.transport_size_bytes
                    - candidate_files.identity.transport_size_bytes
                ),
                "transport_saved_fraction": (
                    1
                    - candidate_files.identity.transport_size_bytes
                    / reference_files.identity.transport_size_bytes
                ),
                "raw_saved_bytes": (
                    reference_files.identity.raw_size_bytes
                    - candidate_files.identity.raw_size_bytes
                ),
                "raw_saved_fraction": (
                    1
                    - candidate_files.identity.raw_size_bytes
                    / reference_files.identity.raw_size_bytes
                ),
            },
            "contract_validation": {
                "non_precision_contract_equal": True,
                "non_precision_contract_sha256": _contract_fingerprint(reference.manifest),
                "tokenizer_payloads_equal": True,
                "tokenizer_payload_sha256": _tokenizer_payloads(reference),
                **precision_validation,
            },
            "text_conditions": {
                "mae": float(np.mean(np.abs(condition_difference))),
                "rmse": float(np.sqrt(np.mean(condition_difference * condition_difference))),
                "max_abs": float(np.max(np.abs(condition_difference))),
                "cosine_mean": float(np.mean(condition_cosines)),
                "cosine_min": float(np.min(condition_cosines)),
            },
            "motion_fidelity": _motion_metrics(
                reference_outputs,
                candidate_outputs,
                fps,
            ),
            "motion_fidelity_by_window": per_window_metrics,
            "worst_cases": {
                "per_window": _worst_cases(
                    [case for cases in per_window_cases for case in cases],
                    fps,
                ),
                "accumulated": _worst_cases(accumulated_cases, fps),
            },
            "continuation_coverage": {
                "history_frames_observed": sorted({int(record["history_frames"]) for record in continuity_records}),
                "generation_token_offsets_observed": sorted(
                    {int(record["generation_token_offset"]) for record in continuity_records}
                ),
                "nonzero_first_heading_count": sum(value > 1e-8 for value in absolute_headings),
                "first_heading_abs_min": min(absolute_headings),
                "first_heading_abs_max": max(absolute_headings),
                "decoded_world_translation_norm_max": max(translation_norms),
            },
            "cpu_timing": {
                "reference_total_seconds": reference_elapsed,
                "candidate_total_seconds": candidate_elapsed,
                "reference_mean_seconds_per_case": reference_elapsed / len(reference_outputs),
                "candidate_mean_seconds_per_case": candidate_elapsed / len(candidate_outputs),
                "windows_per_case": args.windows,
                "note": (
                    "CPU FP16 timing is diagnostic only; the deployment backend is WebGPU. "
                    "Each case includes every consecutive window."
                ),
            },
        }
        printed_report = _write_reports(
            report,
            detailed_output=args.output,
            public_output=args.public_output,
        )
        print(
            json.dumps(
                printed_report,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
