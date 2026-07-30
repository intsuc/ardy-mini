# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Benchmark ARDY's production LLM2Vec and distilled MiniLM condition paths.

Teacher and student intentionally live behind separate subcommands so each
measurement is made in a fresh process.  ``teacher``/``student`` measure the
encoder wrappers alone; ``full-teacher``/``full-student`` additionally load the
same Core40 model and measure encoder plus both checkpoint text projections:

    uv run python scripts/minilm/benchmark_encoders.py teacher --output teacher.json
    uv run python scripts/minilm/benchmark_encoders.py student \
        --student-path artifacts/minilm-ardy-core40 --output student.json
    uv run python scripts/minilm/benchmark_encoders.py full-teacher \
        --checkpoints-dir checkpoints --output full-teacher.json
    uv run python scripts/minilm/benchmark_encoders.py full-student \
        --checkpoints-dir checkpoints \
        --student-path artifacts/minilm-ardy-core40 --output full-student.json

Every timed call uses the public production wrapper with an external batch
size of one.  CUDA timings are synchronized and CUDA memory numbers are
PyTorch allocator numbers; RSS and MemAvailable are sampled from ``/proc``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import torch

DEFAULT_BASE_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
DEFAULT_PEFT_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
DEFAULT_FOUNDATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_FOUNDATION_MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
DEFAULT_BASE_MODEL_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
DEFAULT_PEFT_MODEL_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"
DEFAULT_STUDENT_PATH = "artifacts/minilm-ardy-core40"
DEFAULT_PROMPT = "A person walks forward at a steady pace."
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_ARDY_MODEL = "core"
FULL_STACK_ARDY_MODEL = "ARDY-Core-RP-20FPS-Horizon40"
LEGACY_TEXT_DIM = 4096
CONDITION_DIM = 1024
DIRECT_TEXT_DIM = 2 * CONDITION_DIM
_T = TypeVar("_T")


@dataclass(frozen=True)
class HostMemory:
    rss_bytes: int | None
    lifetime_peak_rss_bytes: int | None
    mem_available_bytes: int | None


@dataclass
class FullConditionStack:
    """Loaded components for the production condition path."""

    ardy_model: Any
    encoder: Any
    root_projection: Any
    body_projection: Any
    encoder_kind: str
    resolved_ardy_model: str
    model_dir: Path


class _MemorySampler:
    """Lightweight sampler for peaks that may disappear before a stage ends."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.max_rss_bytes: int | None = None
        self.min_mem_available_bytes: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)

    def sample(self) -> None:
        snapshot = _host_memory()
        if snapshot.rss_bytes is not None:
            self.max_rss_bytes = (
                snapshot.rss_bytes if self.max_rss_bytes is None else max(self.max_rss_bytes, snapshot.rss_bytes)
            )
        if snapshot.mem_available_bytes is not None:
            self.min_mem_available_bytes = (
                snapshot.mem_available_bytes
                if self.min_mem_available_bytes is None
                else min(self.min_mem_available_bytes, snapshot.mem_available_bytes)
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def __enter__(self):
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.sample()
        self._stop.set()
        self._thread.join()


def _proc_value(path: Path, key: str) -> int | None:
    """Read a kB-valued key from a Linux procfs file and return bytes."""
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                label, separator, value = line.partition(":")
                if separator and label == key:
                    fields = value.split()
                    if not fields:
                        return None
                    multiplier = 1024 if len(fields) == 1 or fields[1].lower() == "kb" else 1
                    return int(fields[0]) * multiplier
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _host_memory() -> HostMemory:
    return HostMemory(
        rss_bytes=_proc_value(Path("/proc/self/status"), "VmRSS"),
        lifetime_peak_rss_bytes=_proc_value(Path("/proc/self/status"), "VmHWM"),
        mem_available_bytes=_proc_value(Path("/proc/meminfo"), "MemAvailable"),
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device={requested!r} requested, but CUDA is unavailable")
    return requested


def _cuda_device(device: str) -> torch.device | None:
    resolved = torch.device(device)
    return resolved if resolved.type == "cuda" else None


def _cuda_synchronize(device: torch.device | None) -> None:
    if device is not None:
        torch.cuda.synchronize(device)


def _cuda_current(device: torch.device | None) -> dict[str, int] | None:
    if device is None:
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _optional_subtract(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left - right


def _measure_stage(
    operation: Callable[[], _T],
    cuda_device: torch.device | None,
) -> tuple[_T, float, dict[str, Any]]:
    _cuda_synchronize(cuda_device)
    before = _host_memory()
    cuda_before = _cuda_current(cuda_device)
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)

    with _MemorySampler() as sampler:
        started = time.perf_counter()
        result = operation()
        _cuda_synchronize(cuda_device)
        elapsed = time.perf_counter() - started

    after = _host_memory()
    cuda_after = _cuda_current(cuda_device)
    cuda_peak = None
    if cuda_device is not None:
        cuda_peak = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
        }

    sampled_peak_rss = sampler.max_rss_bytes
    sampled_drop = _optional_subtract(before.mem_available_bytes, sampler.min_mem_available_bytes)
    stage_memory = {
        "host_before": asdict(before),
        "host_after": asdict(after),
        "sampled_peak_rss_bytes": sampled_peak_rss,
        "rss_current_delta_bytes": _optional_subtract(after.rss_bytes, before.rss_bytes),
        "rss_sampled_peak_delta_bytes": _optional_subtract(sampled_peak_rss, before.rss_bytes),
        "sampled_min_mem_available_bytes": sampler.min_mem_available_bytes,
        "mem_available_end_drop_bytes": _optional_subtract(
            before.mem_available_bytes,
            after.mem_available_bytes,
        ),
        "mem_available_peak_drop_bytes": None if sampled_drop is None else max(0, sampled_drop),
        "cuda_before": cuda_before,
        "cuda_after": cuda_after,
        "cuda_peak": cuda_peak,
    }
    return result, elapsed, stage_memory


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("at least one timing sample is required")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "values": values,
    }


def _parameter_stats(owner: Any) -> dict[str, int]:
    module = getattr(owner, "model", owner)
    parameters_method = getattr(module, "parameters", None)
    if not callable(parameters_method):
        raise TypeError(f"{type(owner).__name__} does not expose model parameters")
    parameters = list(parameters_method())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in parameters)),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
    }


def _combined_parameter_stats(ardy_model: Any, encoder: Any) -> dict[str, Any]:
    ardy = _parameter_stats(ardy_model)
    encoder_stats = _parameter_stats(encoder)
    combined = {
        key: ardy[key] + encoder_stats[key]
        for key in ("parameter_count", "parameter_bytes", "trainable_parameter_count")
    }
    return {
        **combined,
        "ardy_model": ardy,
        "encoder": encoder_stats,
        "combined": combined,
    }


def _encoder_kind(command: str) -> str:
    return command.removeprefix("full-")


def _is_full_stack(command: str) -> bool:
    return command.startswith("full-")


def _load_teacher(args: argparse.Namespace, device: str):
    from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    if __package__:
        from .cache_teacher import resolve_pinned_snapshot
    else:
        from cache_teacher import resolve_pinned_snapshot

    # The production wrapper permits TEXT_ENCODER_DEVICE to override its
    # argument.  Make the benchmark CLI authoritative for this child process.
    previous_device = os.environ.get("TEXT_ENCODER_DEVICE")
    os.environ["TEXT_ENCODER_DEVICE"] = device
    try:
        foundation_snapshot = resolve_pinned_snapshot(
            args.foundation_model,
            args.foundation_model_revision,
            allow_patterns=(
                "config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
            ),
            local_files_only=True,
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
            local_files_only=True,
        )
        peft_snapshot = resolve_pinned_snapshot(
            args.peft_model,
            args.peft_model_revision,
            allow_patterns=("adapter_config.json", "adapter_model.safetensors"),
            local_files_only=True,
        )
        return LLM2VecEncoder(
            base_model_name_or_path=str(base_snapshot),
            peft_model_name_or_path=str(peft_snapshot),
            foundation_model_name_or_path=str(foundation_snapshot),
            dtype=args.dtype,
            llm_dim=args.llm_dim,
            device=device,
        )
    finally:
        if previous_device is None:
            os.environ.pop("TEXT_ENCODER_DEVICE", None)
        else:
            os.environ["TEXT_ENCODER_DEVICE"] = previous_device


def _load_student(
    args: argparse.Namespace,
    device: str,
    expected_ardy_model: str | None = None,
):
    from ardy.model.minilm_encoder import MiniLMArdyEncoder

    return MiniLMArdyEncoder(
        model_name_or_path=args.student_path,
        dtype=args.dtype,
        device=device,
        expected_ardy_model=expected_ardy_model or args.expected_ardy_model,
        max_length=args.max_length,
    )


def _load_encoder(
    encoder_kind: str,
    args: argparse.Namespace,
    device: str,
    *,
    expected_ardy_model: str | None = None,
):
    if encoder_kind == "teacher":
        return _load_teacher(args, device)
    if encoder_kind == "student":
        return _load_student(args, device, expected_ardy_model)
    raise ValueError(f"unsupported encoder kind {encoder_kind!r}")


def _validate_projection(
    projection: Any,
    *,
    branch: str,
    projected_text_index: int,
) -> None:
    actual_index = getattr(projection, "projected_text_index", None)
    in_features = getattr(projection, "in_features", None)
    out_features = getattr(projection, "out_features", None)
    if (actual_index, in_features, out_features) != (
        projected_text_index,
        LEGACY_TEXT_DIM,
        CONDITION_DIM,
    ):
        raise ValueError(
            f"Core40 {branch} projection is incompatible: "
            f"projected_text_index={actual_index}, in_features={in_features}, "
            f"out_features={out_features}"
        )


def _load_full_stack(args: argparse.Namespace, device: str) -> FullConditionStack:
    from ardy.model.load_model import load_model
    from ardy.model.registry import resolve_model_name

    checkpoints_dir = Path(args.checkpoints_dir)
    resolved_model = resolve_model_name(args.model, checkpoints_dir=str(checkpoints_dir))
    if resolved_model != FULL_STACK_ARDY_MODEL:
        raise ValueError(
            "full-stack comparison is fixed to the shared Core40 checkpoint; "
            f"{args.model!r} resolved to {resolved_model!r}, expected {FULL_STACK_ARDY_MODEL!r}"
        )
    model_dir = (checkpoints_dir / resolved_model).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Core40 model directory not found: {model_dir}")

    ardy_model = load_model(
        resolved_model,
        device=device,
        text_encoder=False,
        checkpoints_dir=str(checkpoints_dir),
    )
    if getattr(ardy_model, "text_encoder", None) is not None:
        raise RuntimeError("load_model(..., text_encoder=False) unexpectedly attached an encoder")

    kind = _encoder_kind(args.encoder)
    if kind == "student" and args.expected_ardy_model not in (None, resolved_model):
        raise ValueError(
            f"--expected-ardy-model={args.expected_ardy_model!r} conflicts with full-stack {resolved_model!r}"
        )
    encoder = _load_encoder(
        kind,
        args,
        device,
        expected_ardy_model=resolved_model,
    )
    root_projection = ardy_model.denoiser.root_model.embed_text
    body_projection = ardy_model.denoiser.body_model.embed_text
    _validate_projection(
        root_projection,
        branch="root",
        projected_text_index=0,
    )
    _validate_projection(
        body_projection,
        branch="body",
        projected_text_index=1,
    )
    return FullConditionStack(
        ardy_model=ardy_model,
        encoder=encoder,
        root_projection=root_projection,
        body_projection=body_projection,
        encoder_kind=kind,
        resolved_ardy_model=resolved_model,
        model_dir=model_dir,
    )


def _normalize_lengths(lengths: Any) -> Any:
    if isinstance(lengths, torch.Tensor):
        return lengths.detach().cpu().tolist()
    if hasattr(lengths, "tolist"):
        return lengths.tolist()
    return lengths


@torch.inference_mode()
def _full_condition_forward(stack: FullConditionStack, texts: list[str]):
    encoded, lengths = stack.encoder(texts)
    normalized_lengths = _normalize_lengths(lengths)
    if encoded.ndim != 3 or encoded.shape[:2] != (1, 1):
        raise ValueError(f"full-stack encoder must return [1, 1, D], got {tuple(encoded.shape)}")
    if normalized_lengths != [1]:
        raise ValueError(f"full-stack encoder returned unexpected lengths {normalized_lengths!r}")

    input_dim = int(encoded.shape[-1])
    if stack.encoder_kind == "teacher" and input_dim != LEGACY_TEXT_DIM:
        raise ValueError(f"teacher must exercise the legacy 4096 path, got {input_dim}")
    if stack.encoder_kind == "student" and input_dim != DIRECT_TEXT_DIM:
        raise ValueError(f"student must exercise the direct 2048 path, got {input_dim}")

    root = stack.root_projection(encoded)
    body = stack.body_projection(encoded)
    conditions = torch.cat((root, body), dim=-1)
    if conditions.shape != (1, 1, DIRECT_TEXT_DIM):
        raise ValueError(
            f"root/body projection output must be [1, 1, {DIRECT_TEXT_DIM}], got {tuple(conditions.shape)}"
        )
    return conditions, lengths


def _timed_encode(encoder: Any, prompt: str, cuda_device: torch.device | None):
    # A list containing one prompt is ARDY's normal production call shape.
    _cuda_synchronize(cuda_device)
    started = time.perf_counter()
    output = encoder([prompt])
    _cuda_synchronize(cuda_device)
    return output, time.perf_counter() - started


def _shape_and_lengths(output: Any) -> tuple[list[int], str, Any]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("production text encoder must return (tensor, lengths)")
    tensor, lengths = output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"production text encoder returned {type(tensor).__name__}, expected Tensor")
    normalized_lengths = _normalize_lengths(lengths)
    return list(tensor.shape), str(tensor.dtype).removeprefix("torch."), normalized_lengths


def _model_device_dtype(owner: Any) -> tuple[str | None, str | None]:
    module = getattr(owner, "model", owner)
    parameters_method = getattr(module, "parameters", None)
    if not callable(parameters_method):
        return None, None
    try:
        parameter = next(parameters_method())
    except StopIteration:
        return None, None
    return str(parameter.device), str(parameter.dtype).removeprefix("torch.")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file_hashes(base: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        path = base / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"identity file not found: {path}")
        hashes[relative_path] = _sha256_file(path)
    return hashes


def _artifact_file_hashes(artifact_dir: Path) -> dict[str, str]:
    required = (
        artifact_dir / "ardy_minilm_config.json",
        artifact_dir / "condition_heads.safetensors",
    )
    backbone_dir = artifact_dir / "backbone"
    if not backbone_dir.is_dir():
        raise FileNotFoundError(f"artifact backbone directory not found: {backbone_dir}")
    paths = [*required, *sorted(path for path in backbone_dir.rglob("*") if path.is_file())]
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"artifact identity file not found: {path}")
        hashes[path.relative_to(artifact_dir).as_posix()] = _sha256_file(path)
    return hashes


def _artifact_checkpoint_hashes(config: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    containers = [config]
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata)
    for container in containers:
        values.append(container.get("teacher_checkpoint_sha256"))
        values.append(container.get("checkpoint_sha256"))
        teacher_metadata = container.get("teacher_metadata")
        if isinstance(teacher_metadata, dict):
            values.append(teacher_metadata.get("checkpoint_sha256"))
        teacher_lineage = container.get("teacher_cache_lineage")
        if isinstance(teacher_lineage, dict):
            values.append(teacher_lineage.get("checkpoint_sha256"))
    return {value for value in values if isinstance(value, str) and value}


def _teacher_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete immutable teacher identity used by this process."""

    return {
        "foundation_model": args.foundation_model,
        "foundation_model_revision": args.foundation_model_revision,
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "peft_model": args.peft_model,
        "peft_model_revision": args.peft_model_revision,
        "llm_dim": args.llm_dim,
    }


def _student_artifact_identity(encoder: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Hash and summarize the complete local student artifact after measurement."""

    artifact_dir = Path(args.student_path).resolve()
    artifact_config = encoder.artifact_config
    return {
        "path": str(artifact_dir),
        "files_sha256": _artifact_file_hashes(artifact_dir),
        "artifact_fingerprint": artifact_config.get("artifact_fingerprint"),
        "format_version": artifact_config.get("format_version"),
        "base_model": artifact_config.get("base_model"),
        "compatible_ardy_models": artifact_config.get("compatible_ardy_models"),
        "output_dim": artifact_config.get("output_dim"),
    }


def _full_stack_identity(stack: FullConditionStack, args: argparse.Namespace) -> dict[str, Any]:
    model_hashes = _required_file_hashes(
        stack.model_dir,
        ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors"),
    )
    checkpoint_sha256 = model_hashes["denoiser.safetensors"]
    identity: dict[str, Any] = {
        "ardy": {
            "requested_model": args.model,
            "resolved_model": stack.resolved_ardy_model,
            "model_dir": str(stack.model_dir),
            "files_sha256": model_hashes,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "condition_path": {
            "encoder_kind": stack.encoder_kind,
            "encoder_output_dim": LEGACY_TEXT_DIM if stack.encoder_kind == "teacher" else DIRECT_TEXT_DIM,
            "root_projection": {
                "class": f"{type(stack.root_projection).__module__}.{type(stack.root_projection).__qualname__}",
                "projected_text_index": stack.root_projection.projected_text_index,
                "in_features": stack.root_projection.in_features,
                "out_features": stack.root_projection.out_features,
            },
            "body_projection": {
                "class": f"{type(stack.body_projection).__module__}.{type(stack.body_projection).__qualname__}",
                "projected_text_index": stack.body_projection.projected_text_index,
                "in_features": stack.body_projection.in_features,
                "out_features": stack.body_projection.out_features,
            },
            "projection_bias_included": True,
            "output_order": ["root", "body"],
            "output_dim": DIRECT_TEXT_DIM,
        },
    }
    if stack.encoder_kind == "teacher":
        identity["teacher"] = _teacher_identity(args)
        return identity

    artifact_config = stack.encoder.artifact_config
    compatible_models = artifact_config.get("compatible_ardy_models")
    if not isinstance(compatible_models, list) or stack.resolved_ardy_model not in compatible_models:
        raise ValueError(
            "student artifact does not explicitly declare compatibility with "
            f"{stack.resolved_ardy_model!r}: {compatible_models!r}"
        )
    if artifact_config.get("output_dim") != DIRECT_TEXT_DIM:
        raise ValueError(
            f"student artifact output_dim must be {DIRECT_TEXT_DIM}, got {artifact_config.get('output_dim')!r}"
        )
    expected_checkpoint_hashes = _artifact_checkpoint_hashes(artifact_config)
    if not expected_checkpoint_hashes:
        raise ValueError("student artifact does not declare its teacher ARDY checkpoint SHA-256")
    if expected_checkpoint_hashes and expected_checkpoint_hashes != {checkpoint_sha256}:
        raise ValueError(
            "student artifact was distilled for a different ARDY checkpoint: "
            f"loaded={checkpoint_sha256}, artifact={sorted(expected_checkpoint_hashes)}"
        )
    student_identity = _student_artifact_identity(stack.encoder, args)
    student_identity.update(
        {
            "expected_checkpoint_sha256": sorted(expected_checkpoint_hashes),
            "checkpoint_sha256_verified": True,
        }
    )
    identity["student_artifact"] = student_identity
    return identity


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version(), "torch": torch.__version__}
    for distribution in ("ardy", "transformers", "peft", "safetensors"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _hardware_identity(cuda_device: torch.device | None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cuda": None,
    }
    if cuda_device is None:
        return identity
    index = cuda_device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    identity["cuda"] = {
        "index": index,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": torch.version.cuda,
    }
    return identity


def _minimum_present(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _maximum_present(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    cuda_device = _cuda_device(device)
    process_start = _host_memory()
    kind = _encoder_kind(args.encoder)
    full_stack = _is_full_stack(args.encoder)
    stack: FullConditionStack | None = None

    if full_stack:
        stack, load_seconds, load_memory = _measure_stage(
            lambda: _load_full_stack(args, device),
            cuda_device,
        )
        encoder = stack.encoder
        production_call = lambda texts: _full_condition_forward(stack, texts)
        parameter_stats = _combined_parameter_stats(stack.ardy_model, encoder)
        ardy_model_device, ardy_model_dtype = _model_device_dtype(stack.ardy_model)
    else:
        encoder, load_seconds, load_memory = _measure_stage(
            lambda: _load_encoder(kind, args, device),
            cuda_device,
        )
        production_call = encoder
        parameter_stats = _parameter_stats(encoder)
        ardy_model_device, ardy_model_dtype = None, None
    model_device, model_dtype = _model_device_dtype(encoder)

    first_result, first_stage_seconds, first_memory = _measure_stage(
        lambda: _timed_encode(production_call, args.prompt, cuda_device),
        cuda_device,
    )
    first_output, first_latency = first_result
    output_shape, output_dtype, lengths = _shape_and_lengths(first_output)
    if len(output_shape) != 3 or output_shape[:2] != [1, 1]:
        raise RuntimeError(f"production batch-one wrapper returned unexpected shape {output_shape}")
    expected_output_dim = LEGACY_TEXT_DIM if kind == "teacher" and not full_stack else DIRECT_TEXT_DIM
    if output_shape != [1, 1, expected_output_dim]:
        raise RuntimeError(
            f"{args.encoder} returned {output_shape}, expected [1, 1, {expected_output_dim}]"
        )
    if lengths != [1]:
        raise RuntimeError(f"{args.encoder} returned unexpected lengths {lengths!r}")
    def warm_operation():
        latencies: list[float] = []
        last_output = None
        for _ in range(args.warm_runs):
            last_output, latency = _timed_encode(production_call, args.prompt, cuda_device)
            latencies.append(latency)
        return last_output, latencies

    warm_result, warm_stage_seconds, warm_memory = _measure_stage(warm_operation, cuda_device)
    warm_output, warm_latencies = warm_result
    warm_shape, warm_dtype, warm_lengths = _shape_and_lengths(warm_output)
    if warm_shape != output_shape:
        raise RuntimeError(f"output shape changed from {output_shape} to {warm_shape}")
    if warm_dtype != output_dtype:
        raise RuntimeError(f"output dtype changed from {output_dtype} to {warm_dtype}")
    if warm_lengths != lengths:
        raise RuntimeError(f"output lengths changed from {lengths!r} to {warm_lengths!r}")

    process_end = _host_memory()
    stage_memories = (load_memory, first_memory, warm_memory)
    sampled_min_available = _minimum_present([stage["sampled_min_mem_available_bytes"] for stage in stage_memories])
    rss_peak = _maximum_present(
        [
            process_end.lifetime_peak_rss_bytes,
            *[stage["sampled_peak_rss_bytes"] for stage in stage_memories],
        ]
    )
    cuda_allocated_peak = _maximum_present(
        [None if stage["cuda_peak"] is None else stage["cuda_peak"]["allocated_bytes"] for stage in stage_memories]
    )
    cuda_reserved_peak = _maximum_present(
        [None if stage["cuda_peak"] is None else stage["cuda_peak"]["reserved_bytes"] for stage in stage_memories]
    )
    identity_started = time.perf_counter()
    if stack is not None:
        identity = _full_stack_identity(stack, args)
    elif kind == "teacher":
        identity = {"teacher": _teacher_identity(args)}
    else:
        identity = {"student_artifact": _student_artifact_identity(encoder, args)}
    identity_seconds = time.perf_counter() - identity_started
    result: dict[str, Any] = {
        "schema_version": 1,
        "encoder": args.encoder,
        "encoder_kind": kind,
        "benchmark_scope": "full_condition_stack" if full_stack else "encoder_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "versions": _package_versions(),
        "hardware": _hardware_identity(cuda_device),
        "configuration": {
            "fresh_process_required": True,
            "process_id": os.getpid(),
            "device_requested": args.device,
            "device_resolved": device,
            "model_device": model_device,
            "dtype_requested": args.dtype,
            "model_dtype": model_dtype,
            "ardy_model_device": ardy_model_device,
            "ardy_model_dtype": ardy_model_dtype,
            "prompt": args.prompt,
            "warm_runs": args.warm_runs,
            "first_measured_calls": 1,
            "warmup_calls_before_warm_measurement": 1,
            "warm_measured_calls": args.warm_runs,
            "external_batch_size": 1,
            "production_wrapper": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
            "timed_operation": (
                "prompt -> encoder -> root_model.embed_text/body_model.embed_text -> concat[root,body]"
                if full_stack
                else "prompt -> encoder"
            ),
            "load_timing_scope": (
                "path-specific imports, local-only pinned Hugging Face snapshot resolution, "
                "Core40 construction, encoder construction, and device transfer"
                if full_stack and kind == "teacher"
                else (
                    "path-specific imports, local-only pinned Hugging Face snapshot resolution, "
                    "encoder construction, and device transfer"
                    if kind == "teacher"
                    else (
                        "path-specific imports, Core40 construction, local artifact validation, "
                        "encoder construction, and device transfer"
                        if full_stack
                        else (
                            "path-specific imports, local artifact validation, "
                            "encoder construction, and device transfer"
                        )
                    )
                )
            ),
            "cuda_context_timing": (
                "CUDA synchronization initializes the context before each stage timer; "
                "context initialization is excluded from stage latency"
                if cuda_device is not None
                else None
            ),
        },
        "timing_seconds": {
            "load": load_seconds,
            "first_encode_stage": first_stage_seconds,
            "warm_encode_stage": warm_stage_seconds,
            "first_encode": _latency_summary([first_latency]),
            "warm_encode": _latency_summary(warm_latencies),
            "identity_sha256_after_measurement": identity_seconds,
        },
        "parameters": parameter_stats,
        "output": {
            "shape": output_shape,
            "dtype": output_dtype,
            "lengths": lengths,
            "warm_shape": warm_shape,
            "warm_dtype": warm_dtype,
            "warm_lengths": warm_lengths,
        },
        "memory": {
            "measurement_notes": {
                "rss": "Linux /proc/self/status VmRSS; sampled every 5 ms during each stage",
                "rss_peak": "stage sampled peak plus process-lifetime /proc VmHWM",
                "mem_available": (
                    "Linux system-wide /proc/meminfo MemAvailable; drops are relative to "
                    "each stage start and are sensitive to other processes and page cache"
                ),
                "cuda": (
                    "absolute PyTorch allocator allocated/reserved bytes on the selected "
                    "device; excludes CUDA driver/context and non-PyTorch allocations"
                ),
                "unified_memory": (
                    "RSS/MemAvailable and CUDA allocator figures can describe overlapping "
                    "physical memory on unified-memory systems; do not add them"
                ),
                "snapshot_resolution": (
                    "teacher load measurements include local-only pinned Hugging Face "
                    "snapshot cache resolution; network downloads are disabled"
                    if kind == "teacher"
                    else "student load measurements read the supplied local artifact"
                ),
                "identity_hashing": "model/artifact SHA-256 is computed after process_end and excluded from metrics",
            },
            "process_start": asdict(process_start),
            "process_end": asdict(process_end),
            "summary": {
                "rss_current_bytes": process_end.rss_bytes,
                "rss_peak_bytes": rss_peak,
                "mem_available_before_bytes": process_start.mem_available_bytes,
                "mem_available_after_bytes": process_end.mem_available_bytes,
                "mem_available_min_bytes": sampled_min_available,
                "mem_available_drop_bytes": (
                    None
                    if process_start.mem_available_bytes is None or sampled_min_available is None
                    else max(0, process_start.mem_available_bytes - sampled_min_available)
                ),
                "cuda_allocated_current_bytes": (
                    None if cuda_device is None else int(torch.cuda.memory_allocated(cuda_device))
                ),
                "cuda_allocated_peak_bytes": cuda_allocated_peak,
                "cuda_reserved_current_bytes": (
                    None if cuda_device is None else int(torch.cuda.memory_reserved(cuda_device))
                ),
                "cuda_reserved_peak_bytes": cuda_reserved_peak,
            },
            "load": load_memory,
            "first_encode": first_memory,
            "warm_encode": warm_memory,
        },
        "identity": identity,
    }
    if kind == "teacher":
        result["configuration"].update(_teacher_identity(args))
        result["configuration"]["teacher_internal_batch_size"] = 1
    else:
        result["configuration"].update(
            {
                "student_path": str(Path(args.student_path)),
                "expected_ardy_model": (stack.resolved_ardy_model if stack is not None else args.expected_ardy_model),
                "max_length": args.max_length,
            }
        )
    if stack is not None:
        result["configuration"].update(
            {
                "ardy_model_requested": args.model,
                "ardy_model_resolved": stack.resolved_ardy_model,
                "checkpoints_dir": str(Path(args.checkpoints_dir)),
                "condition_projection_path": (
                    "legacy_4096_linear" if kind == "teacher" else "direct_2048_root_body_select"
                ),
                "condition_projection_bias_included": True,
            }
        )

    _write_json(args.output, result)
    print(json.dumps({"encoder": args.encoder, "output": str(args.output), "result": result}, indent=2))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="model inference dtype",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="single English motion prompt")
    parser.add_argument("--warm-runs", type=int, default=30, help="number of measured warm batch-one calls")


def _add_teacher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--peft-model", default=DEFAULT_PEFT_MODEL)
    parser.add_argument("--foundation-model", default=DEFAULT_FOUNDATION_MODEL)
    parser.add_argument(
        "--foundation-model-revision",
        default=DEFAULT_FOUNDATION_MODEL_REVISION,
    )
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_MODEL_REVISION,
    )
    parser.add_argument(
        "--peft-model-revision",
        default=DEFAULT_PEFT_MODEL_REVISION,
    )
    parser.add_argument("--llm-dim", type=int, default=LEGACY_TEXT_DIM)


def _add_student_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--student-path", default=DEFAULT_STUDENT_PATH)
    parser.add_argument(
        "--expected-ardy-model",
        default=None,
        help="optional compatibility key checked against the artifact",
    )
    parser.add_argument("--max-length", type=int, default=None)


def _add_full_stack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_ARDY_MODEL,
        help=f"must resolve to {FULL_STACK_ARDY_MODEL}",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help="local directory containing the shared Core40 checkpoint",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="encoder", required=True)

    teacher = subparsers.add_parser("teacher", help="benchmark the production LLM2Vec wrapper")
    _add_common_arguments(teacher)
    _add_teacher_arguments(teacher)

    student = subparsers.add_parser("student", help="benchmark the production distilled MiniLM wrapper")
    _add_common_arguments(student)
    _add_student_arguments(student)

    full_teacher = subparsers.add_parser(
        "full-teacher",
        help="benchmark LLM2Vec plus shared Core40 root/body projections",
    )
    _add_common_arguments(full_teacher)
    _add_teacher_arguments(full_teacher)
    _add_full_stack_arguments(full_teacher)

    full_student = subparsers.add_parser(
        "full-student",
        help="benchmark MiniLM plus shared Core40 direct root/body paths",
    )
    _add_common_arguments(full_student)
    _add_student_arguments(full_student)
    _add_full_stack_arguments(full_student)

    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error("--warm-runs must be at least 1")
    if _encoder_kind(args.encoder) == "teacher" and args.llm_dim != LEGACY_TEXT_DIM:
        parser.error(f"teacher benchmarks require --llm-dim={LEGACY_TEXT_DIM}")
    return args


if __name__ == "__main__":
    run(parse_args())
