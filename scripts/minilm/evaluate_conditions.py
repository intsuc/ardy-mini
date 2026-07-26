# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained MiniLM artifact against cached ARDY teacher conditions.

Expected cache shards are ``torch.save`` dictionaries with these keys:

* ``texts``: motion-prompt strings
* ``splits``: split labels (the default evaluated label is ``test``)
* ``teacher_embeddings``: raw LLM2Vec outputs shaped ``[N, 4096]``
* ``targets``: concatenated bias-free root/body conditions shaped ``[N, 2048]``

The cache must contain the complete ``metadata.json`` written by
``cache_teacher.py``. The evaluator validates the full manifest, shard hashes,
counts, splits, tensor shapes/dtypes, and finite values before inference.

Example:

    uv run python scripts/minilm/evaluate_conditions.py \
        --teacher-cache artifacts/teacher-cache \
        --student-path artifacts/minilm-ardy-core40 \
        --device cuda --dtype bfloat16 \
        --output artifacts/minilm-ardy-core40/test-condition-metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ardy.minilm_teacher_cache import (
    VALID_SPLITS,
    load_teacher_cache,
    teacher_cache_fingerprint,
)
from ardy.model.minilm_encoder import MiniLMArdyEncoder

DEFAULT_STUDENT_PATH = "artifacts/minilm-ardy-core40"
EXPECTED_TEACHER_DIM = 4096
EXPECTED_CONDITION_DIM = 1024


@dataclass
class _MetricAccumulator:
    example_count: int = 0
    element_count: int = 0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    target_squared_sum: float = 0.0
    cosine_values: list[float] = field(default_factory=list)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}")
        if prediction.ndim != 2:
            raise ValueError(f"metrics require [N, D] tensors, got {tuple(prediction.shape)}")

        prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
        target = target.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(prediction).all():
            raise ValueError("student prediction contains NaN or infinity")
        if not torch.isfinite(target).all():
            raise ValueError("teacher target contains NaN or infinity")

        error = prediction - target
        self.example_count += prediction.shape[0]
        self.element_count += prediction.numel()
        self.squared_error_sum += error.square().sum().item()
        self.absolute_error_sum += error.abs().sum().item()
        self.target_squared_sum += target.square().sum().item()
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-12)
        self.cosine_values.extend(cosine.tolist())

    def finalize(self) -> dict[str, Any]:
        if self.example_count == 0 or self.element_count == 0:
            raise ValueError("cannot finalize empty metrics")
        cosine = torch.tensor(self.cosine_values, dtype=torch.float64)
        rmse = math.sqrt(self.squared_error_sum / self.element_count)
        target_rms = math.sqrt(self.target_squared_sum / self.element_count)
        return {
            "examples": self.example_count,
            "elements": self.element_count,
            "cosine": {
                "mean": cosine.mean().item(),
                "p5": torch.quantile(cosine, 0.05).item(),
            },
            "rmse": rmse,
            "nrmse": None if target_rms == 0.0 else rmse / target_rms,
            "mae": self.absolute_error_sum / self.element_count,
            "target_rms": target_rms,
        }


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device={requested!r} requested, but CUDA is unavailable")
    return requested


def _selected_indices(splits: Sequence[str], split: str) -> list[int]:
    return [index for index, value in enumerate(splits) if value == split]


def _batches(values: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _normalize_student_output(output: Any, expected_rows: int, output_dim: int) -> torch.Tensor:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("MiniLM production wrapper must return (tensor, lengths)")
    conditions, lengths = output
    if not isinstance(conditions, torch.Tensor):
        raise TypeError(f"student returned {type(conditions).__name__}, expected Tensor")
    if conditions.ndim == 3 and conditions.shape[1] == 1:
        conditions = conditions[:, 0, :]
    if conditions.shape != (expected_rows, output_dim):
        raise ValueError(f"student returned shape {tuple(conditions.shape)}, expected ({expected_rows}, {output_dim})")
    normalized_lengths = lengths.tolist() if hasattr(lengths, "tolist") else lengths
    if normalized_lengths != [1] * expected_rows:
        raise ValueError(f"student returned unexpected token lengths {normalized_lengths!r}")
    return conditions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    cache = load_teacher_cache(
        args.teacher_cache,
        expected_teacher_dim=args.teacher_dim,
        expected_target_dim=2 * EXPECTED_CONDITION_DIM,
        keep_teacher_embeddings=False,
        keep_targets=True,
    )
    shard_paths = cache.shard_paths
    metadata_path = cache.metadata_path
    cache_metadata = cache.metadata

    load_started = time.perf_counter()
    student = MiniLMArdyEncoder(
        model_name_or_path=args.student_path,
        dtype=args.dtype,
        device=device,
        expected_ardy_model=args.expected_ardy_model,
        max_length=args.max_length,
    )
    student_load_seconds = time.perf_counter() - load_started
    artifact_config = student.artifact_config
    condition_dim = int(artifact_config["condition_dim"])
    output_dim = int(artifact_config["output_dim"])
    if condition_dim != EXPECTED_CONDITION_DIM or output_dim != 2 * condition_dim:
        raise ValueError(
            "evaluation requires an ARDY dual-condition artifact with "
            f"condition_dim={EXPECTED_CONDITION_DIM}, got condition_dim={condition_dim}, output_dim={output_dim}"
        )
    declared_target_dim = cache_metadata["target_dim"]
    if declared_target_dim != output_dim:
        raise ValueError(
            f"{metadata_path}: target_dim is {declared_target_dim}, but the student output_dim is {output_dim}"
        )

    metrics = {
        "root": _MetricAccumulator(),
        "body": _MetricAccumulator(),
        "overall": _MetricAccumulator(),
    }
    total_cache_examples = 0
    evaluated_examples = 0
    evaluated_shards = 0
    inference_seconds = 0.0

    for shard in cache.shards:
        texts = shard.texts
        splits = shard.splits
        targets = shard.targets
        if targets is None:
            raise RuntimeError("strict teacher-cache loader did not retain targets")
        total_cache_examples += len(texts)
        indices = _selected_indices(splits, args.split)
        if args.max_samples is not None:
            remaining = args.max_samples - evaluated_examples
            if remaining <= 0:
                break
            indices = indices[:remaining]
        if not indices:
            continue
        evaluated_shards += 1

        for batch_indices in _batches(indices, args.batch_size):
            batch_texts = [texts[index] for index in batch_indices]
            batch_targets = targets[torch.tensor(batch_indices, dtype=torch.long)]
            started = time.perf_counter()
            student_output = student(batch_texts)
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            inference_seconds += time.perf_counter() - started
            prediction = _normalize_student_output(student_output, len(batch_texts), output_dim)

            metrics["root"].update(
                prediction[:, :condition_dim],
                batch_targets[:, :condition_dim],
            )
            metrics["body"].update(
                prediction[:, condition_dim:],
                batch_targets[:, condition_dim:],
            )
            metrics["overall"].update(prediction, batch_targets)
            evaluated_examples += len(batch_texts)

    if evaluated_examples == 0:
        raise ValueError(f"no examples with split {args.split!r} were found in {len(shard_paths)} teacher-cache shards")

    artifact_summary = {
        key: artifact_config.get(key)
        for key in (
            "format_version",
            "artifact_fingerprint",
            "base_model",
            "condition_dim",
            "output_dim",
            "compatible_ardy_models",
            "max_length",
        )
        if key in artifact_config
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "teacher_cache": str(args.teacher_cache),
            "teacher_dim": args.teacher_dim,
            "student_path": str(args.student_path),
            "split": args.split,
            "device_requested": args.device,
            "device_resolved": device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "expected_ardy_model": args.expected_ardy_model,
            "nrmse_definition": "RMSE divided by target RMS over the same elements",
        },
        "dataset": {
            "metadata_path": None if metadata_path is None else str(metadata_path),
            "metadata": cache_metadata,
            "shards": [str(path) for path in shard_paths],
            "shard_count": len(shard_paths),
            "evaluated_shard_count": evaluated_shards,
            "cache_fingerprint": teacher_cache_fingerprint(cache),
            "examples_seen_in_loaded_shards": total_cache_examples,
            "evaluated_examples": evaluated_examples,
        },
        "student_artifact": artifact_summary,
        "timing_seconds": {
            "student_load": student_load_seconds,
            "student_inference_total": inference_seconds,
            "student_inference_per_example": inference_seconds / evaluated_examples,
        },
        "metrics": {name: accumulator.finalize() for name, accumulator in metrics.items()},
    }
    _write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "result": result}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        required=True,
        help="complete teacher-cache directory or its metadata.json",
    )
    parser.add_argument("--student-path", type=Path, default=Path(DEFAULT_STUDENT_PATH))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=VALID_SPLITS, default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--teacher-dim", type=int, default=EXPECTED_TEACHER_DIM)
    parser.add_argument("--expected-ardy-model", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.teacher_dim < 1:
        parser.error("--teacher-dim must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
