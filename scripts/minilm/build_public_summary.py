#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Build the prompt-free public MiniLM summary from local raw reports.

The 50- and 100-epoch candidates are selected using validation metrics before
the winner-only test and benchmark reports are inspected. Raw reports remain
under the Git-ignored ``artifacts/`` tree; only the allowlisted aggregate is
intended for ``reports/minilm_core40_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PureWindowsPath
from typing import Any

from ardy.minilm_teacher_cache import validated_teacher_lineage

_SHA256_LENGTH = 64
_SELECTION_METRIC = "mean(root_cosine, body_cosine)"
_FP32_SELECTION_METRIC = "mean(root.cosine.mean, body.cosine.mean)"
_SHARED_TRAJECTORY_PREFIX_EPOCHS = 50
_CONDITION_DIM = 1024
_TEACHER_DIM = 4096
_FINAL_BATCH_SIZE = 64
_FINAL_MOTION_SOURCES = ("overview_description", "events.description")
_FINAL_MOTION_PROMPT_COUNT = 64
_FINAL_MOTION_SEEDS = (0, 1, 2)
_FINAL_MOTION_DURATION_SECONDS = 4.0
_FINAL_MOTION_FPS = 20.0
_FINAL_MOTION_FRAMES = 80
_FINAL_MOTION_DIFFUSION_STEPS = 10
_FINAL_MOTION_CFG_WEIGHT = (2.0, 2.0)
_FINAL_MOTION_SELECTION_ALGORITHM = "evenly_spaced_numpy_linspace_v1"
_MOTION_METRIC_FIELDS = (
    "root_ade_m",
    "root_fde_m",
    "global_mpjpe_m",
    "root_aligned_mpjpe_m",
    "joint_velocity_error_m_per_s",
    "motion_cosine",
    "heading_mae_deg",
    "foot_contact_agreement",
    "foot_contact_macro_f1",
    "foot_contact_macro_iou",
)
_NORMALIZED_MOTION_FIELDS = tuple(
    f"{metric}_vs_teacher_diversity"
    for metric in (
        "root_ade_m",
        "root_fde_m",
        "global_mpjpe_m",
        "root_aligned_mpjpe_m",
        "joint_velocity_error_m_per_s",
    )
)
_BENCHMARK_VERSION_FIELDS = (
    "python",
    "torch",
    "ardy",
    "transformers",
    "peft",
    "safetensors",
)
_BENCHMARK_MEMORY_FIELDS = (
    "rss_current_bytes",
    "rss_peak_bytes",
    "mem_available_before_bytes",
    "mem_available_after_bytes",
    "mem_available_min_bytes",
    "mem_available_drop_bytes",
    "cuda_allocated_current_bytes",
    "cuda_allocated_peak_bytes",
    "cuda_reserved_current_bytes",
    "cuda_reserved_peak_bytes",
)
_PROMPT_TEXT_KEYS = frozenset(
    (
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "text",
        "texts",
        "caption",
        "captions",
        "worst_prompt",
        "worst_prompts",
    )
)
_RECORD_ID_KEYS = frozenset(
    (
        "record_id",
        "record_ids",
        "timeline_id",
        "timeline_ids",
        "take_id",
        "take_ids",
        "actor_id",
        "actor_ids",
    )
)
_TRAINING_CONFIGURATION_FIELDS = (
    "seed",
    "train_examples",
    "validation_examples",
    "test_examples",
    "epochs",
    "head_warmup_epochs",
    "batch_size",
    "adapter_dim",
    "pooling_mode",
    "backbone_lr",
    "head_lr",
    "weight_decay",
    "warmup_ratio",
    "lr_schedule_epochs",
    "lr_schedule_steps",
    "warmup_steps",
    "cosine_weight",
    "relational_weight",
    "train_max_length",
    "runtime_max_length",
    "num_workers",
    "device",
    "device_details",
    "bf16_autocast",
    "deterministic_algorithms",
    "cublas_workspace_config",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "train_batches_per_epoch",
    "optimizer_updates",
    "base_model",
    "base_model_revision",
    "ardy_model",
    "runtime_versions",
)
_TRAINING_VALIDATION_FIELDS = tuple(
    f"{part}_{metric}"
    for part in ("root", "body")
    for metric in ("cosine", "rmse", "mae", "nrmse")
)


@dataclass(frozen=True)
class LoadedReport:
    path: Path
    value: dict[str, Any]
    sha256: str
    size_bytes: int

    def public_identity(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class Candidate:
    label: str
    epochs: int
    report: dict[str, Any]
    public: dict[str, Any]

    @property
    def score(self) -> float:
        return float(self.report["best_validation_score"])

    @property
    def artifact_fingerprint(self) -> str:
        return str(self.report["artifact_fingerprint"])


@dataclass(frozen=True)
class ConditionEvaluation:
    candidate: Candidate
    split: str
    score: float
    lineage: dict[str, Any]
    public: dict[str, Any]


def _load_report(path: Path) -> LoadedReport:
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid report JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path}: report must be a JSON object")
    return LoadedReport(
        path=path,
        value=value,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(
    value: dict[str, Any],
    expected: tuple[str, ...] | frozenset[str],
    field: str,
) -> None:
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{field} keys differ: missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )


def _bounded(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    result = _finite(value, field)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{field} must be between {minimum} and {maximum}, got {result}"
        )
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative, got {result}")
    return result


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _validate_public_value(value: Any, field: str = "summary") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = key.casefold().replace("-", "_")
            if normalized_key in _PROMPT_TEXT_KEYS:
                raise ValueError(f"{field}.{key} must not contain prompt text")
            if normalized_key in _RECORD_ID_KEYS:
                raise ValueError(f"{field}.{key} must not contain dataset record IDs")
            _validate_public_value(item, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_public_value(item, f"{field}[{index}]")
        return
    if not isinstance(value, str):
        return
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith(("file://", "~/", "~\\"))
    ):
        raise ValueError(f"{field} must not contain an absolute local path")


def _picked(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        field: _json_copy(source[field])
        for field in fields
        if field in source
    }


def _without_timing_seconds(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing_seconds(item)
            for key, item in value.items()
            if key != "seconds"
        }
    if isinstance(value, list):
        return [_without_timing_seconds(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _condition_metric(
    value: Any,
    *,
    field: str,
    expected_examples: int,
    expected_width: int,
) -> dict[str, Any]:
    metric = _object(value, field)
    _exact_keys(
        metric,
        (
            "examples",
            "elements",
            "cosine",
            "rmse",
            "nrmse",
            "mae",
            "target_rms",
        ),
        field,
    )
    examples = _integer(metric["examples"], f"{field}.examples", positive=True)
    if examples != expected_examples:
        raise ValueError(
            f"{field}.examples must be {expected_examples}, got {examples}"
        )
    elements = _integer(metric["elements"], f"{field}.elements", positive=True)
    expected_elements = expected_examples * expected_width
    if elements != expected_elements:
        raise ValueError(
            f"{field}.elements must be {expected_elements}, got {elements}"
        )
    cosine = _object(metric["cosine"], f"{field}.cosine")
    _exact_keys(cosine, ("mean", "p5"), f"{field}.cosine")
    target_rms = _nonnegative(metric["target_rms"], f"{field}.target_rms")
    if target_rms == 0.0:
        raise ValueError(f"{field}.target_rms must be positive")
    nrmse = _nonnegative(metric["nrmse"], f"{field}.nrmse")
    return {
        "examples": examples,
        "elements": elements,
        "cosine": {
            "mean": _bounded(
                cosine["mean"],
                f"{field}.cosine.mean",
                minimum=-1.0,
                maximum=1.0,
            ),
            "p5": _bounded(
                cosine["p5"],
                f"{field}.cosine.p5",
                minimum=-1.0,
                maximum=1.0,
            ),
        },
        "rmse": _nonnegative(metric["rmse"], f"{field}.rmse"),
        "nrmse": nrmse,
        "mae": _nonnegative(metric["mae"], f"{field}.mae"),
        "target_rms": target_rms,
    }


def _condition_metrics(
    value: Any,
    *,
    field: str,
    expected_examples: int,
) -> dict[str, Any]:
    metrics = _object(value, field)
    _exact_keys(metrics, ("root", "body", "overall"), field)
    return {
        "root": _condition_metric(
            metrics["root"],
            field=f"{field}.root",
            expected_examples=expected_examples,
            expected_width=_CONDITION_DIM,
        ),
        "body": _condition_metric(
            metrics["body"],
            field=f"{field}.body",
            expected_examples=expected_examples,
            expected_width=_CONDITION_DIM,
        ),
        "overall": _condition_metric(
            metrics["overall"],
            field=f"{field}.overall",
            expected_examples=expected_examples,
            expected_width=2 * _CONDITION_DIM,
        ),
    }


def _selected_prompt_sha256(texts: list[str]) -> str:
    encoded = json.dumps(
        texts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_number_or_none(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field)


def _strict_motion_metrics(value: Any, field: str) -> dict[str, float | None]:
    metrics = _object(value, field)
    _exact_keys(metrics, _MOTION_METRIC_FIELDS, field)
    result: dict[str, float | None] = {}
    for name in _MOTION_METRIC_FIELDS:
        metric = _metric_number_or_none(metrics[name], f"{field}.{name}")
        if metric is not None:
            if name in {
                "motion_cosine",
            }:
                if not -1.0 <= metric <= 1.0:
                    raise ValueError(f"{field}.{name} must be between -1 and 1")
            elif name in {
                "foot_contact_agreement",
                "foot_contact_macro_f1",
                "foot_contact_macro_iou",
            }:
                if not 0.0 <= metric <= 1.0:
                    raise ValueError(f"{field}.{name} must be between 0 and 1")
            elif metric < 0.0:
                raise ValueError(f"{field}.{name} must be non-negative")
        result[name] = metric
    return result


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return None if not finite else math.fsum(finite) / len(finite)


def _validate_metric_summary(
    observed: Any,
    rows: list[dict[str, float | None]],
    *,
    field: str,
) -> dict[str, float | None]:
    summary = _strict_motion_metrics(observed, field)
    for name in _MOTION_METRIC_FIELDS:
        expected = _mean_optional([row[name] for row in rows])
        actual = summary[name]
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"{field}.{name} is inconsistent with cases")
            continue
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{field}.{name} is inconsistent with cases")
    return summary


def _training_candidate(
    report: dict[str, Any],
    *,
    label: str,
    expected_epochs: int,
) -> Candidate:
    if report.get("schema_version") != 1:
        raise ValueError(f"{label} training report schema_version must be 1")
    selection = _object(report.get("selection"), f"{label}.selection")
    expected_selection = {
        "split": "val",
        "metric": _SELECTION_METRIC,
        "test_evaluated": False,
    }
    if selection != expected_selection:
        raise ValueError(
            f"{label} candidate selection must be validation-only: "
            f"expected {expected_selection!r}, got {selection!r}"
        )
    configuration = _object(
        report.get("configuration"),
        f"{label}.configuration",
    )
    if configuration.get("epochs") != expected_epochs:
        raise ValueError(
            f"{label} configuration.epochs must be {expected_epochs}"
        )
    if configuration.get("bf16_autocast") is not True:
        raise ValueError(
            f"{label} configuration.bf16_autocast must be true so the "
            "within-run checkpoint-selection precision is explicit"
        )
    history = _list(report.get("history"), f"{label}.history")
    if len(history) != expected_epochs:
        raise ValueError(
            f"{label} history contains {len(history)} epochs, expected "
            f"{expected_epochs}"
        )
    observed_epochs = [
        _integer(
            _object(row, f"{label}.history[{index}]").get("epoch"),
            f"{label}.history[{index}].epoch",
            positive=True,
        )
        for index, row in enumerate(history)
    ]
    if observed_epochs != list(range(1, expected_epochs + 1)):
        raise ValueError(f"{label} training history epochs are not contiguous")

    validation = _object(report.get("validation"), f"{label}.validation")
    _exact_keys(
        validation,
        _TRAINING_VALIDATION_FIELDS,
        f"{label}.validation",
    )
    for name in _TRAINING_VALIDATION_FIELDS:
        value = _finite(validation[name], f"{label}.validation.{name}")
        if name.endswith("_cosine"):
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"{label}.validation.{name} must be between -1 and 1"
                )
        elif value < 0.0:
            raise ValueError(
                f"{label}.validation.{name} must be non-negative"
            )
    root_cosine = _finite(
        validation.get("root_cosine"),
        f"{label}.validation.root_cosine",
    )
    body_cosine = _finite(
        validation.get("body_cosine"),
        f"{label}.validation.body_cosine",
    )
    score = _finite(
        report.get("best_validation_score"),
        f"{label}.best_validation_score",
    )
    expected_score = 0.5 * (root_cosine + body_cosine)
    if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label} best_validation_score is inconsistent with root/body "
            "validation cosine"
        )
    best_epoch = _integer(
        report.get("best_epoch"),
        f"{label}.best_epoch",
        positive=True,
    )
    if best_epoch > expected_epochs:
        raise ValueError(f"{label}.best_epoch exceeds the configured epochs")
    best_history_validation = _object(
        _object(
            history[best_epoch - 1],
            f"{label}.history[{best_epoch - 1}]",
        ).get("validation"),
        f"{label}.history[{best_epoch - 1}].validation",
    )
    if best_history_validation != validation:
        raise ValueError(
            f"{label} validation does not match the selected history epoch"
        )
    history_scores = []
    for index, row in enumerate(history):
        row_validation = _object(
            _object(row, f"{label}.history[{index}]").get("validation"),
            f"{label}.history[{index}].validation",
        )
        _exact_keys(
            row_validation,
            _TRAINING_VALIDATION_FIELDS,
            f"{label}.history[{index}].validation",
        )
        for name in _TRAINING_VALIDATION_FIELDS:
            _finite(
                row_validation[name],
                f"{label}.history[{index}].validation.{name}",
            )
        history_scores.append(
            0.5
            * (
                _finite(
                    row_validation.get("root_cosine"),
                    f"{label}.history[{index}].validation.root_cosine",
                )
                + _finite(
                    row_validation.get("body_cosine"),
                    f"{label}.history[{index}].validation.body_cosine",
                )
            )
        )
    observed_best_score = max(history_scores)
    observed_best_epoch = history_scores.index(observed_best_score) + 1
    if not math.isclose(
        score,
        observed_best_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or best_epoch != observed_best_epoch:
        raise ValueError(
            f"{label} best validation selection is inconsistent with history"
        )

    fingerprint = _sha256(
        report.get("artifact_fingerprint"),
        f"{label}.artifact_fingerprint",
    )
    cache_fingerprint = _sha256(
        report.get("teacher_cache_fingerprint"),
        f"{label}.teacher_cache_fingerprint",
    )
    payload_size = _integer(
        report.get("artifact_payload_size_bytes"),
        f"{label}.artifact_payload_size_bytes",
        positive=True,
    )
    public = {
        "configured_epochs": expected_epochs,
        "within_run_checkpoint_selection": {
            "precision": "bfloat16_autocast",
            "split": "val",
            "metric": _SELECTION_METRIC,
            "best_epoch": best_epoch,
            "best_validation_score": score,
            "validation": _json_copy(validation),
        },
        "artifact_fingerprint": fingerprint,
        "artifact_payload_size_bytes": payload_size,
        "teacher_cache_fingerprint": cache_fingerprint,
        "elapsed_seconds": _finite(
            report.get("elapsed_seconds"),
            f"{label}.elapsed_seconds",
        ),
        "configuration": _picked(
            configuration,
            _TRAINING_CONFIGURATION_FIELDS,
        ),
    }
    return Candidate(
        label=label,
        epochs=expected_epochs,
        report=report,
        public=public,
    )


def _validate_candidate_pair(left: Candidate, right: Candidate) -> None:
    candidates_by_epochs = {
        candidate.epochs: candidate for candidate in (left, right)
    }
    if set(candidates_by_epochs) != {
        _SHARED_TRAJECTORY_PREFIX_EPOCHS,
        100,
    }:
        raise ValueError("training candidates must be the 50e and 100e runs")
    candidate_50 = candidates_by_epochs[_SHARED_TRAJECTORY_PREFIX_EPOCHS]
    candidate_100 = candidates_by_epochs[100]

    for field in ("base_model", "ardy_model", "teacher_cache_fingerprint"):
        if left.report.get(field) != right.report.get(field):
            raise ValueError(f"training candidates differ in {field}")
    left_config = dict(_object(left.report["configuration"], "50e.configuration"))
    right_config = dict(
        _object(right.report["configuration"], "100e.configuration")
    )
    for key in (
        "epochs",
        "optimizer_updates",
        "selected_epoch",
        "best_validation_score",
    ):
        left_config.pop(key, None)
        right_config.pop(key, None)
    if left_config != right_config:
        raise ValueError(
            "50e and 100e training configurations differ outside the allowed "
            "epoch/update/selection fields"
        )

    history_50 = _list(candidate_50.report.get("history"), "50e.history")
    history_100 = _list(candidate_100.report.get("history"), "100e.history")
    normalized_50 = _without_timing_seconds(history_50)
    normalized_100_prefix = _without_timing_seconds(
        history_100[:_SHARED_TRAJECTORY_PREFIX_EPOCHS]
    )
    if _canonical_json(normalized_50) != _canonical_json(
        normalized_100_prefix
    ):
        raise ValueError(
            "50e history must exactly match the first 50 epochs of the 100e "
            "history after removing timing-only 'seconds' fields"
        )


def _select_winner(
    left: Candidate,
    right: Candidate,
    *,
    fp32_scores: dict[str, float],
) -> Candidate:
    """Select saved artifacts by complete FP32 validation only.

    Each saved artifact was selected from its own epoch history by BF16
    validation during training. This second-stage comparison deliberately
    ignores those BF16 scores and compares only the two saved artifacts under
    the same complete FP32 validation protocol. An exact tie favors 50 epochs.
    """

    _validate_candidate_pair(left, right)
    expected_labels = {left.label, right.label}
    if set(fp32_scores) != expected_labels:
        raise ValueError(
            "FP32 validation scores must identify exactly the two candidates"
        )
    scores = {
        label: _finite(score, f"fp32_scores.{label}")
        for label, score in fp32_scores.items()
    }
    if scores[right.label] > scores[left.label]:
        return right
    return left


def _condition_evaluation(
    report: dict[str, Any],
    *,
    candidate: Candidate,
    expected_split: str,
    expected_lineage: dict[str, Any] | None = None,
) -> ConditionEvaluation:
    if report.get("schema_version") != 1:
        raise ValueError("condition report schema_version must be 1")
    configuration = _object(
        report.get("configuration"),
        "condition.configuration",
    )
    if configuration.get("split") != expected_split:
        raise ValueError(
            f"condition report must evaluate split={expected_split!r}"
        )
    if configuration.get("max_samples") is not None:
        raise ValueError(
            f"condition report must evaluate the complete {expected_split} split"
        )
    if configuration.get("dtype") != "float32":
        raise ValueError("condition report dtype must be 'float32'")
    if configuration.get("teacher_dim") != _TEACHER_DIM:
        raise ValueError(
            f"condition report teacher_dim must be {_TEACHER_DIM}"
        )
    if configuration.get("batch_size") != _FINAL_BATCH_SIZE:
        raise ValueError(
            f"condition report batch_size must be {_FINAL_BATCH_SIZE}"
        )
    if configuration.get("max_length") is not None:
        raise ValueError(
            "condition report must use the artifact's saved max_length"
        )
    if configuration.get("expected_ardy_model") != candidate.report.get(
        "ardy_model"
    ):
        raise ValueError(
            "condition report expected_ardy_model differs from the candidate"
        )
    nrmse_definition = configuration.get("nrmse_definition")
    if not isinstance(nrmse_definition, str) or not nrmse_definition:
        raise ValueError("condition report nrmse_definition is missing")

    dataset = _object(report.get("dataset"), "condition.dataset")
    metadata = _object(dataset.get("metadata"), "condition.dataset.metadata")
    lineage = validated_teacher_lineage(metadata)
    if (
        expected_lineage is not None
        and _canonical_json(lineage) != _canonical_json(expected_lineage)
    ):
        raise ValueError("condition report teacher-cache lineage differs")
    cache_fingerprint = _sha256(
        dataset.get("cache_fingerprint"),
        "condition.dataset.cache_fingerprint",
    )
    if cache_fingerprint != candidate.report["teacher_cache_fingerprint"]:
        raise ValueError(
            "condition report uses a different teacher cache than the candidate"
        )
    examples_seen = _integer(
        dataset.get("examples_seen_in_loaded_shards"),
        "condition.dataset.examples_seen_in_loaded_shards",
        positive=True,
    )
    if examples_seen != lineage["count"]:
        raise ValueError(
            "condition report did not load the complete teacher cache"
        )
    evaluated_examples = _integer(
        dataset.get("evaluated_examples"),
        "condition.dataset.evaluated_examples",
        positive=True,
    )
    expected_examples = lineage["split_counts"][expected_split]
    if evaluated_examples != expected_examples:
        raise ValueError(
            f"condition report does not cover the complete lineage "
            f"{expected_split} split"
        )

    student = _object(
        report.get("student_artifact"),
        "condition.student_artifact",
    )
    if student.get("artifact_fingerprint") != candidate.artifact_fingerprint:
        raise ValueError(
            "condition report does not evaluate the expected candidate"
        )
    if student.get("base_model") != candidate.report.get("base_model"):
        raise ValueError(
            "condition report student base model differs from candidate"
        )
    if (
        student.get("condition_dim") != _CONDITION_DIM
        or student.get("output_dim") != 2 * _CONDITION_DIM
    ):
        raise ValueError("condition report has an incompatible student output")
    runtime_max_length = candidate.report["configuration"].get(
        "runtime_max_length"
    )
    if student.get("max_length") != runtime_max_length:
        raise ValueError(
            "condition report student max_length differs from the candidate"
        )
    metrics = _condition_metrics(
        report.get("metrics"),
        field="condition.metrics",
        expected_examples=evaluated_examples,
    )
    score = 0.5 * (
        metrics["root"]["cosine"]["mean"]
        + metrics["body"]["cosine"]["mean"]
    )
    public = {
        "split": expected_split,
        "evaluated_examples": evaluated_examples,
        "artifact_fingerprint": candidate.artifact_fingerprint,
        "configuration": {
            "teacher_dim": _TEACHER_DIM,
            "dtype": "float32",
            "batch_size": _FINAL_BATCH_SIZE,
            "max_length": None,
            "max_samples": None,
            "expected_ardy_model": candidate.report["ardy_model"],
            "nrmse_definition": nrmse_definition,
        },
        "selection_metric": _FP32_SELECTION_METRIC,
        "selection_score": score,
        "metrics": metrics,
    }
    return ConditionEvaluation(
        candidate=candidate,
        split=expected_split,
        score=score,
        lineage=lineage,
        public=public,
    )


def _dataset_public(
    evaluation: ConditionEvaluation,
) -> dict[str, Any]:
    lineage = evaluation.lineage
    corpus = lineage["corpus_provenance"]
    return {
        "teacher_cache_fingerprint": evaluation.candidate.report[
            "teacher_cache_fingerprint"
        ],
        "prompt_manifest_sha256": lineage["input_sha256"],
        "prompt_provenance_sidecar_sha256": lineage["input_metadata_sha256"],
        "total_prompts": lineage["count"],
        "split_counts": _json_copy(lineage["split_counts"]),
        "corpus_provenance": _json_copy(corpus),
        "teacher": _picked(
            lineage,
            (
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
            ),
        ),
    }


def _motion_summary(
    report: dict[str, Any],
    *,
    winner: Candidate,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise ValueError("motion report schema_version must be 1")
    scope = _object(report.get("scope"), "motion.scope")
    if scope.get("split") != "test":
        raise ValueError("motion report must evaluate split='test'")
    if scope.get("student_artifact_fingerprint") != winner.artifact_fingerprint:
        raise ValueError(
            "motion report does not evaluate the validation-selected winner"
        )
    if scope.get("teacher_cache_fingerprint") != winner.report[
        "teacher_cache_fingerprint"
    ]:
        raise ValueError("motion report uses a different teacher cache")
    if scope.get("checkpoint_sha256") != lineage["checkpoint_sha256"]:
        raise ValueError("motion report uses a different ARDY checkpoint")
    if scope.get("prompt_manifest_sha256") != lineage["input_sha256"]:
        raise ValueError("motion report uses a different prompt manifest")
    if scope.get("resolved_model") != winner.report.get("ardy_model"):
        raise ValueError("motion report uses a different ARDY model")
    held_out = _integer(
        scope.get("held_out_test_prompts_available"),
        "motion.scope.held_out_test_prompts_available",
        positive=True,
    )
    if held_out != lineage["split_counts"]["test"]:
        raise ValueError("motion report test count differs from cache lineage")
    source_filter = _list(
        scope.get("source_filter"),
        "motion.scope.source_filter",
    )
    if source_filter != list(_FINAL_MOTION_SOURCES):
        raise ValueError(
            "motion report source_filter must contain both Timeline sources "
            "in canonical order"
        )
    eligible = _integer(
        scope.get("eligible_test_prompts"),
        "motion.scope.eligible_test_prompts",
        positive=True,
    )
    if not _FINAL_MOTION_PROMPT_COUNT <= eligible <= held_out:
        raise ValueError(
            "motion report eligible prompt count is inconsistent with the "
            "final protocol"
        )
    prompt_count = _integer(
        scope.get("prompts"),
        "motion.scope.prompts",
        positive=True,
    )
    if prompt_count != _FINAL_MOTION_PROMPT_COUNT:
        raise ValueError(
            f"motion report must evaluate {_FINAL_MOTION_PROMPT_COUNT} prompts"
        )
    seeds = _list(scope.get("seeds"), "motion.scope.seeds")
    if seeds != list(_FINAL_MOTION_SEEDS):
        raise ValueError(
            f"motion report seeds must be {list(_FINAL_MOTION_SEEDS)}"
        )
    if scope.get("prompt_selection_algorithm") != (
        _FINAL_MOTION_SELECTION_ALGORITHM
    ):
        raise ValueError("motion report prompt selection algorithm is invalid")
    selected_digest = _sha256(
        scope.get("selected_prompt_sha256"),
        "motion.scope.selected_prompt_sha256",
    )
    if (
        _finite(
            scope.get("duration_seconds"),
            "motion.scope.duration_seconds",
        )
        != _FINAL_MOTION_DURATION_SECONDS
    ):
        raise ValueError("motion report duration_seconds must be 4.0")
    if _finite(scope.get("fps"), "motion.scope.fps") != _FINAL_MOTION_FPS:
        raise ValueError("motion report fps must be 20.0")
    if scope.get("num_frames") != _FINAL_MOTION_FRAMES:
        raise ValueError("motion report num_frames must be 80")
    if scope.get("diffusion_steps") != _FINAL_MOTION_DIFFUSION_STEPS:
        raise ValueError("motion report diffusion_steps must be 10")
    cfg_weight = _list(
        scope.get("cfg_weight"),
        "motion.scope.cfg_weight",
    )
    if cfg_weight != list(_FINAL_MOTION_CFG_WEIGHT):
        raise ValueError("motion report cfg_weight must be [2.0, 2.0]")
    if scope.get("postprocess_applied") is not False:
        raise ValueError("motion report must use raw inverse motion output")
    if scope.get("student_dtype") != "float32":
        raise ValueError("final motion report student_dtype must be float32")

    determinism = _object(
        scope.get("determinism"),
        "motion.scope.determinism",
    )
    expected_determinism = {
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    if determinism != expected_determinism:
        raise ValueError(
            "motion report determinism settings differ from the final protocol"
        )
    repeatability = _object(
        scope.get("repeatability_check"),
        "motion.scope.repeatability_check",
    )
    expected_repeatability = {
        "enabled": True,
        "exact_equal": True,
        "prompt_index": 0,
        "seed": _FINAL_MOTION_SEEDS[0],
        "compared_keys": [
            "root_positions",
            "posed_joints",
            "global_root_heading",
            "foot_contacts",
        ],
    }
    if repeatability != expected_repeatability:
        raise ValueError(
            "motion report must pass the exact first-case repeatability check"
        )
    expected_seed_pairs = list(combinations(_FINAL_MOTION_SEEDS, 2))
    if scope.get("teacher_diversity_seed_pairs") != len(
        expected_seed_pairs
    ):
        raise ValueError(
            "motion report teacher diversity pair count is inconsistent"
        )

    cases = _list(report.get("cases"), "motion.cases")
    if len(cases) != prompt_count * len(seeds):
        raise ValueError("motion report case count is inconsistent")
    expected_case_order = [
        (prompt_index, seed)
        for prompt_index in range(prompt_count)
        for seed in _FINAL_MOTION_SEEDS
    ]
    prompt_texts: list[str | None] = [None] * prompt_count
    paired_rows: list[dict[str, float | None]] = []
    expected_case_keys = {
        "prompt_index",
        "text",
        "seed",
        "teacher_generation_seconds",
        "student_generation_seconds",
        *_MOTION_METRIC_FIELDS,
    }
    for row_index, (value, expected) in enumerate(
        zip(cases, expected_case_order, strict=True)
    ):
        case = _object(value, f"motion.cases[{row_index}]")
        _exact_keys(
            case,
            frozenset(expected_case_keys),
            f"motion.cases[{row_index}]",
        )
        prompt_index = _integer(
            case["prompt_index"],
            f"motion.cases[{row_index}].prompt_index",
        )
        seed = _integer(
            case["seed"],
            f"motion.cases[{row_index}].seed",
        )
        if (prompt_index, seed) != expected:
            raise ValueError("motion report cases are not in canonical order")
        text = case["text"]
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"motion.cases[{row_index}].text must be non-empty"
            )
        if prompt_texts[prompt_index] is None:
            prompt_texts[prompt_index] = text
        elif prompt_texts[prompt_index] != text:
            raise ValueError(
                "motion report changes prompt text between seeds"
            )
        _nonnegative(
            case["teacher_generation_seconds"],
            f"motion.cases[{row_index}].teacher_generation_seconds",
        )
        _nonnegative(
            case["student_generation_seconds"],
            f"motion.cases[{row_index}].student_generation_seconds",
        )
        paired_rows.append(
            _strict_motion_metrics(
                {name: case[name] for name in _MOTION_METRIC_FIELDS},
                f"motion.cases[{row_index}].metrics",
            )
        )
    ordered_prompt_texts = [
        text
        for text in prompt_texts
        if text is not None
    ]
    if len(ordered_prompt_texts) != prompt_count:
        raise ValueError("motion report does not cover every selected prompt")
    if _selected_prompt_sha256(ordered_prompt_texts) != selected_digest:
        raise ValueError(
            "motion report selected_prompt_sha256 differs from its cases"
        )
    paired_summary = _validate_metric_summary(
        report.get("paired_teacher_student"),
        paired_rows,
        field="motion.paired_teacher_student",
    )

    diversity_cases = _list(
        report.get("teacher_diversity_cases"),
        "motion.teacher_diversity_cases",
    )
    expected_diversity_order = [
        (prompt_index, seed_a, seed_b)
        for seed_a, seed_b in expected_seed_pairs
        for prompt_index in range(prompt_count)
    ]
    if len(diversity_cases) != len(expected_diversity_order):
        raise ValueError("motion teacher-diversity case count is inconsistent")
    expected_diversity_keys = {
        "prompt_index",
        "text",
        "seed_a",
        "seed_b",
        *_MOTION_METRIC_FIELDS,
    }
    diversity_rows: list[dict[str, float | None]] = []
    for row_index, (value, expected) in enumerate(
        zip(diversity_cases, expected_diversity_order, strict=True)
    ):
        case = _object(
            value,
            f"motion.teacher_diversity_cases[{row_index}]",
        )
        _exact_keys(
            case,
            frozenset(expected_diversity_keys),
            f"motion.teacher_diversity_cases[{row_index}]",
        )
        prompt_index = _integer(
            case["prompt_index"],
            f"motion.teacher_diversity_cases[{row_index}].prompt_index",
        )
        seed_a = _integer(
            case["seed_a"],
            f"motion.teacher_diversity_cases[{row_index}].seed_a",
        )
        seed_b = _integer(
            case["seed_b"],
            f"motion.teacher_diversity_cases[{row_index}].seed_b",
        )
        if (prompt_index, seed_a, seed_b) != expected:
            raise ValueError(
                "motion teacher-diversity cases are not in canonical order"
            )
        if case["text"] != ordered_prompt_texts[prompt_index]:
            raise ValueError(
                "motion teacher-diversity prompt differs from paired cases"
            )
        diversity_rows.append(
            _strict_motion_metrics(
                {name: case[name] for name in _MOTION_METRIC_FIELDS},
                f"motion.teacher_diversity_cases[{row_index}].metrics",
            )
        )
    diversity_summary = _validate_metric_summary(
        report.get("teacher_seed_diversity"),
        diversity_rows,
        field="motion.teacher_seed_diversity",
    )

    normalized = _object(
        report.get("normalized_by_teacher_diversity"),
        "motion.normalized_by_teacher_diversity",
    )
    _exact_keys(
        normalized,
        _NORMALIZED_MOTION_FIELDS,
        "motion.normalized_by_teacher_diversity",
    )
    normalized_public: dict[str, float | None] = {}
    for name in _NORMALIZED_MOTION_FIELDS:
        base_name = name.removesuffix("_vs_teacher_diversity")
        numerator = paired_summary[base_name]
        denominator = diversity_summary[base_name]
        expected_ratio = (
            None
            if numerator is None
            or denominator is None
            or denominator <= 0.0
            else numerator / denominator
        )
        actual_ratio = _metric_number_or_none(
            normalized[name],
            f"motion.normalized_by_teacher_diversity.{name}",
        )
        if actual_ratio is not None and actual_ratio < 0.0:
            raise ValueError(
                f"motion.normalized_by_teacher_diversity.{name} "
                "must be non-negative"
            )
        if expected_ratio is None or actual_ratio is None:
            if expected_ratio is not actual_ratio:
                raise ValueError(
                    f"motion normalized metric {name} is inconsistent"
                )
        elif not math.isclose(
            actual_ratio,
            expected_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"motion normalized metric {name} is inconsistent"
            )
        normalized_public[name] = actual_ratio

    result = {
        "scope": {
            "split": "test",
            "resolved_model": winner.report["ardy_model"],
            "checkpoint_sha256": lineage["checkpoint_sha256"],
            "teacher_cache_fingerprint": winner.report[
                "teacher_cache_fingerprint"
            ],
            "prompt_manifest_sha256": lineage["input_sha256"],
            "source_filter": list(_FINAL_MOTION_SOURCES),
            "eligible_test_prompts": eligible,
            "held_out_test_prompts_available": held_out,
            "prompt_count": prompt_count,
            "prompt_selection_algorithm": (
                _FINAL_MOTION_SELECTION_ALGORITHM
            ),
            "selected_prompt_sha256": selected_digest,
            "seeds": list(_FINAL_MOTION_SEEDS),
            "teacher_diversity_seed_pairs": len(expected_seed_pairs),
            "duration_seconds": _FINAL_MOTION_DURATION_SECONDS,
            "fps": _FINAL_MOTION_FPS,
            "num_frames": _FINAL_MOTION_FRAMES,
            "diffusion_steps": _FINAL_MOTION_DIFFUSION_STEPS,
            "cfg_weight": list(_FINAL_MOTION_CFG_WEIGHT),
            "student_dtype": "float32",
            "postprocess_applied": False,
            "determinism": expected_determinism,
            "repeatability_check": expected_repeatability,
        },
        "paired_teacher_student": paired_summary,
        "teacher_seed_diversity": diversity_summary,
        "normalized_by_teacher_diversity": normalized_public,
    }
    return result


def _latency_percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _latency_summary(
    value: Any,
    field: str,
    *,
    expected_samples: int,
) -> dict[str, Any]:
    summary = _object(value, field)
    _exact_keys(
        summary,
        ("samples", "mean", "min", "p50", "p95", "max", "values"),
        field,
    )
    samples = _integer(summary["samples"], f"{field}.samples", positive=True)
    if samples != expected_samples:
        raise ValueError(
            f"{field}.samples must be {expected_samples}, got {samples}"
        )
    raw_values = _list(summary["values"], f"{field}.values")
    if len(raw_values) != samples:
        raise ValueError(f"{field}.values length differs from samples")
    values = [
        _nonnegative(value, f"{field}.values[{index}]")
        for index, value in enumerate(raw_values)
    ]
    ordered = sorted(values)
    expected = {
        "mean": math.fsum(values) / samples,
        "min": ordered[0],
        "p50": _latency_percentile(ordered, 0.50),
        "p95": _latency_percentile(ordered, 0.95),
        "max": ordered[-1],
    }
    result: dict[str, Any] = {"samples": samples}
    for name, expected_value in expected.items():
        actual = _nonnegative(summary[name], f"{field}.{name}")
        if not math.isclose(
            actual,
            expected_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{field}.{name} is inconsistent with values")
        result[name] = actual
    return result


def _benchmark_versions(value: Any, field: str) -> dict[str, Any]:
    versions = _object(value, field)
    _exact_keys(versions, _BENCHMARK_VERSION_FIELDS, field)
    for name, version in versions.items():
        if version is not None and (
            not isinstance(version, str) or not version
        ):
            raise ValueError(f"{field}.{name} must be a non-empty string or null")
    return _json_copy(versions)


def _benchmark_hardware(value: Any, field: str) -> dict[str, Any]:
    hardware = _object(value, field)
    _exact_keys(
        hardware,
        ("platform", "machine", "logical_cpu_count", "cuda"),
        field,
    )
    for name in ("platform", "machine"):
        if not isinstance(hardware[name], str) or not hardware[name]:
            raise ValueError(f"{field}.{name} must be a non-empty string")
    _integer(
        hardware["logical_cpu_count"],
        f"{field}.logical_cpu_count",
        positive=True,
    )
    cuda = hardware["cuda"]
    if cuda is not None:
        cuda = _object(cuda, f"{field}.cuda")
        _exact_keys(
            cuda,
            (
                "index",
                "name",
                "capability",
                "total_memory_bytes",
                "torch_cuda_version",
            ),
            f"{field}.cuda",
        )
        _integer(cuda["index"], f"{field}.cuda.index")
        if not isinstance(cuda["name"], str) or not cuda["name"]:
            raise ValueError(f"{field}.cuda.name must be non-empty")
        capability = _list(
            cuda["capability"],
            f"{field}.cuda.capability",
        )
        if len(capability) != 2:
            raise ValueError(f"{field}.cuda.capability must have two integers")
        for index, component in enumerate(capability):
            _integer(component, f"{field}.cuda.capability[{index}]")
        _integer(
            cuda["total_memory_bytes"],
            f"{field}.cuda.total_memory_bytes",
            positive=True,
        )
        if (
            not isinstance(cuda["torch_cuda_version"], str)
            or not cuda["torch_cuda_version"]
        ):
            raise ValueError(
                f"{field}.cuda.torch_cuda_version must be non-empty"
            )
    return _json_copy(hardware)


def _benchmark_memory_summary(value: Any, field: str) -> dict[str, Any]:
    summary = _object(value, field)
    _exact_keys(summary, _BENCHMARK_MEMORY_FIELDS, field)
    result: dict[str, Any] = {}
    for name in _BENCHMARK_MEMORY_FIELDS:
        item = summary[name]
        result[name] = (
            None
            if item is None
            else _integer(item, f"{field}.{name}")
        )
    return result


def _parameter_stats(value: Any, field: str) -> dict[str, int]:
    stats = _object(value, field)
    _exact_keys(
        stats,
        ("parameter_count", "parameter_bytes", "trainable_parameter_count"),
        field,
    )
    return {
        name: _integer(stats[name], f"{field}.{name}")
        for name in (
            "parameter_count",
            "parameter_bytes",
            "trainable_parameter_count",
        )
    }


def _benchmark_parameters(
    value: Any,
    *,
    field: str,
    full_stack: bool,
) -> dict[str, Any]:
    parameters = _object(value, field)
    if not full_stack:
        return _parameter_stats(parameters, field)
    _exact_keys(
        parameters,
        (
            "parameter_count",
            "parameter_bytes",
            "trainable_parameter_count",
            "ardy_model",
            "encoder",
            "combined",
        ),
        field,
    )
    top = {
        name: _integer(parameters[name], f"{field}.{name}")
        for name in (
            "parameter_count",
            "parameter_bytes",
            "trainable_parameter_count",
        )
    }
    ardy = _parameter_stats(parameters["ardy_model"], f"{field}.ardy_model")
    encoder = _parameter_stats(parameters["encoder"], f"{field}.encoder")
    combined = _parameter_stats(parameters["combined"], f"{field}.combined")
    expected_combined = {
        name: ardy[name] + encoder[name]
        for name in combined
    }
    if top != combined or combined != expected_combined:
        raise ValueError(f"{field} combined counts are inconsistent")
    return {
        **top,
        "ardy_model": ardy,
        "encoder": encoder,
        "combined": combined,
    }


def _benchmark_output(
    value: Any,
    *,
    field: str,
    expected_dim: int,
) -> dict[str, Any]:
    output = _object(value, field)
    _exact_keys(
        output,
        ("shape", "dtype", "lengths", "warm_shape", "warm_dtype", "warm_lengths"),
        field,
    )
    expected_shape = [1, 1, expected_dim]
    if output["shape"] != expected_shape or output["warm_shape"] != expected_shape:
        raise ValueError(f"{field} shape must be {expected_shape}")
    if (
        not isinstance(output["dtype"], str)
        or output["dtype"] != output["warm_dtype"]
    ):
        raise ValueError(f"{field} dtype is invalid or changed")
    if output["lengths"] != [1] or output["warm_lengths"] != [1]:
        raise ValueError(f"{field} lengths must be [1]")
    return _json_copy(output)


def _portable_benchmark_identity(
    identity: dict[str, Any],
) -> dict[str, Any]:
    def portable_hashes(value: Any, field: str) -> dict[str, str]:
        hashes = _object(value, field)
        if not hashes:
            raise ValueError(f"{field} must not be empty")
        result: dict[str, str] = {}
        for name, digest in hashes.items():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or PureWindowsPath(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise ValueError(
                    f"{field} contains a non-portable relative filename"
                )
            result[name] = _sha256(digest, f"{field}.{name}")
        return result

    result: dict[str, Any] = {}
    if "teacher" in identity:
        result["teacher"] = _json_copy(
            _object(identity["teacher"], "benchmark.identity.teacher")
        )
    if "student_artifact" in identity:
        student = _object(
            identity["student_artifact"],
            "benchmark.identity.student_artifact",
        )
        result["student_artifact"] = _picked(
            student,
            (
                "artifact_fingerprint",
                "format_version",
                "base_model",
                "compatible_ardy_models",
                "output_dim",
                "expected_checkpoint_sha256",
                "checkpoint_sha256_verified",
            ),
        )
        result["student_artifact"]["files_sha256"] = portable_hashes(
            student.get("files_sha256"),
            "benchmark.identity.student_artifact.files_sha256",
        )
    if "ardy" in identity:
        ardy = _object(identity["ardy"], "benchmark.identity.ardy")
        result["ardy"] = _picked(
            ardy,
            (
                "requested_model",
                "resolved_model",
                "checkpoint_sha256",
            ),
        )
        result["ardy"]["files_sha256"] = portable_hashes(
            ardy.get("files_sha256"),
            "benchmark.identity.ardy.files_sha256",
        )
    if "condition_path" in identity:
        condition_path = _object(
            identity["condition_path"],
            "benchmark.identity.condition_path",
        )
        result["condition_path"] = _picked(
            condition_path,
            (
                "encoder_kind",
                "encoder_output_dim",
                "projection_bias_included",
                "output_order",
                "output_dim",
            ),
        )
        for name in ("root_projection", "body_projection"):
            projection = _object(
                condition_path.get(name),
                f"benchmark.identity.condition_path.{name}",
            )
            result["condition_path"][name] = _picked(
                projection,
                (
                    "projected_text_index",
                    "in_features",
                    "out_features",
                ),
            )
    return result


def _benchmark_summary(
    report: dict[str, Any],
    *,
    expected_encoder: str,
    expected_scope: str,
    winner: Candidate,
    lineage: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if report.get("schema_version") != 1:
        raise ValueError(f"{expected_encoder} benchmark schema_version must be 1")
    if report.get("encoder") != expected_encoder:
        raise ValueError(
            f"expected benchmark encoder {expected_encoder!r}, got "
            f"{report.get('encoder')!r}"
        )
    if report.get("benchmark_scope") != expected_scope:
        raise ValueError(
            f"{expected_encoder} benchmark_scope must be {expected_scope!r}"
        )
    configuration = _object(
        report.get("configuration"),
        f"{expected_encoder}.configuration",
    )
    prompt = configuration.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{expected_encoder} benchmark prompt is missing")
    if configuration.get("fresh_process_required") is not True:
        raise ValueError(
            f"{expected_encoder} benchmark must use a fresh process"
        )
    if configuration.get("first_measured_calls") != 1:
        raise ValueError(
            f"{expected_encoder} benchmark must measure one first call"
        )
    if configuration.get("warmup_calls_before_warm_measurement") != 1:
        raise ValueError(
            f"{expected_encoder} benchmark warmup protocol is invalid"
        )
    if configuration.get("external_batch_size") != 1:
        raise ValueError(
            f"{expected_encoder} benchmark external batch size must be one"
        )
    warm_runs = _integer(
        configuration.get("warm_runs"),
        f"{expected_encoder}.configuration.warm_runs",
        positive=True,
    )
    if configuration.get("warm_measured_calls") != warm_runs:
        raise ValueError(
            f"{expected_encoder} benchmark warm call count is inconsistent"
        )
    identity = _object(report.get("identity"), f"{expected_encoder}.identity")
    kind = report.get("encoder_kind")
    if kind == "teacher":
        teacher = _object(
            identity.get("teacher"),
            f"{expected_encoder}.identity.teacher",
        )
        expected_teacher = {
            "foundation_model": lineage["foundation_model_name_or_path"],
            "foundation_model_revision": lineage["foundation_model_revision"],
            "base_model": lineage["base_model_name_or_path"],
            "base_model_revision": lineage["base_model_revision"],
            "peft_model": lineage["peft_model_name_or_path"],
            "peft_model_revision": lineage["peft_model_revision"],
            "llm_dim": lineage["teacher_dim"],
        }
        if teacher != expected_teacher:
            raise ValueError(
                f"{expected_encoder} benchmark teacher identity differs from "
                "the evaluated cache"
            )
    elif kind == "student":
        student = _object(
            identity.get("student_artifact"),
            f"{expected_encoder}.identity.student_artifact",
        )
        if student.get("artifact_fingerprint") != winner.artifact_fingerprint:
            raise ValueError(
                f"{expected_encoder} benchmark does not measure the winner"
            )
    else:
        raise ValueError(f"{expected_encoder} encoder_kind is invalid")

    if expected_scope == "full_condition_stack":
        ardy = _object(
            identity.get("ardy"),
            f"{expected_encoder}.identity.ardy",
        )
        if ardy.get("checkpoint_sha256") != lineage["checkpoint_sha256"]:
            raise ValueError(
                f"{expected_encoder} benchmark uses a different ARDY checkpoint"
            )
        if ardy.get("resolved_model") != winner.report.get("ardy_model"):
            raise ValueError(
                f"{expected_encoder} benchmark uses a different ARDY model"
            )
        ardy_hashes = _object(
            ardy.get("files_sha256"),
            f"{expected_encoder}.identity.ardy.files_sha256",
        )
        if ardy_hashes.get("denoiser.safetensors") != lineage[
            "checkpoint_sha256"
        ]:
            raise ValueError(
                f"{expected_encoder} benchmark ARDY file hashes are "
                "inconsistent with its checkpoint"
            )
        condition_path = _object(
            identity.get("condition_path"),
            f"{expected_encoder}.identity.condition_path",
        )
        if (
            condition_path.get("encoder_kind") != kind
            or condition_path.get("output_dim") != 2 * _CONDITION_DIM
            or condition_path.get("output_order") != ["root", "body"]
            or condition_path.get("projection_bias_included") is not True
        ):
            raise ValueError(
                f"{expected_encoder} benchmark condition path is invalid"
            )
        if kind == "student":
            student = _object(
                identity.get("student_artifact"),
                f"{expected_encoder}.identity.student_artifact",
            )
            if student.get("expected_checkpoint_sha256") != [
                lineage["checkpoint_sha256"]
            ] or student.get("checkpoint_sha256_verified") is not True:
                raise ValueError(
                    f"{expected_encoder} student checkpoint verification "
                    "is missing"
                )

    timing = _object(report.get("timing_seconds"), f"{expected_encoder}.timing")
    memory = _object(report.get("memory"), f"{expected_encoder}.memory")
    full_stack = expected_scope == "full_condition_stack"
    expected_output_dim = (
        2048
        if full_stack or kind == "student"
        else 4096
    )
    public = {
        "encoder": expected_encoder,
        "encoder_kind": kind,
        "benchmark_scope": expected_scope,
        "versions": _benchmark_versions(
            report.get("versions"),
            f"{expected_encoder}.versions",
        ),
        "hardware": _benchmark_hardware(
            report.get("hardware"),
            f"{expected_encoder}.hardware",
        ),
        "configuration": _picked(
            configuration,
            (
                "fresh_process_required",
                "device_requested",
                "device_resolved",
                "model_device",
                "dtype_requested",
                "model_dtype",
                "ardy_model_device",
                "ardy_model_dtype",
                "warm_runs",
                "first_measured_calls",
                "warmup_calls_before_warm_measurement",
                "warm_measured_calls",
                "external_batch_size",
                "production_wrapper",
                "timed_operation",
                "load_timing_scope",
                "cuda_context_timing",
            ),
        ),
        "timing_seconds": {
            "load": _nonnegative(
                timing.get("load"),
                f"{expected_encoder}.load",
            ),
            "first_encode": _latency_summary(
                timing.get("first_encode"),
                f"{expected_encoder}.first_encode",
                expected_samples=1,
            ),
            "warm_encode": _latency_summary(
                timing.get("warm_encode"),
                f"{expected_encoder}.warm_encode",
                expected_samples=warm_runs,
            ),
        },
        "parameters": _benchmark_parameters(
            report.get("parameters"),
            field=f"{expected_encoder}.parameters",
            full_stack=full_stack,
        ),
        "memory_summary": _benchmark_memory_summary(
            memory.get("summary"),
            f"{expected_encoder}.memory.summary",
        ),
        "output": _benchmark_output(
            report.get("output"),
            field=f"{expected_encoder}.output",
            expected_dim=expected_output_dim,
        ),
        "identity": _portable_benchmark_identity(identity),
    }
    return public, prompt


def _ratio(left: Any, right: Any) -> float | None:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
        or not math.isfinite(float(left))
        or not math.isfinite(float(right))
        or float(right) <= 0
    ):
        return None
    return float(left) / float(right)


def _reduction_fraction(reference: Any, candidate: Any) -> float | None:
    ratio = _ratio(candidate, reference)
    return None if ratio is None else 1.0 - ratio


def _benchmark_comparison(
    teacher: dict[str, Any],
    student: dict[str, Any],
) -> dict[str, Any]:
    teacher_timing = teacher["timing_seconds"]
    student_timing = student["timing_seconds"]
    teacher_memory = teacher["memory_summary"]
    student_memory = student["memory_summary"]
    return {
        "load_speedup": _ratio(
            teacher_timing["load"],
            student_timing["load"],
        ),
        "first_encode_p50_speedup": _ratio(
            teacher_timing["first_encode"].get("p50"),
            student_timing["first_encode"].get("p50"),
        ),
        "warm_encode_p50_speedup": _ratio(
            teacher_timing["warm_encode"].get("p50"),
            student_timing["warm_encode"].get("p50"),
        ),
        "warm_encode_mean_speedup": _ratio(
            teacher_timing["warm_encode"].get("mean"),
            student_timing["warm_encode"].get("mean"),
        ),
        "parameter_bytes_reduction_fraction": _reduction_fraction(
            teacher["parameters"].get("parameter_bytes"),
            student["parameters"].get("parameter_bytes"),
        ),
        "rss_peak_reduction_fraction": _reduction_fraction(
            teacher_memory.get("rss_peak_bytes"),
            student_memory.get("rss_peak_bytes"),
        ),
        "cuda_allocated_peak_reduction_fraction": _reduction_fraction(
            teacher_memory.get("cuda_allocated_peak_bytes"),
            student_memory.get("cuda_allocated_peak_bytes"),
        ),
    }


def _benchmark_pair(
    teacher_report: dict[str, Any],
    student_report: dict[str, Any],
    *,
    scope: str,
    teacher_encoder: str,
    student_encoder: str,
    winner: Candidate,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    teacher, teacher_prompt = _benchmark_summary(
        teacher_report,
        expected_encoder=teacher_encoder,
        expected_scope=scope,
        winner=winner,
        lineage=lineage,
    )
    student, student_prompt = _benchmark_summary(
        student_report,
        expected_encoder=student_encoder,
        expected_scope=scope,
        winner=winner,
        lineage=lineage,
    )
    if teacher_prompt != student_prompt:
        raise ValueError(f"{scope} benchmarks do not use the same prompt")
    for field in ("hardware", "versions"):
        if teacher[field] != student[field]:
            raise ValueError(f"{scope} benchmark {field} differ")
    for field in ("device_resolved", "dtype_requested", "warm_runs"):
        if (
            teacher["configuration"].get(field)
            != student["configuration"].get(field)
        ):
            raise ValueError(f"{scope} benchmark configuration.{field} differs")
    return {
        "teacher": teacher,
        "student": student,
        "comparison": _benchmark_comparison(teacher, student),
    }


def build_public_summary(
    *,
    training_50: LoadedReport,
    training_100: LoadedReport,
    validation_fp32_50: LoadedReport,
    validation_fp32_100: LoadedReport,
    condition_test: LoadedReport,
    motion_test: LoadedReport,
    benchmark_encoder_teacher: LoadedReport,
    benchmark_encoder_student: LoadedReport,
    benchmark_full_teacher: LoadedReport,
    benchmark_full_student: LoadedReport,
) -> dict[str, Any]:
    """Validate raw inputs and return the allowlisted public aggregate."""

    candidate_50 = _training_candidate(
        training_50.value,
        label="50e",
        expected_epochs=50,
    )
    candidate_100 = _training_candidate(
        training_100.value,
        label="100e",
        expected_epochs=100,
    )
    _validate_candidate_pair(candidate_50, candidate_100)
    validation_50 = _condition_evaluation(
        validation_fp32_50.value,
        candidate=candidate_50,
        expected_split="val",
    )
    validation_100 = _condition_evaluation(
        validation_fp32_100.value,
        candidate=candidate_100,
        expected_split="val",
        expected_lineage=validation_50.lineage,
    )
    winner = _select_winner(
        candidate_50,
        candidate_100,
        fp32_scores={
            candidate_50.label: validation_50.score,
            candidate_100.label: validation_100.score,
        },
    )

    test_evaluation = _condition_evaluation(
        condition_test.value,
        candidate=winner,
        expected_split="test",
        expected_lineage=validation_50.lineage,
    )
    dataset = _dataset_public(test_evaluation)
    condition = test_evaluation.public
    lineage = test_evaluation.lineage
    motion = _motion_summary(
        motion_test.value,
        winner=winner,
        lineage=lineage,
    )
    encoder_only = _benchmark_pair(
        benchmark_encoder_teacher.value,
        benchmark_encoder_student.value,
        scope="encoder_only",
        teacher_encoder="teacher",
        student_encoder="student",
        winner=winner,
        lineage=lineage,
    )
    full_stack = _benchmark_pair(
        benchmark_full_teacher.value,
        benchmark_full_student.value,
        scope="full_condition_stack",
        teacher_encoder="full-teacher",
        student_encoder="full-student",
        winner=winner,
        lineage=lineage,
    )
    benchmark_prompts = {
        benchmark_encoder_teacher.value["configuration"]["prompt"],
        benchmark_encoder_student.value["configuration"]["prompt"],
        benchmark_full_teacher.value["configuration"]["prompt"],
        benchmark_full_student.value["configuration"]["prompt"],
    }
    if len(benchmark_prompts) != 1:
        raise ValueError("all benchmark reports must use the same prompt")

    loaded_inputs = {
        "training_50e": training_50,
        "training_100e": training_100,
        "validation_fp32_50e": validation_fp32_50,
        "validation_fp32_100e": validation_fp32_100,
        "condition_test": condition_test,
        "motion_test": motion_test,
        "benchmark_encoder_teacher": benchmark_encoder_teacher,
        "benchmark_encoder_student": benchmark_encoder_student,
        "benchmark_full_teacher": benchmark_full_teacher,
        "benchmark_full_student": benchmark_full_student,
    }
    summary = {
        "format": "ardy-minilm-core40-public-summary",
        "format_version": 1,
        "report_kind": "aggregate_only",
        "distribution": {
            "contains_model_weights": False,
            "contains_dataset_records": False,
            "contains_prompt_text": False,
            "contains_dataset_record_ids": False,
            "contains_absolute_paths": False,
            "third_party_notices": "../THIRD_PARTY_MODELS_AND_DATA.md",
        },
        "raw_report_inputs": {
            name: loaded.public_identity()
            for name, loaded in loaded_inputs.items()
        },
        "selection": {
            "split": "val",
            "metric": _FP32_SELECTION_METRIC,
            "rule": (
                "highest complete-split FP32 saved-artifact score; exact ties "
                "select the shorter 50-epoch candidate"
            ),
            "test_metrics_used_for_selection": False,
            "shared_trajectory_prefix_epochs": (
                _SHARED_TRAJECTORY_PREFIX_EPOCHS
            ),
            "shared_trajectory_prefix_verified": True,
            "within_run_checkpoint_selection": {
                "precision": "bfloat16_autocast",
                "split": "val",
                "metric": _SELECTION_METRIC,
                "scope": (
                    "Each training run retained its best epoch using BF16 "
                    "validation. Discarded epoch states cannot be re-ranked "
                    "under FP32 without retraining or saved checkpoints."
                ),
                "candidates": {
                    candidate_50.label: candidate_50.public,
                    candidate_100.label: candidate_100.public,
                },
            },
            "between_saved_artifacts": {
                "precision": "float32",
                "split": "val",
                "metric": _FP32_SELECTION_METRIC,
                "complete_split": True,
                "candidates": {
                    validation_50.candidate.label: validation_50.public,
                    validation_100.candidate.label: validation_100.public,
                },
            },
            "winner": winner.label,
        },
        "dataset_and_teacher": dataset,
        "winner": {
            "candidate": winner.label,
            "artifact_fingerprint": winner.artifact_fingerprint,
            "artifact_payload_size_bytes": winner.public[
                "artifact_payload_size_bytes"
            ],
            "base_model": winner.report["base_model"],
            "ardy_model": winner.report["ardy_model"],
        },
        "test_evaluation": {
            "condition_fidelity": condition,
            "motion_fidelity": motion,
        },
        "performance": {
            "encoder_only": encoder_only,
            "full_condition_stack": full_stack,
            "benchmark_prompt_sha256": hashlib.sha256(
                next(iter(benchmark_prompts)).encode("utf-8")
            ).hexdigest(),
        },
        "limitations": [
            (
                "Motion metrics measure same-seed replacement fidelity, not "
                "ground-truth text-to-motion semantics."
            ),
            (
                "The proprietary Rigplay test split and TMR evaluator are "
                "unavailable, so these are not paper-comparable FID or "
                "R-precision results."
            ),
            (
                "Benchmark timing and memory are specific to the recorded "
                "software, hardware, and fresh-process protocol."
            ),
            (
                "A single deterministic training seed does not establish "
                "statistical significance."
            ),
            (
                "Within each run, the saved epoch was selected under BF16 "
                "autocast. FP32 comparison covers the saved 50e and 100e "
                "artifacts, not every discarded epoch checkpoint."
            ),
        ],
    }
    _validate_public_value(summary)
    return summary


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-50", type=Path, required=True)
    parser.add_argument("--training-100", type=Path, required=True)
    parser.add_argument("--validation-fp32-50", type=Path, required=True)
    parser.add_argument("--validation-fp32-100", type=Path, required=True)
    parser.add_argument("--condition-test", type=Path, required=True)
    parser.add_argument("--motion-test", type=Path, required=True)
    parser.add_argument(
        "--benchmark-encoder-teacher",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--benchmark-encoder-student",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--benchmark-full-teacher",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--benchmark-full-student",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = (
        args.training_50,
        args.training_100,
        args.validation_fp32_50,
        args.validation_fp32_100,
        args.condition_test,
        args.motion_test,
        args.benchmark_encoder_teacher,
        args.benchmark_encoder_student,
        args.benchmark_full_teacher,
        args.benchmark_full_student,
    )
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("every raw report input must be a different file")
    if args.output.resolve() in resolved_inputs:
        raise ValueError("--output must differ from every raw report input")

    summary = build_public_summary(
        training_50=_load_report(args.training_50),
        training_100=_load_report(args.training_100),
        validation_fp32_50=_load_report(args.validation_fp32_50),
        validation_fp32_100=_load_report(args.validation_fp32_100),
        condition_test=_load_report(args.condition_test),
        motion_test=_load_report(args.motion_test),
        benchmark_encoder_teacher=_load_report(
            args.benchmark_encoder_teacher
        ),
        benchmark_encoder_student=_load_report(
            args.benchmark_encoder_student
        ),
        benchmark_full_teacher=_load_report(args.benchmark_full_teacher),
        benchmark_full_student=_load_report(args.benchmark_full_student),
    )
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
