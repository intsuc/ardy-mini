# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for the prompt-free MiniLM public aggregate builder."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ardy.minilm_teacher_cache import (
    TIMELINE_PROMPT_DEDUPLICATION,
    TIMELINE_PROMPT_GROUPING,
    TIMELINE_PROMPT_MAX_CHARACTERS,
    TIMELINE_PROMPT_NORMALIZATION,
    TIMELINE_SPLIT_HASH_NAMESPACE,
    prompt_provenance_sha256,
)
from scripts.minilm.build_public_summary import (
    LoadedReport,
    _select_winner,
    _selected_prompt_sha256,
    _training_candidate,
    _validate_public_value,
    build_public_summary,
    main,
)

_CACHE_FINGERPRINT = "a" * 64
_CHECKPOINT_SHA256 = "b" * 64
_WINNER_FINGERPRINT = "f" * 64
_LOSER_FINGERPRINT = "5" * 64
_MANIFEST_SHA256 = "1" * 64
_ARDY_MODEL = "ARDY-Core-RP-20FPS-Horizon40"
_FOUNDATION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
_FOUNDATION_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
_BASE_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
_BASE_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
_PEFT_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
_PEFT_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"
_STUDENT_BASE = "sentence-transformers/all-MiniLM-L6-v2"


def _corpus_provenance() -> dict:
    return {
        "format": "ardy-minilm-prompt-provenance",
        "format_version": 1,
        "dataset": {
            "repo_id": "nvidia/SEED-Timeline-Annotations",
            "revision": "b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f",
            "filename": "timelines.jsonl",
            "sha256": ("379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8"),
            "size_bytes": 80_373_523,
            "resolved_from": "hugging_face_hub",
            "owner": "NVIDIA",
            "license": "CC BY 4.0",
            "url": ("https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations"),
        },
        "preparation": {
            "sources": ["overview_description", "events.description"],
            "normalization": TIMELINE_PROMPT_NORMALIZATION,
            "deduplication": TIMELINE_PROMPT_DEDUPLICATION,
            "max_prompt_characters": TIMELINE_PROMPT_MAX_CHARACTERS,
            "grouping": TIMELINE_PROMPT_GROUPING,
            "split_hash_namespace": TIMELINE_SPLIT_HASH_NAMESPACE,
            "seed": 20260726,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "sample_size": 0,
        },
        "counts": {
            "timeline_rows": 66,
            "recording_families": 66,
            "recording_components": 66,
            "missing_propagation_references": 0,
            "raw_overview_descriptions": 33,
            "raw_event_descriptions": 33,
            "dropped_prompt_too_long": 0,
            "unique_descriptions": 66,
            "unique_before_sampling": 66,
            "written": 66,
            "groups_written": 66,
            "splits": {"train": 1, "val": 1, "test": 64},
            "split_groups": {"train": 1, "val": 1, "test": 64},
            "sources": {
                "overview_description": 33,
                "events.description": 33,
            },
        },
        "manifest": {
            "filename": "prompts.jsonl",
            "sha256": _MANIFEST_SHA256,
        },
    }


def _teacher_metadata() -> dict:
    corpus_provenance = _corpus_provenance()
    return {
        "format_version": 3,
        "input_path": "/private/prompts.jsonl",
        "input_sha256": _MANIFEST_SHA256,
        "input_metadata_sha256": prompt_provenance_sha256(corpus_provenance),
        "corpus_provenance": corpus_provenance,
        "model_name": f"{_BASE_MODEL} + {_PEFT_MODEL}",
        "foundation_model_name_or_path": _FOUNDATION_MODEL,
        "foundation_model_revision": _FOUNDATION_REVISION,
        "base_model_name_or_path": _BASE_MODEL,
        "base_model_revision": _BASE_REVISION,
        "peft_model_name_or_path": _PEFT_MODEL,
        "peft_model_revision": _PEFT_REVISION,
        "model_revisions": {
            "foundation_model": _FOUNDATION_MODEL,
            "foundation": _FOUNDATION_REVISION,
            "base": _BASE_REVISION,
            "peft": _PEFT_REVISION,
        },
        "versions": {
            "torch": "2.9.1",
            "transformers": "5.1.0",
            "peft": "0.18.1",
            "safetensors": "0.7.0",
        },
        "device": {
            "cli_requested": "cpu",
            "env_override": None,
            "requested": "cpu",
            "resolved": "cpu",
        },
        "provenance_status": "recorded",
        "status": "complete",
        "count": 66,
        "completed_count": 66,
        "split_counts": {"train": 1, "val": 1, "test": 64},
        "shard_size": 66,
        "shards": ["teacher-00000.pt"],
        "shard_sha256": {"teacher-00000.pt": "c" * 64},
        "teacher_dim": 4096,
        "target_dim": 2048,
        "target_keys": [
            "denoiser.backbone.root_model.embed_text.weight",
            "denoiser.backbone.body_model.embed_text.weight",
        ],
        "target_order": ["root", "body"],
        "bias_applied": False,
        "dtype": {
            "teacher_model": "bfloat16",
            "teacher_embeddings": "float32",
            "projection_weights": "float32",
            "targets": "float32",
        },
        "teacher_batch_size": 1,
        "checkpoint_path": "/private/denoiser.safetensors",
        "checkpoint_sha256": _CHECKPOINT_SHA256,
    }


def _training_report(epochs: int, score: float, fingerprint: str) -> dict:
    validation = {
        "root_cosine": score + 0.01,
        "body_cosine": score - 0.01,
        "root_rmse": 1.0,
        "body_rmse": 1.1,
        "root_mae": 0.7,
        "body_mae": 0.8,
        "root_nrmse": 0.2,
        "body_nrmse": 0.22,
    }
    lower_validation = {
        "root_cosine": 0.71,
        "body_cosine": 0.69,
        "root_rmse": 2.0,
        "body_rmse": 2.1,
        "root_mae": 1.7,
        "body_mae": 1.8,
        "root_nrmse": 0.4,
        "body_nrmse": 0.42,
    }
    prefix_validation = {
        "root_cosine": 0.81,
        "body_cosine": 0.79,
        "root_rmse": 1.0,
        "body_rmse": 1.1,
        "root_mae": 0.7,
        "body_mae": 0.8,
        "root_nrmse": 0.2,
        "body_nrmse": 0.22,
    }
    history = [
        {
            "epoch": epoch,
            "seconds": 1.0,
            "train": {"loss": 1.0},
            "validation": (validation if epoch == epochs else prefix_validation if epoch == 50 else lower_validation),
        }
        for epoch in range(1, epochs + 1)
    ]
    return {
        "schema_version": 1,
        "artifact": f"candidate-{epochs}",
        "artifact_fingerprint": fingerprint,
        "artifact_payload_size_bytes": 1000 + epochs,
        "base_model": _STUDENT_BASE,
        "ardy_model": _ARDY_MODEL,
        "configuration": {
            "seed": 20260726,
            "epochs": epochs,
            "optimizer_updates": epochs * 10,
            "lr_schedule_epochs": 100,
            "runtime_max_length": 128,
            "bf16_autocast": True,
            "base_model": _STUDENT_BASE,
            "ardy_model": _ARDY_MODEL,
        },
        "best_validation_score": score,
        "best_epoch": epochs,
        "teacher_cache_fingerprint": _CACHE_FINGERPRINT,
        "runtime_versions": {"torch": "test"},
        "selection": {
            "split": "val",
            "metric": "mean(root_cosine, body_cosine)",
            "test_evaluated": False,
        },
        "validation": validation,
        "elapsed_seconds": float(epochs),
        "history": history,
    }


def _condition_report(
    *,
    fingerprint: str = _WINNER_FINGERPRINT,
    split: str = "test",
    cosine: float = 0.9,
    dtype: str = "float32",
) -> dict:
    examples = _teacher_metadata()["split_counts"][split]

    def metric(width: int) -> dict:
        return {
            "examples": examples,
            "elements": examples * width,
            "cosine": {"mean": cosine, "p5": cosine - 0.1},
            "rmse": 1.0,
            "nrmse": 0.2,
            "mae": 0.7,
            "target_rms": 5.0,
        }

    return {
        "schema_version": 1,
        "command": ["/private/python", "evaluate_conditions.py"],
        "configuration": {
            "teacher_cache": "/private/teacher",
            "student_path": "/private/student",
            "split": split,
            "teacher_dim": 4096,
            "dtype": dtype,
            "batch_size": 64,
            "max_samples": None,
            "max_length": None,
            "expected_ardy_model": _ARDY_MODEL,
            "nrmse_definition": "RMSE divided by target RMS",
        },
        "dataset": {
            "metadata_path": "/private/teacher/metadata.json",
            "metadata": _teacher_metadata(),
            "shards": ["/private/teacher/teacher-00000.pt"],
            "cache_fingerprint": _CACHE_FINGERPRINT,
            "examples_seen_in_loaded_shards": 66,
            "evaluated_examples": examples,
        },
        "student_artifact": {
            "artifact_fingerprint": fingerprint,
            "base_model": _STUDENT_BASE,
            "condition_dim": 1024,
            "output_dim": 2048,
            "max_length": 128,
        },
        "metrics": {
            "root": metric(1024),
            "body": metric(1024),
            "overall": metric(2048),
        },
    }


def _motion_report(fingerprint: str = _WINNER_FINGERPRINT) -> dict:
    metrics = {
        "root_ade_m": 0.1,
        "root_fde_m": 0.2,
        "global_mpjpe_m": 0.1,
        "root_aligned_mpjpe_m": 0.05,
        "joint_velocity_error_m_per_s": 0.2,
        "motion_cosine": 0.99,
        "heading_mae_deg": 1.0,
        "foot_contact_agreement": 0.9,
        "foot_contact_macro_f1": 0.8,
        "foot_contact_macro_iou": 0.7,
    }
    texts = [f"private Timeline prompt {index}" for index in range(64)]
    seeds = [0, 1, 2]
    seed_pairs = [(0, 1), (0, 2), (1, 2)]
    return {
        "schema_version": 1,
        "scope": {
            "split": "test",
            "checkpoint_dir": "/private/checkpoint",
            "student": "/private/student",
            "teacher_cache": "/private/teacher",
            "prompt_manifest": "/private/prompts.jsonl",
            "requested_model": "core",
            "resolved_model": _ARDY_MODEL,
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "student_artifact_fingerprint": fingerprint,
            "teacher_cache_fingerprint": _CACHE_FINGERPRINT,
            "prompt_manifest_sha256": _MANIFEST_SHA256,
            "source_filter": ["overview_description", "events.description"],
            "prompt_selection_algorithm": (
                "evenly_spaced_numpy_linspace_v1"
            ),
            "selected_prompt_sha256": _selected_prompt_sha256(texts),
            "eligible_test_prompts": 64,
            "held_out_test_prompts_available": 64,
            "prompts": 64,
            "seeds": seeds,
            "teacher_diversity_seed_pairs": 3,
            "duration_seconds": 4.0,
            "fps": 20.0,
            "num_frames": 80,
            "diffusion_steps": 10,
            "cfg_weight": [2.0, 2.0],
            "student_dtype": "float32",
            "determinism": {
                "torch_deterministic_algorithms": True,
                "torch_deterministic_warn_only": False,
                "cublas_workspace_config": ":4096:8",
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "repeatability_check": {
                "enabled": True,
                "exact_equal": True,
                "prompt_index": 0,
                "seed": 0,
                "compared_keys": [
                    "root_positions",
                    "posed_joints",
                    "global_root_heading",
                    "foot_contacts",
                ],
            },
            "postprocess_applied": False,
            "motion_output_stage": "raw inverse output",
            "metric_note": "paired fidelity",
        },
        "paired_teacher_student": copy.deepcopy(metrics),
        "teacher_seed_diversity": copy.deepcopy(metrics),
        "normalized_by_teacher_diversity": {
            f"{name}_vs_teacher_diversity": 1.0
            for name in (
                "root_ade_m",
                "root_fde_m",
                "global_mpjpe_m",
                "root_aligned_mpjpe_m",
                "joint_velocity_error_m_per_s",
            )
        },
        "cases": [
            {
                "prompt_index": prompt_index,
                "text": texts[prompt_index],
                "seed": seed,
                "teacher_generation_seconds": 1.0,
                "student_generation_seconds": 1.0,
                **metrics,
            }
            for prompt_index in range(64)
            for seed in seeds
        ],
        "teacher_diversity_cases": [
            {
                "prompt_index": prompt_index,
                "text": texts[prompt_index],
                "seed_a": seed_a,
                "seed_b": seed_b,
                **metrics,
            }
            for seed_a, seed_b in seed_pairs
            for prompt_index in range(64)
        ],
    }


def _teacher_identity() -> dict:
    return {
        "foundation_model": _FOUNDATION_MODEL,
        "foundation_model_revision": _FOUNDATION_REVISION,
        "base_model": _BASE_MODEL,
        "base_model_revision": _BASE_REVISION,
        "peft_model": _PEFT_MODEL,
        "peft_model_revision": _PEFT_REVISION,
        "llm_dim": 4096,
    }


def _benchmark_report(encoder: str) -> dict:
    full = encoder.startswith("full-")
    kind = encoder.removeprefix("full-")
    identity = (
        {"teacher": _teacher_identity()}
        if kind == "teacher"
        else {
            "student_artifact": {
                "path": "/private/student",
                "files_sha256": {"backbone/model.safetensors": "d" * 64},
                "artifact_fingerprint": _WINNER_FINGERPRINT,
                "format_version": 2,
                "base_model": _STUDENT_BASE,
                "compatible_ardy_models": [_ARDY_MODEL],
                "output_dim": 2048,
            }
        }
    )
    if full:
        if kind == "student":
            identity["student_artifact"].update(
                {
                    "expected_checkpoint_sha256": [
                        _CHECKPOINT_SHA256
                    ],
                    "checkpoint_sha256_verified": True,
                }
            )
        identity.update(
            {
                "ardy": {
                    "requested_model": "core",
                    "resolved_model": _ARDY_MODEL,
                    "model_dir": "/private/checkpoint",
                    "files_sha256": {"denoiser.safetensors": _CHECKPOINT_SHA256},
                    "checkpoint_sha256": _CHECKPOINT_SHA256,
                },
                "condition_path": {
                    "encoder_kind": kind,
                    "encoder_output_dim": (
                        4096 if kind == "teacher" else 2048
                    ),
                    "root_projection": {
                        "class": "private.RootProjection",
                        "projected_text_index": 0,
                        "in_features": (
                            4096 if kind == "teacher" else 2048
                        ),
                        "out_features": 1024,
                    },
                    "body_projection": {
                        "class": "private.BodyProjection",
                        "projected_text_index": 1,
                        "in_features": (
                            4096 if kind == "teacher" else 2048
                        ),
                        "out_features": 1024,
                    },
                    "projection_bias_included": True,
                    "output_order": ["root", "body"],
                    "output_dim": 2048,
                },
            }
        )
    latency_value = 2.0 if kind == "teacher" else 1.0

    def latency(samples: int) -> dict:
        values = [latency_value] * samples
        return {
            "samples": samples,
            "mean": latency_value,
            "min": latency_value,
            "p50": latency_value,
            "p95": latency_value,
            "max": latency_value,
            "values": values,
        }

    encoder_parameters = {
        "parameter_count": 20 if kind == "teacher" else 10,
        "parameter_bytes": 200 if kind == "teacher" else 100,
        "trainable_parameter_count": 0,
    }
    if full:
        ardy_parameters = {
            "parameter_count": 30,
            "parameter_bytes": 300,
            "trainable_parameter_count": 0,
        }
        combined_parameters = {
            name: ardy_parameters[name] + encoder_parameters[name]
            for name in encoder_parameters
        }
        parameters = {
            **combined_parameters,
            "ardy_model": ardy_parameters,
            "encoder": encoder_parameters,
            "combined": combined_parameters,
        }
    else:
        parameters = encoder_parameters
    memory_value = 200 if kind == "teacher" else 100
    memory_summary = {
        "rss_current_bytes": memory_value,
        "rss_peak_bytes": memory_value,
        "mem_available_before_bytes": memory_value,
        "mem_available_after_bytes": memory_value,
        "mem_available_min_bytes": memory_value,
        "mem_available_drop_bytes": 0,
        "cuda_allocated_current_bytes": memory_value,
        "cuda_allocated_peak_bytes": memory_value,
        "cuda_reserved_current_bytes": memory_value,
        "cuda_reserved_peak_bytes": memory_value,
    }
    output_dim = 2048 if full or kind == "student" else 4096
    return {
        "schema_version": 1,
        "encoder": encoder,
        "encoder_kind": kind,
        "benchmark_scope": "full_condition_stack" if full else "encoder_only",
        "command": ["/private/python", "benchmark.py"],
        "versions": {
            "python": "3.11",
            "torch": "test",
            "ardy": "test",
            "transformers": "test",
            "peft": "test",
            "safetensors": "test",
        },
        "hardware": {
            "platform": "test",
            "machine": "test",
            "logical_cpu_count": 1,
            "cuda": {
                "index": 0,
                "name": "test GPU",
                "capability": [8, 0],
                "total_memory_bytes": 1000,
                "torch_cuda_version": "test",
            },
        },
        "configuration": {
            "prompt": "private benchmark prompt",
            "student_path": "/private/student",
            "fresh_process_required": True,
            "device_requested": "cuda",
            "device_resolved": "cuda",
            "model_device": "cuda",
            "dtype_requested": "bfloat16",
            "model_dtype": "bfloat16",
            "warm_runs": 2,
            "first_measured_calls": 1,
            "warmup_calls_before_warm_measurement": 1,
            "warm_measured_calls": 2,
            "external_batch_size": 1,
            "production_wrapper": "test.Wrapper",
            "timed_operation": "prompt to conditions",
            "load_timing_scope": "local load",
        },
        "timing_seconds": {
            "load": 4.0 if kind == "teacher" else 1.0,
            "first_encode": latency(1),
            "warm_encode": latency(2),
        },
        "parameters": parameters,
        "memory": {"summary": memory_summary},
        "output": {
            "shape": [1, 1, output_dim],
            "dtype": "bfloat16",
            "lengths": [1],
            "warm_shape": [1, 1, output_dim],
            "warm_dtype": "bfloat16",
            "warm_lengths": [1],
        },
        "identity": identity,
    }


def _loaded(name: str, value: dict) -> LoadedReport:
    encoded = json.dumps(value, sort_keys=True).encode()
    return LoadedReport(
        path=Path(f"/private/{name}.json"),
        value=value,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _inputs(
    *,
    condition_fingerprint: str = _WINNER_FINGERPRINT,
) -> dict[str, LoadedReport]:
    return {
        "training_50": _loaded(
            "training-50",
            _training_report(50, 0.8, _LOSER_FINGERPRINT),
        ),
        "training_100": _loaded(
            "training-100",
            _training_report(100, 0.9, _WINNER_FINGERPRINT),
        ),
        "validation_fp32_50": _loaded(
            "validation-fp32-50",
            _condition_report(
                fingerprint=_LOSER_FINGERPRINT,
                split="val",
                cosine=0.8,
            ),
        ),
        "validation_fp32_100": _loaded(
            "validation-fp32-100",
            _condition_report(
                fingerprint=_WINNER_FINGERPRINT,
                split="val",
                cosine=0.9,
            ),
        ),
        "condition_test": _loaded(
            "condition",
            _condition_report(fingerprint=condition_fingerprint),
        ),
        "motion_test": _loaded("motion", _motion_report()),
        "benchmark_encoder_teacher": _loaded(
            "encoder-teacher",
            _benchmark_report("teacher"),
        ),
        "benchmark_encoder_student": _loaded(
            "encoder-student",
            _benchmark_report("student"),
        ),
        "benchmark_full_teacher": _loaded(
            "full-teacher",
            _benchmark_report("full-teacher"),
        ),
        "benchmark_full_student": _loaded(
            "full-student",
            _benchmark_report("full-student"),
        ),
    }


def test_public_builder_selects_on_validation_and_drops_private_raw_fields():
    inputs = _inputs()
    summary = build_public_summary(**inputs)

    assert summary["selection"]["winner"] == "100e"
    assert summary["selection"]["test_metrics_used_for_selection"] is False
    assert summary["selection"]["shared_trajectory_prefix_epochs"] == 50
    assert summary["selection"]["shared_trajectory_prefix_verified"] is True
    assert summary["winner"]["artifact_fingerprint"] == _WINNER_FINGERPRINT
    assert summary["test_evaluation"]["condition_fidelity"]["metrics"]["overall"]["cosine"]["mean"] == 0.9
    assert summary["performance"]["encoder_only"]["comparison"]["warm_encode_p50_speedup"] == 2.0
    encoded = json.dumps(summary, sort_keys=True)
    assert "private Timeline prompt" not in encoded
    assert "private benchmark prompt" not in encoded
    assert "/private/" not in encoded
    assert "private-record" not in encoded

    changed_inputs = dict(inputs)
    changed_condition = copy.deepcopy(inputs["condition_test"].value)
    changed_condition["metrics"]["overall"]["cosine"]["mean"] = -0.5
    changed_inputs["condition_test"] = _loaded(
        "changed-condition",
        changed_condition,
    )
    changed = build_public_summary(**changed_inputs)
    assert changed["selection"]["winner"] == "100e"
    assert (
        changed["selection"]["between_saved_artifacts"]
        == summary["selection"]["between_saved_artifacts"]
    )


def test_winner_selection_accepts_only_validation_training_candidates():
    candidate_50 = _training_candidate(
        _training_report(50, 0.8, _LOSER_FINGERPRINT),
        label="50e",
        expected_epochs=50,
    )
    candidate_100 = _training_candidate(
        _training_report(100, 0.9, _WINNER_FINGERPRINT),
        label="100e",
        expected_epochs=100,
    )

    winner = _select_winner(
        candidate_50,
        candidate_100,
        fp32_scores={"50e": 0.8, "100e": 0.9},
    )

    assert winner.label == "100e"
    assert winner.score == 0.9


def test_saved_artifact_selection_uses_fp32_not_bf16_training_scores():
    inputs = _inputs()
    inputs["validation_fp32_50"] = _loaded(
        "validation-fp32-50",
        _condition_report(
            fingerprint=_LOSER_FINGERPRINT,
            split="val",
            cosine=0.95,
        ),
    )
    inputs["validation_fp32_100"] = _loaded(
        "validation-fp32-100",
        _condition_report(
            fingerprint=_WINNER_FINGERPRINT,
            split="val",
            cosine=0.90,
        ),
    )
    inputs["condition_test"] = _loaded(
        "condition-50",
        _condition_report(fingerprint=_LOSER_FINGERPRINT),
    )
    inputs["motion_test"] = _loaded(
        "motion-50",
        _motion_report(fingerprint=_LOSER_FINGERPRINT),
    )
    for field in ("benchmark_encoder_student", "benchmark_full_student"):
        changed = copy.deepcopy(inputs[field].value)
        changed["identity"]["student_artifact"][
            "artifact_fingerprint"
        ] = _LOSER_FINGERPRINT
        inputs[field] = _loaded(field, changed)

    summary = build_public_summary(**inputs)

    assert summary["selection"]["winner"] == "50e"
    assert summary["winner"]["artifact_fingerprint"] == _LOSER_FINGERPRINT


def test_public_builder_ignores_only_seconds_in_shared_training_prefix():
    inputs = _inputs()
    changed = copy.deepcopy(inputs["training_100"].value)
    for index, row in enumerate(changed["history"][:50]):
        row["seconds"] = float(index + 100)
    inputs["training_100"] = _loaded("changed-training-100", changed)

    summary = build_public_summary(**inputs)

    assert summary["selection"]["shared_trajectory_prefix_verified"] is True


def test_public_builder_rejects_non_shared_training_prefix():
    inputs = _inputs()
    changed = copy.deepcopy(inputs["training_100"].value)
    changed["history"][12]["train"]["loss"] = 0.5
    inputs["training_100"] = _loaded("changed-training-100", changed)

    with pytest.raises(ValueError, match="first 50 epochs"):
        build_public_summary(**inputs)


def test_public_builder_rejects_test_report_for_nonwinner():
    with pytest.raises(ValueError, match="expected candidate"):
        build_public_summary(**_inputs(condition_fingerprint=_LOSER_FINGERPRINT))


def test_public_builder_rejects_non_fp32_or_partial_validation():
    inputs = _inputs()
    changed = copy.deepcopy(inputs["validation_fp32_50"].value)
    changed["configuration"]["dtype"] = "bfloat16"
    inputs["validation_fp32_50"] = _loaded("bf16-validation", changed)
    with pytest.raises(ValueError, match="dtype must be 'float32'"):
        build_public_summary(**inputs)

    inputs = _inputs()
    changed = copy.deepcopy(inputs["validation_fp32_100"].value)
    changed["configuration"]["max_samples"] = 1
    inputs["validation_fp32_100"] = _loaded("partial-validation", changed)
    with pytest.raises(ValueError, match="complete val split"):
        build_public_summary(**inputs)


def test_public_builder_rejects_validation_cache_mismatch():
    inputs = _inputs()
    changed = copy.deepcopy(inputs["validation_fp32_100"].value)
    changed["dataset"]["cache_fingerprint"] = "9" * 64
    inputs["validation_fp32_100"] = _loaded("wrong-cache", changed)

    with pytest.raises(ValueError, match="different teacher cache"):
        build_public_summary(**inputs)


def test_public_guard_rejects_prompt_text_paths_and_record_ids():
    for value, message in (
        ({"prompt": "secret"}, "prompt text"),
        ({"artifact": "/private/model"}, "absolute local path"),
        ({"record_id": "secret"}, "record IDs"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_public_value(value)


def test_cli_writes_only_the_public_schema(tmp_path: Path, capsys):
    inputs = _inputs()
    arguments: list[str] = []
    option_names = {
        "training_50": "--training-50",
        "training_100": "--training-100",
        "validation_fp32_50": "--validation-fp32-50",
        "validation_fp32_100": "--validation-fp32-100",
        "condition_test": "--condition-test",
        "motion_test": "--motion-test",
        "benchmark_encoder_teacher": "--benchmark-encoder-teacher",
        "benchmark_encoder_student": "--benchmark-encoder-student",
        "benchmark_full_teacher": "--benchmark-full-teacher",
        "benchmark_full_student": "--benchmark-full-student",
    }
    for field, option in option_names.items():
        path = tmp_path / f"{field}.json"
        path.write_text(
            json.dumps(inputs[field].value),
            encoding="utf-8",
        )
        arguments.extend((option, str(path)))
    output = tmp_path / "summary.json"
    arguments.extend(("--output", str(output)))

    assert main(arguments) == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["format"] == "ardy-minilm-core40-public-summary"
    assert summary["selection"]["winner"] == "100e"
    assert set(summary["raw_report_inputs"]) == {
        "training_50e",
        "training_100e",
        "validation_fp32_50e",
        "validation_fp32_100e",
        "condition_test",
        "motion_test",
        "benchmark_encoder_teacher",
        "benchmark_encoder_student",
        "benchmark_full_teacher",
        "benchmark_full_student",
    }
    capsys.readouterr()
