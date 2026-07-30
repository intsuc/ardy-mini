# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for continuous browser-style FP32/mixed-FP16 evaluation."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ardy.browser.precision import MIXED_FP16_POLICIES, MIXED_FP16_POLICY_VERSION
from scripts.evaluate_browser_fp16 import (
    ModelFilesRuntime,
    PortableRandom,
    PreparedHistory,
    RolloutState,
    _build_public_report,
    _canonical_model_path,
    _capture_file_identity,
    _make_window,
    _prepare_history,
    _prepare_model_files,
    _recenter_and_requantize,
    _validate_common_manifest,
    _validate_compatible_models,
    _validate_json_finite,
    _validate_public_report,
    _verify_file_identity,
    _worst_cases,
    _write_json_report,
    _write_reports,
)


def _dimensions() -> dict[str, int]:
    return {
        "fps": 20,
        "num_frames_per_token": 1,
        "max_tokens": 4,
        "max_frames": 4,
        "generation_tokens": 2,
        "generation_frames": 2,
        "history_tokens": 2,
        "history_frames": 2,
        "root_features_per_frame": 5,
        "nframe_root_dim": 5,
        "latent_dim": 2,
        "hybrid_dim": 7,
        "motion_dim": 6,
        "body_dim": 1,
        "text_condition_dim": 8,
        "num_joints": 1,
    }


def _motion_manifest() -> dict:
    return {
        "dimensions": _dimensions(),
        "recenter": {
            "root_mean": [0, 0, 0, 0, 0],
            "root_std": [1, 1, 1, 1, 1],
            "position_indices": [0, 1, 2],
            "heading_indices": [3, 4],
        },
        "latent_quantization": {
            "levels": [4, 4],
            "mean": [0, 0],
            "std": [1, 1],
        },
    }


def test_portable_random_matches_browser_uint32_stream_and_float32_noise():
    random = PortableRandom(1)
    assert [random.next_uint32() for _ in range(5)] == [
        2693262067,
        11749833,
        2265367787,
        4213581821,
        4159151403,
    ]

    left = PortableRandom(42).normal_array((2, 8))
    right = PortableRandom(42).normal_array((2, 8))
    assert left.dtype == np.float32
    np.testing.assert_array_equal(left, right)
    assert np.isfinite(left).all()


def test_continuation_history_and_window_match_browser_offsets_and_world_transform():
    manifest = _motion_manifest()
    global_hybrid = np.asarray(
        [
            [
                [0, 0.5, 1, 1, 0, 0.25, 0.75],
                [1, 0.5, 2, 1, 0, -0.25, -0.75],
                [2, 0.5, 3, 0, 1, 0.25, 0.75],
                [3, 0.5, 4, 1, 0, -0.25, -0.75],
            ]
        ],
        dtype=np.float32,
    )
    state = RolloutState(
        global_hybrid=global_hybrid,
        initial_translation=np.asarray([10, 0.5, 20], dtype=np.float32),
        initial_heading=0.35,
    )

    prepared = _prepare_history(state, manifest)

    assert prepared.history_tokens == 2
    assert prepared.history_frames == 2
    np.testing.assert_array_equal(prepared.global_translation, np.asarray([3, 0.5, 4], dtype=np.float32))
    assert prepared.first_heading_angle == pytest.approx(math.pi / 2)
    assert prepared.history is not None
    np.testing.assert_array_equal(prepared.history[0, :, :3], np.asarray([[-1, 0, -1], [0, 0, 0]]))

    noise = np.ones((1, 2, 7), dtype=np.float32)
    sample, masks = _make_window(manifest["dimensions"], prepared, noise)
    np.testing.assert_array_equal(sample[:, :2], prepared.history)
    np.testing.assert_array_equal(sample[:, 2:], noise)
    np.testing.assert_array_equal(masks["history_token_mask"], [[1, 1, 0, 0]])
    np.testing.assert_array_equal(masks["generation_token_mask"], [[0, 0, 1, 1]])


def test_recenter_requantize_accumulates_translation_and_uses_tail_heading():
    manifest = _motion_manifest()
    sample = np.asarray(
        [
            [
                [0, 0, 1, 1, 0, 0.25, 0.75],
                [1, 0, 2, 1, 0, -0.25, -0.75],
                [2, 0, 3, 0, 1, 0.25, 0.75],
                [3, 0, 4, 1, 0, -0.25, -0.75],
            ]
        ],
        dtype=np.float32,
    )
    prepared = PreparedHistory(
        history=None,
        history_tokens=2,
        history_frames=2,
        global_translation=np.asarray([10, 0, 20], dtype=np.float32),
        first_heading_angle=0.25,
    )

    recentered, translation, next_heading = _recenter_and_requantize(
        sample,
        manifest,
        prepared,
    )

    np.testing.assert_array_equal(translation, np.asarray([13, 0, 24], dtype=np.float32))
    assert next_heading == pytest.approx(math.pi / 2)
    assert recentered[0, 0, 0] == -3
    assert recentered[0, 0, 2] == -3
    assert recentered[0, 3, 0] == 0
    assert recentered[0, 3, 2] == 0
    assert recentered[0, 2, 5] == 0
    assert recentered[0, 2, 6] == 1


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_REFERENCE_GRAPHS = {
    "text_encoder": b"fp32 text graph payload",
    "denoiser": b"byte-identical fp32 denoiser",
    "decoder": b"fp32 decoder graph payload",
}
_CANDIDATE_GRAPHS = {
    "text_encoder": _REFERENCE_GRAPHS["text_encoder"],
    "denoiser": _REFERENCE_GRAPHS["denoiser"],
    "decoder": b"mixed decoder",
}


def _model_runtime(directory: Path, *, candidate: bool) -> ModelFilesRuntime:
    tokenizer_bytes = b'{"version":"test"}'
    config_bytes = b'{"model_max_length":128}'
    tokenizer_dir = directory / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (tokenizer_dir / "tokenizer_config.json").write_bytes(config_bytes)
    graph_bytes = _CANDIDATE_GRAPHS if candidate else _REFERENCE_GRAPHS
    graph_paths = {
        "text_encoder": "text.onnx",
        "denoiser": "denoiser.onnx",
        "decoder": "decoder.onnx",
    }
    for graph_name, relative_path in graph_paths.items():
        (directory / relative_path).write_bytes(graph_bytes[graph_name])
    manifest = {
        "format": "ardy-browser-model-files",
        "schema_version": 1,
        "model": {"id": "test", "revision": "1" * 64, "variant": "test"},
        "files": {
            "tokenizer/tokenizer.json": {
                "sha256": _digest(tokenizer_bytes),
                "size_bytes": len(tokenizer_bytes),
            },
            "tokenizer/tokenizer_config.json": {
                "sha256": _digest(config_bytes),
                "size_bytes": len(config_bytes),
            },
        },
        "tokenizer": {"directory": "tokenizer", "max_length": 128},
        "graphs": {graph_name: {"model": relative_path} for graph_name, relative_path in graph_paths.items()},
        "dimensions": _dimensions(),
        "generation": {"max_frames": 4},
        "diffusion": {"timesteps": [1, 0]},
        "recenter": _motion_manifest()["recenter"],
        "latent_quantization": _motion_manifest()["latent_quantization"],
        "runtime": {
            "contract_revision": 3,
            "text_only": True,
            "required_webgpu_features": ["shader-f16"],
        },
    }
    if candidate:
        manifest["precision"] = {
            "format": "mixed-fp16",
            "policy_version": MIXED_FP16_POLICY_VERSION,
            "graphs": {
                graph_name: {
                    "policy_id": policy.policy_id,
                    "conversion_mode": policy.conversion_mode,
                    "source_sha256": _digest(_REFERENCE_GRAPHS[graph_name]),
                    "output_sha256": _digest(_CANDIDATE_GRAPHS[graph_name]),
                    "source_size_bytes": len(_REFERENCE_GRAPHS[graph_name]),
                    "output_size_bytes": len(_CANDIDATE_GRAPHS[graph_name]),
                    "size_reduction_bytes": (len(_REFERENCE_GRAPHS[graph_name]) - len(_CANDIDATE_GRAPHS[graph_name])),
                    "output_initializers": {
                        "count_by_dtype": {"float": 1},
                    },
                }
                for graph_name, policy in MIXED_FP16_POLICIES.items()
            },
        }
    else:
        manifest["precision"] = {"format": "fp32"}
    return ModelFilesRuntime(
        directory=directory,
        manifest=manifest,
        text_encoder=None,
        denoiser=None,
        decoder=None,
    )


def test_model_compatibility_allows_only_precision_metadata(tmp_path: Path):
    reference = _model_runtime(tmp_path / "reference", candidate=False)
    candidate = _model_runtime(tmp_path / "candidate", candidate=True)

    validation = _validate_compatible_models(reference, candidate)
    assert validation["candidate_policy_version"] == MIXED_FP16_POLICY_VERSION
    assert validation["identity_graphs_byte_identical"] == [
        "denoiser",
        "text_encoder",
    ]

    incompatible = copy.deepcopy(candidate)
    incompatible.manifest["recenter"]["root_mean"][0] = 1
    with pytest.raises(ValueError, match=r"non-precision contracts differ.*recenter\.root_mean"):
        _validate_compatible_models(reference, incompatible)

    incompatible = copy.deepcopy(candidate)
    incompatible.manifest["runtime"]["required_webgpu_features"] = []
    with pytest.raises(
        ValueError,
        match=r"non-precision contracts differ.*runtime\.required_webgpu_features",
    ):
        _validate_compatible_models(reference, incompatible)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["precision"].__setitem__("policy_version", 2),
            r"precision\.policy_version=3",
        ),
        (
            lambda manifest: manifest["precision"]["graphs"]["denoiser"].__setitem__(
                "policy_id",
                "old-policy",
            ),
            r"denoiser policy_id",
        ),
        (
            lambda manifest: manifest["precision"]["graphs"]["denoiser"].__setitem__(
                "conversion_mode",
                "mixed-fp16",
            ),
            r"denoiser conversion_mode",
        ),
        (
            lambda manifest: manifest["precision"]["graphs"]["denoiser"]["output_initializers"][
                "count_by_dtype"
            ].__setitem__("float16", 1),
            r"must not contain reduced-precision initializers",
        ),
    ],
)
def test_model_compatibility_rejects_nonproduction_precision_contract(
    tmp_path: Path,
    mutation,
    message: str,
):
    reference = _model_runtime(tmp_path / "reference", candidate=False)
    candidate = _model_runtime(tmp_path / "candidate", candidate=True)
    mutation(candidate.manifest)

    with pytest.raises(ValueError, match=message):
        _validate_compatible_models(reference, candidate)


@pytest.mark.parametrize("graph_name", ["text_encoder", "denoiser"])
def test_model_compatibility_rejects_nonidentical_identity_graph(
    tmp_path: Path,
    graph_name: str,
):
    reference = _model_runtime(tmp_path / "reference", candidate=False)
    candidate = _model_runtime(tmp_path / "candidate", candidate=True)
    tampered = f"quantized {graph_name}".encode()
    graph_path = candidate.manifest["graphs"][graph_name]["model"]
    (candidate.directory / graph_path).write_bytes(tampered)
    candidate.manifest["precision"]["graphs"][graph_name]["output_sha256"] = _digest(tampered)
    candidate.manifest["precision"]["graphs"][graph_name]["output_size_bytes"] = len(tampered)

    with pytest.raises(ValueError, match=rf"{graph_name} must be byte-identical"):
        _validate_compatible_models(reference, candidate)


def _metric_output(offset: float, contacts: bool) -> dict[str, np.ndarray]:
    frames = 2
    identity = np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 1, 3, 3)).copy()
    root = np.zeros((1, frames, 3), dtype=np.float32)
    root[..., 0] = offset
    return {
        "normalized_motion": np.full((1, frames, 2), 1 + offset, dtype=np.float32),
        "posed_joints": root[:, :, None].copy(),
        "local_rotations": identity,
        "global_rotations": identity,
        "root_positions": root,
        "foot_contacts": np.full((1, frames, 4), contacts),
        "global_root_heading": np.broadcast_to(
            np.asarray([1, 0], dtype=np.float32),
            (1, frames, 2),
        ).copy(),
    }


def test_worst_case_report_attributes_prompt_seed_and_window():
    reference = _metric_output(0, True)
    cases = [
        (
            {"prompt": "small", "seed": 1, "window_index": 0},
            reference,
            _metric_output(0.1, True),
        ),
        (
            {"prompt": "large", "seed": 2, "window_index": 3},
            reference,
            _metric_output(1.0, False),
        ),
    ]

    worst = _worst_cases(cases, fps=20)

    assert worst["mpjpe_m"]["prompt"] == "large"
    assert worst["mpjpe_m"]["seed"] == 2
    assert worst["mpjpe_m"]["window_index"] == 3
    assert worst["contact_agreement"]["prompt"] == "large"


def _detailed_report_with_private_attribution() -> dict:
    return {
        "schema_version": 3,
        "method": {
            "prompt_count": 1,
            "prompt_split": "test",
            "prompt_sha256": "1" * 64,
            "seeds": [7],
        },
        "runtime_environment": {
            "python": {"implementation": "CPython", "version": "3.12.0"},
            "packages": {"onnxruntime": "test"},
        },
        "prompt_manifest": {
            "filename": "prompts.jsonl",
            "size_bytes": 123,
            "sha256": "2" * 64,
            "provenance_sidecar": {
                "filename": "prompts.metadata.json",
                "size_bytes": 456,
                "sha256": "3" * 64,
                "content": {
                    "dataset": {
                        "url": (
                            "https://huggingface.co/datasets/"
                            "nvidia/SEED-Timeline-Annotations"
                        )
                    }
                },
            },
        },
        "models": {
            "reference": {
                "id": "test",
                "revision": "4" * 64,
                "manifest_sha256": "5" * 64,
                "transport_size_bytes": 100,
                "raw_size_bytes": 200,
            },
            "candidate": {
                "id": "test",
                "revision": "4" * 64,
                "manifest_sha256": "6" * 64,
                "transport_size_bytes": 90,
                "raw_size_bytes": 180,
            },
        },
        "contract_validation": {"non_precision_contract_equal": True},
        "text_conditions": {"rmse": 0.0},
        "motion_fidelity": {"mpjpe_m": {"mean": 0.0}},
        "motion_fidelity_by_window": [],
        "continuation_coverage": {"history_frames_observed": [0, 40]},
        "cpu_timing": {"reference_total_seconds": 1.0},
        "worst_cases": {
            "per_window": {
                "mpjpe_m": {
                    "prompt": "private Timeline prompt",
                    "prompt_index": 0,
                    "debug_path": "/home/example/private-output.npy",
                }
            }
        },
    }


def test_public_report_is_allowlisted_aggregate_without_prompt_text_or_paths(
    tmp_path: Path,
):
    detailed = _detailed_report_with_private_attribution()
    detailed_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"

    printed = _write_reports(
        detailed,
        detailed_output=detailed_output,
        public_output=public_output,
    )
    public = json.loads(public_output.read_text(encoding="utf-8"))

    assert printed == public
    assert public["format"] == "ardy-browser-fp16-aggregate"
    assert public["format_version"] == 2
    assert public["source_report_schema_version"] == 3
    assert "worst_cases" not in public
    assert public["runtime_environment"]["python"]["implementation"] == "CPython"
    assert "private Timeline prompt" in detailed_output.read_text(encoding="utf-8")
    public_json = public_output.read_text(encoding="utf-8")
    assert "private Timeline prompt" not in public_json
    assert "/home/example" not in public_json
    assert public["method"]["prompt_sha256"] == "1" * 64
    assert public["prompt_manifest"]["sha256"] == "2" * 64


def test_public_report_rejects_absolute_paths_and_requires_provenance():
    detailed = _detailed_report_with_private_attribution()
    detailed["models"]["reference"]["id"] = "C:\\private\\reference"
    with pytest.raises(ValueError, match="absolute local path"):
        _build_public_report(detailed)

    public = _build_public_report(_detailed_report_with_private_attribution())
    public["models"]["candidate"]["id"] = "/private/candidate"
    with pytest.raises(ValueError, match="absolute local path"):
        _validate_public_report(public)

    missing_provenance = _detailed_report_with_private_attribution()
    missing_provenance["prompt_manifest"].pop("provenance_sidecar")
    with pytest.raises(TypeError, match="requires a validated"):
        _build_public_report(missing_provenance)


@pytest.mark.parametrize(
    "value",
    [
        "../decoder.onnx",
        "/decoder.onnx",
        "graphs//decoder.onnx",
        "graphs/./decoder.onnx",
        "C:\\private\\decoder.onnx",
        "graphs\\decoder.onnx",
    ],
)
def test_model_paths_must_be_safe_canonical_posix(value: str):
    with pytest.raises(ValueError, match="safe canonical POSIX path"):
        _canonical_model_path(value, "test path")


def _gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _write_model_files_input(
    directory: Path,
    payloads: dict[str, bytes],
    *,
    mutate_manifest=None,
) -> tuple[dict, bytes]:
    directory.mkdir()
    files = {}
    for relative_path, payload in payloads.items():
        transport_path = f"{relative_path}.gz"
        compressed = _gzip_bytes(payload)
        destination = directory / transport_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(compressed)
        files[relative_path] = {
            "sha256": _digest(payload),
            "size_bytes": len(payload),
            "transport": {
                "path": transport_path,
                "compression": "gzip",
                "sha256": _digest(compressed),
                "size_bytes": len(compressed),
            },
        }
    manifest = {
        "format": "ardy-browser-model-files",
        "schema_version": 1,
        "model": {
            "id": "test-model",
            "revision": "7" * 64,
            "variant": "test",
        },
        "files": files,
        "tokenizer": {"directory": "tokenizer", "max_length": 128},
        "graphs": {
            "text_encoder": {"model": "text.onnx"},
            "denoiser": {"model": "denoiser.onnx"},
            "decoder": {"model": "decoder.onnx"},
        },
        "runtime": {
            "contract_revision": 3,
            "required_webgpu_features": ["shader-f16"],
            "text_only": True,
        },
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    encoded = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (directory / "model.json.gz").write_bytes(_gzip_bytes(encoded))
    return manifest, encoded


def _model_payloads() -> dict[str, bytes]:
    return {
        "text.onnx": b"text",
        "denoiser.onnx": b"denoiser",
        "decoder.onnx": b"decoder",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b'{"model_max_length":128}',
    }


def test_model_files_are_verified_and_decompressed_with_portable_identity(
    tmp_path: Path,
):
    source = tmp_path / "source"
    payloads = _model_payloads()
    manifest, encoded_manifest = _write_model_files_input(source, payloads)

    prepared = _prepare_model_files(source, tmp_path / "prepared")

    assert prepared.manifest == manifest
    assert prepared.identity.to_report() == {
        "id": "test-model",
        "revision": "7" * 64,
        "manifest_sha256": _digest(encoded_manifest),
        "transport_size_bytes": (
            (source / "model.json.gz").stat().st_size
            + sum(
                record["transport"]["size_bytes"]
                for record in manifest["files"].values()
            )
        ),
        "raw_size_bytes": len(encoded_manifest) + sum(map(len, payloads.values())),
    }
    for relative_path, payload in payloads.items():
        assert (prepared.directory / relative_path).read_bytes() == payload


def test_compressed_manifest_must_be_valid_unique_finite_json(tmp_path: Path):
    invalid_gzip = tmp_path / "invalid-gzip"
    invalid_gzip.mkdir()
    (invalid_gzip / "model.json.gz").write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="valid gzip"):
        _prepare_model_files(invalid_gzip, tmp_path / "invalid-gzip-output")

    duplicate_key = tmp_path / "duplicate-key"
    duplicate_key.mkdir()
    (duplicate_key / "model.json.gz").write_bytes(
        _gzip_bytes(b'{"format":"first","format":"second"}')
    )
    with pytest.raises(ValueError, match="duplicate key"):
        _prepare_model_files(duplicate_key, tmp_path / "duplicate-key-output")

    nonfinite = tmp_path / "nonfinite"
    nonfinite.mkdir()
    (nonfinite / "model.json.gz").write_bytes(
        _gzip_bytes(b'{"format":"ardy-browser-model-files","value":NaN}')
    )
    with pytest.raises(ValueError, match="must be finite"):
        _prepare_model_files(nonfinite, tmp_path / "nonfinite-output")


def test_model_files_reject_unsafe_duplicate_and_undeclared_transports(
    tmp_path: Path,
):
    unsafe = tmp_path / "unsafe"
    _write_model_files_input(
        unsafe,
        _model_payloads(),
        mutate_manifest=lambda manifest: manifest["files"]["decoder.onnx"][
            "transport"
        ].__setitem__("path", "../decoder.onnx.gz"),
    )
    with pytest.raises(ValueError, match="safe canonical POSIX path"):
        _prepare_model_files(unsafe, tmp_path / "unsafe-output")

    duplicate = tmp_path / "duplicate"
    _write_model_files_input(
        duplicate,
        _model_payloads(),
        mutate_manifest=lambda manifest: manifest["files"]["decoder.onnx"][
            "transport"
        ].__setitem__(
            "path",
            manifest["files"]["denoiser.onnx"]["transport"]["path"],
        )
    )
    with pytest.raises(ValueError, match="Duplicate model transport path"):
        _prepare_model_files(duplicate, tmp_path / "duplicate-output")

    extra = tmp_path / "extra"
    _write_model_files_input(extra, _model_payloads())
    (extra / "undeclared.gz").write_bytes(_gzip_bytes(b"undeclared"))
    with pytest.raises(ValueError, match="do not match manifest transports"):
        _prepare_model_files(extra, tmp_path / "extra-output")


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("transport", "size_bytes"), 1, "Compressed size mismatch"),
        (("transport", "sha256"), "0" * 64, "Compressed SHA-256 mismatch"),
        (("size_bytes",), 1, "Decompressed size"),
        (("sha256",), "0" * 64, "Decompressed SHA-256 mismatch"),
    ],
)
def test_model_files_reject_declared_size_and_hash_mismatches(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value,
    message: str,
):
    def mutate(manifest: dict) -> None:
        record = manifest["files"]["decoder.onnx"]
        for key in field_path[:-1]:
            record = record[key]
        record[field_path[-1]] = value

    source = tmp_path / "source"
    _write_model_files_input(source, _model_payloads(), mutate_manifest=mutate)
    with pytest.raises(ValueError, match=message):
        _prepare_model_files(source, tmp_path / "output")


def test_manifest_rejects_external_onnx_data_before_model_loading(tmp_path: Path):
    source = tmp_path / "source"
    _write_model_files_input(
        source,
        _model_payloads(),
        mutate_manifest=lambda manifest: manifest["graphs"]["text_encoder"].__setitem__(
            "external_data",
            [{"path": "../../private.bin", "file": "private.bin"}],
        ),
    )
    prepared = _prepare_model_files(source, tmp_path / "prepared")

    with pytest.raises(ValueError, match="must not declare external ONNX data"):
        _validate_common_manifest(
            prepared.directory,
            prepared.manifest,
        )


def test_file_identity_recheck_detects_mutation(tmp_path: Path):
    path = tmp_path / "model.json.gz"
    path.write_bytes(b"first")
    identity = _capture_file_identity(path)
    path.write_bytes(b"second")

    with pytest.raises(RuntimeError, match="changed after"):
        _verify_file_identity(path, identity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reports_reject_nonfinite_numbers_and_never_write_nan(
    tmp_path: Path,
    value: float,
):
    report = {"metric": value}
    with pytest.raises(ValueError, match="must be finite"):
        _validate_json_finite(report)

    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="must be finite"):
        _write_json_report(output, report)
    assert not output.exists()


def test_reference_model_must_explicitly_declare_fp32(tmp_path: Path):
    reference = _model_runtime(tmp_path / "reference", candidate=False)
    candidate = _model_runtime(tmp_path / "candidate", candidate=True)
    reference.manifest.pop("precision")

    with pytest.raises(ValueError, match="precision.format='fp32'"):
        _validate_compatible_models(reference, candidate)
