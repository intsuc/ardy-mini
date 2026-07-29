# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Tests for continuous browser-style FP32/mixed-FP16 evaluation."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from ardy.browser.precision import MIXED_FP16_POLICIES, MIXED_FP16_POLICY_VERSION
from scripts.evaluate_browser_fp16 import (
    PackRuntime,
    PortableRandom,
    PreparedHistory,
    RolloutState,
    _make_window,
    _prepare_history,
    _recenter_and_requantize,
    _validate_compatible_packs,
    _worst_cases,
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


def _pack_runtime(directory: Path, *, candidate: bool) -> PackRuntime:
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
        "format": "ardy-browser-model-pack",
        "schema_version": 2,
        "model": {"id": "test", "variant": "test"},
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
            **({"required_webgpu_features": ["shader-f16"]} if candidate else {}),
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
    return PackRuntime(
        directory=directory,
        manifest=manifest,
        text_encoder=None,
        denoiser=None,
        decoder=None,
    )


def test_pack_compatibility_allows_only_precision_metadata(tmp_path: Path):
    reference = _pack_runtime(tmp_path / "reference", candidate=False)
    candidate = _pack_runtime(tmp_path / "candidate", candidate=True)

    validation = _validate_compatible_packs(reference, candidate)
    assert validation["candidate_policy_version"] == MIXED_FP16_POLICY_VERSION
    assert validation["identity_graphs_byte_identical"] == [
        "denoiser",
        "text_encoder",
    ]

    incompatible = copy.deepcopy(candidate)
    incompatible.manifest["recenter"]["root_mean"][0] = 1
    with pytest.raises(ValueError, match=r"non-precision contracts differ.*recenter\.root_mean"):
        _validate_compatible_packs(reference, incompatible)


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
def test_pack_compatibility_rejects_nonproduction_precision_contract(
    tmp_path: Path,
    mutation,
    message: str,
):
    reference = _pack_runtime(tmp_path / "reference", candidate=False)
    candidate = _pack_runtime(tmp_path / "candidate", candidate=True)
    mutation(candidate.manifest)

    with pytest.raises(ValueError, match=message):
        _validate_compatible_packs(reference, candidate)


@pytest.mark.parametrize("graph_name", ["text_encoder", "denoiser"])
def test_pack_compatibility_rejects_nonidentical_identity_graph(
    tmp_path: Path,
    graph_name: str,
):
    reference = _pack_runtime(tmp_path / "reference", candidate=False)
    candidate = _pack_runtime(tmp_path / "candidate", candidate=True)
    tampered = f"quantized {graph_name}".encode()
    graph_path = candidate.manifest["graphs"][graph_name]["model"]
    (candidate.directory / graph_path).write_bytes(tampered)
    candidate.manifest["precision"]["graphs"][graph_name]["output_sha256"] = _digest(tampered)
    candidate.manifest["precision"]["graphs"][graph_name]["output_size_bytes"] = len(tampered)

    with pytest.raises(ValueError, match=rf"{graph_name} must be byte-identical"):
        _validate_compatible_packs(reference, candidate)


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
