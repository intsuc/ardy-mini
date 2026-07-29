# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Compare an FP32 browser pack with a mixed-FP16 pack on fixed rollouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ardy.browser.precision import (
    MIXED_FP16_POLICIES,
    MIXED_FP16_POLICY_VERSION,
)


@dataclass(frozen=True)
class PackRuntime:
    directory: Path
    manifest: dict[str, Any]
    text_encoder: Any
    denoiser: Any
    decoder: Any


@dataclass
class RolloutState:
    """Browser-equivalent autoregressive state for one model pack."""

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
            "FP32 reference pack and a mixed-FP16 candidate pack."
        )
    )
    parser.add_argument("--reference-pack", type=Path, required=True)
    parser.add_argument("--candidate-pack", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL records containing a `text` field.")
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pack(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if not member.isfile() or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe browser-pack member: {member.name!r}")
            destination = output_dir / member_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read browser-pack member: {member.name!r}")
            with destination.open("wb") as output:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)


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


def _load_runtime(directory: Path) -> PackRuntime:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    graphs = manifest["graphs"]
    return PackRuntime(
        directory=directory,
        manifest=manifest,
        text_encoder=_session(directory / graphs["text_encoder"]["model"]),
        denoiser=_session(directory / graphs["denoiser"]["model"]),
        decoder=_session(directory / graphs["decoder"]["model"]),
    )


def _without_precision_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic model contract shared by FP32 and mixed packs."""
    contract = copy.deepcopy(manifest)
    for key in ("files", "precision", "verification"):
        contract.pop(key, None)
    runtime = contract.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("required_webgpu_features", None)
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


def _tokenizer_payloads(runtime: PackRuntime) -> dict[str, str]:
    directory = str(runtime.manifest["tokenizer"]["directory"])
    payloads: dict[str, str] = {}
    for relative_path in sorted(runtime.manifest["files"]):
        if relative_path.startswith(f"{directory}/"):
            payloads[relative_path] = _sha256(runtime.directory / relative_path)
    return payloads


def _validate_compatible_packs(reference: PackRuntime, candidate: PackRuntime) -> dict[str, Any]:
    """Reject comparisons that change anything except ONNX precision."""
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
        raise ValueError("Candidate pack must declare precision.format='mixed-fp16'.")
    if precision.get("policy_version") != MIXED_FP16_POLICY_VERSION:
        raise ValueError(f"Candidate pack must declare precision.policy_version={MIXED_FP16_POLICY_VERSION}.")
    summaries = precision.get("graphs")
    if not isinstance(summaries, dict) or set(summaries) != set(MIXED_FP16_POLICIES):
        raise ValueError("Candidate pack precision.graphs must contain exactly the three production graphs.")

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
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_prompts(path: Path, split: str, count: int) -> list[str]:
    if count <= 0:
        raise ValueError("--count must be positive.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [str(record["text"]) for record in records if split == "all" or record.get("split") == split]
    unique = list(dict.fromkeys(eligible))
    if len(unique) < count:
        raise ValueError(f"Only {len(unique)} unique prompts are available for split {split!r}; {count} requested.")
    indices = np.linspace(0, len(unique) - 1, count, dtype=np.int64)
    return [unique[int(index)] for index in indices]


def _encode(runtime: PackRuntime, tokenizer, prompt: str) -> np.ndarray:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=runtime.manifest["tokenizer"]["max_length"],
        return_tensors="np",
    )
    encoded.setdefault("token_type_ids", np.zeros_like(encoded["input_ids"]))
    feeds = {name: encoded[name].astype(np.int64) for name in ("input_ids", "attention_mask", "token_type_ids")}
    return runtime.text_encoder.run(None, feeds)[0].astype(np.float32)


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
    runtime: PackRuntime,
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
    runtime: PackRuntime,
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
    return {output.name: value for output, value in zip(runtime.decoder.get_outputs(), outputs)}


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
    runtime: PackRuntime,
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


def main() -> None:
    args = _parse_args()
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
    prompts = _select_prompts(args.prompts, args.split, args.count)

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Run this script with `uv run --extra browser`.") from error

    with tempfile.TemporaryDirectory(prefix="ardy-fp16-eval-") as temporary:
        temporary_dir = Path(temporary)
        reference_dir = temporary_dir / "reference"
        candidate_dir = temporary_dir / "candidate"
        reference_dir.mkdir()
        candidate_dir.mkdir()
        _extract_pack(args.reference_pack, reference_dir)
        _extract_pack(args.candidate_pack, candidate_dir)
        reference = _load_runtime(reference_dir)
        candidate = _load_runtime(candidate_dir)
        precision_validation = _validate_compatible_packs(reference, candidate)
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
            "schema_version": 2,
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
                "seeds": seeds,
                "case_count": len(prompts) * len(seeds),
                "cfg_weight": args.cfg_weight,
                "window_count": args.windows,
                "frames_per_case": accumulated_frames,
                "noise_generator": "browser PortableRandom (Mulberry32 + Marsaglia polar)",
                "initial_translation": initial_translation.astype(float).tolist(),
                "initial_heading": initial_heading,
            },
            "packs": {
                "reference": {
                    "file": args.reference_pack.name,
                    "size_bytes": args.reference_pack.stat().st_size,
                    "sha256": _sha256(args.reference_pack),
                },
                "candidate": {
                    "file": args.candidate_pack.name,
                    "size_bytes": args.candidate_pack.stat().st_size,
                    "sha256": _sha256(args.candidate_pack),
                },
                "saved_bytes": args.reference_pack.stat().st_size - args.candidate_pack.stat().st_size,
                "saved_fraction": 1 - args.candidate_pack.stat().st_size / args.reference_pack.stat().st_size,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
