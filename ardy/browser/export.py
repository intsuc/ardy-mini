# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Build the self-contained ONNX model pack consumed by the browser demo."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ardy.model.load_model import load_model
from ardy.model.minilm_encoder import (
    BACKBONE_DIR,
    MotionConditionStudent,
)
from ardy.model.registry import resolve_model_name

from .wrappers import (
    BrowserMiniLMEncoder,
    BrowserMotionDecoder,
    BrowserTextCFGDenoiser,
)

BROWSER_PACK_FORMAT = "ardy-browser-model-pack"
BROWSER_PACK_SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "ardy-minilm-core40-browser-v1"
DEFAULT_MAX_TOKENS = 20
DEFAULT_MAX_PROMPT_TOKENS = 128
DEFAULT_MAX_OUTPUT_FRAMES = 200
_ONNX_EXPORT_LOCK = threading.Lock()
_NUMERIC_ERROR_LIMITS = {
    "text_encoder": {"text_conditions": 1e-3},
    "denoiser": {"pred_x0": 1e-3},
    "decoder": {
        "normalized_motion": 2e-3,
        "posed_joints": 1e-3,
    },
}


@contextmanager
def _onnx_export_mode():
    """Serialize legacy ONNX tracing and disable the MHA inference fast path."""
    with _ONNX_EXPORT_LOCK:
        mha_backend = getattr(torch.backends, "mha", None)
        previous_fastpath = None
        if mha_backend is not None:
            previous_fastpath = mha_backend.get_fastpath_enabled()
            mha_backend.set_fastpath_enabled(False)
        try:
            try:
                from torch.onnx._internal.torchscript_exporter.utils import (
                    GLOBALS,
                )

                GLOBALS.in_onnx_export = False
            except (ImportError, AttributeError):
                pass
            yield
        finally:
            if mha_backend is not None and previous_fastpath is not None:
                mha_backend.set_fastpath_enabled(previous_fastpath)


@dataclass(frozen=True)
class BrowserExportConfig:
    """Inputs and fixed dimensions for one browser model-pack export."""

    output_dir: Path
    minilm_artifact: Path
    checkpoints_dir: Path | None = None
    model: str = "core"
    model_id: str = DEFAULT_MODEL_ID
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    max_output_frames: int = DEFAULT_MAX_OUTPUT_FRAMES
    opset: int = 17
    device: str = "auto"
    verify: bool = True


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _tensor_values(tensor: torch.Tensor) -> list:
    return tensor.detach().to(device="cpu", dtype=torch.float32).tolist()


def _stats_payload(stats) -> dict[str, list]:
    return {
        "mean": _tensor_values(stats.mean),
        "std": _tensor_values(stats.std),
        "normalization_denominator": _tensor_values(stats.std_eps),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for browser export, but CUDA is unavailable.")
    return device


def _validate_config(config: BrowserExportConfig) -> None:
    if config.opset < 17:
        raise ValueError("Browser export requires ONNX opset 17 or newer.")
    if config.max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if config.max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive.")
    if config.max_output_frames <= 0:
        raise ValueError("max_output_frames must be positive.")
    if not config.minilm_artifact.is_dir():
        raise FileNotFoundError(f"MiniLM artifact directory not found: {config.minilm_artifact}")


def _export_graph(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    module.eval()
    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            module,
            args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def _copy_tokenizer_files(
    artifact_dir: Path,
    output_dir: Path,
) -> list[Path]:
    source_dir = artifact_dir / BACKBONE_DIR
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"MiniLM tokenizer payload is missing required file: {source}")
        destination = tokenizer_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _make_text_dummy(tokenizer, max_length: int, device: torch.device):
    encoded = tokenizer(
        ["A person walks forward."],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    required = ("input_ids", "attention_mask", "token_type_ids")
    missing = [name for name in required if name not in encoded]
    if missing:
        raise ValueError(f"MiniLM tokenizer did not return required browser inputs: {missing}")
    return tuple(encoded[name].to(device=device) for name in required)


def _make_denoiser_dummy(
    model,
    max_tokens: int,
    text_condition_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    nfpt = int(model.num_frames_per_token)
    max_frames = max_tokens * nfpt
    generation_frames = int(model.gen_horizon_len)
    generation_tokens = generation_frames // nfpt
    history_tokens = max_tokens - generation_tokens
    history_frames = history_tokens * nfpt

    history_mask = torch.zeros(1, max_frames, device=device)
    history_mask[:, :history_frames] = 1
    generation_mask = torch.zeros(1, max_frames, device=device)
    generation_mask[:, history_frames : history_frames + generation_frames] = 1
    history_token_mask = torch.zeros(1, max_tokens, device=device)
    history_token_mask[:, :history_tokens] = 1
    generation_token_mask = torch.zeros(1, max_tokens, device=device)
    generation_token_mask[
        :,
        history_tokens : history_tokens + generation_tokens,
    ] = 1

    generator = torch.Generator(device=device)
    generator.manual_seed(20260728)
    return (
        torch.tensor([2.0], dtype=torch.float32, device=device),
        torch.randn(
            1,
            max_tokens,
            model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim,
            generator=generator,
            device=device,
        ),
        torch.tensor([history_frames], dtype=torch.int64, device=device),
        torch.tensor([generation_frames], dtype=torch.int64, device=device),
        history_mask,
        generation_mask,
        history_token_mask,
        generation_token_mask,
        torch.randn(
            1,
            1,
            text_condition_dim,
            generator=generator,
            device=device,
        ),
        torch.tensor(
            [model.diffusion.num_base_steps - 1],
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(1, dtype=torch.float32, device=device),
    )


def _make_decoder_dummy(
    model,
    max_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    nfpt = int(model.num_frames_per_token)
    max_frames = max_tokens * nfpt
    generator = torch.Generator(device=device)
    generator.manual_seed(20260729)
    hybrid = torch.randn(
        1,
        max_tokens,
        model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim,
        generator=generator,
        device=device,
    )
    # Browser generation requantizes latent history before decoding.  Keeping
    # the verification input on the FSQ grid avoids meaningless PyTorch-vs-ONNX
    # differences when a random value sits exactly on a rounding boundary.
    root_tokens = hybrid[:, :, : model.denoiser.nframe_root_dim]
    latent_tokens = model.autoencoder.requantize(hybrid[:, :, model.denoiser.nframe_root_dim :])
    hybrid = torch.cat((root_tokens, latent_tokens), dim=-1)
    motion_pad_mask = torch.ones(
        1,
        max_frames,
        dtype=torch.float32,
        device=device,
    )
    global_translation = torch.zeros(
        1,
        3,
        dtype=torch.float32,
        device=device,
    )
    return hybrid, motion_pad_mask, global_translation


def _check_onnx_models(paths: list[Path]) -> None:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX validation requires the `onnx` package. Run the exporter with `uv run --with onnx ...`."
        ) from error
    for path in paths:
        onnx.checker.check_model(str(path))


def _run_ort(
    path: Path,
    inputs: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Numeric browser-export validation requires `onnxruntime`. "
            "Run with `uv run --with onnx --with onnxruntime ...`, or pass "
            "`--skip-verify` for an export-only run."
        ) from error

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, max(1, torch.get_num_threads()))
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        name: value.detach().to(device="cpu").numpy()
        for name, value in inputs.items()
        if name in {item.name for item in session.get_inputs()}
    }
    values = session.run(None, feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), values)}


def _max_abs(reference: torch.Tensor, actual: np.ndarray) -> float:
    expected = reference.detach().to(device="cpu", dtype=torch.float32).numpy()
    return float(np.max(np.abs(expected - actual)))


def _verify_numeric(
    graph_paths: dict[str, Path],
    modules: dict[str, torch.nn.Module],
    dummy_inputs: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, Any]:
    input_names = {
        "text_encoder": (
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ),
        "denoiser": (
            "cfg_weight",
            "x",
            "history_len",
            "generation_len",
            "history_mask",
            "generation_mask",
            "history_token_mask",
            "generation_token_mask",
            "text_conditions",
            "timestep",
            "first_heading_angle",
        ),
        "decoder": (
            "hybrid_tokens",
            "motion_pad_mask",
            "global_translation",
        ),
    }
    output_names = {
        "text_encoder": ("text_conditions",),
        "denoiser": ("pred_x0",),
        "decoder": ("normalized_motion", "posed_joints"),
    }

    results: dict[str, Any] = {
        "backend": "onnxruntime-cpu",
        "reference": "pytorch-fp32-mha-fastpath-disabled",
        "max_abs_error": {},
        "max_abs_error_limit": _NUMERIC_ERROR_LIMITS,
    }
    for graph_name in ("text_encoder", "denoiser", "decoder"):
        module = modules[graph_name]
        args = dummy_inputs[graph_name]
        # Match the attention implementation used while tracing. PyTorch's
        # native MHA inference fast path is numerically different from the
        # portable primitive graph emitted for ONNX, despite both being valid
        # FP32 executions of the same weights.
        with _onnx_export_mode(), torch.no_grad():
            reference = module(*args)
        references = reference if isinstance(reference, tuple) else (reference,)
        ort_outputs = _run_ort(
            graph_paths[graph_name],
            dict(zip(input_names[graph_name], args)),
        )
        errors = {
            output_name: _max_abs(expected, ort_outputs[output_name])
            for output_name, expected in zip(
                output_names[graph_name],
                references,
            )
        }
        if not all(math.isfinite(value) for value in errors.values()):
            raise RuntimeError(f"Non-finite numeric verification result for {graph_name}: {errors}")
        limits = _NUMERIC_ERROR_LIMITS[graph_name]
        failures = {
            name: {"actual": value, "limit": limits[name]} for name, value in errors.items() if value > limits[name]
        }
        if failures:
            raise RuntimeError(f"ONNX numeric verification failed for {graph_name}: {failures}")
        results["max_abs_error"][graph_name] = errors
    return results


def _graph_contracts() -> dict[str, Any]:
    return {
        "text_encoder": {
            "model": "text_encoder.onnx",
            "inputs": {
                "inputIds": "input_ids",
                "attentionMask": "attention_mask",
                "tokenTypeIds": "token_type_ids",
            },
            "outputs": {"textConditions": "text_conditions"},
            "io": {
                "input_ids": {"dtype": "int64", "shape": [1, "sequence"]},
                "attention_mask": {
                    "dtype": "int64",
                    "shape": [1, "sequence"],
                },
                "token_type_ids": {
                    "dtype": "int64",
                    "shape": [1, "sequence"],
                },
                "text_conditions": {
                    "dtype": "float32",
                    "shape": [1, 1, 2048],
                },
            },
        },
        "denoiser": {
            "model": "denoiser.onnx",
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
            "model": "decoder.onnx",
            "inputs": {
                "hybridTokens": "hybrid_tokens",
                "motionPadMask": "motion_pad_mask",
                "globalTranslation": "global_translation",
            },
            "outputs": {
                "normalizedMotion": "normalized_motion",
                "posedJoints": "posed_joints",
            },
        },
    }


def _build_manifest(
    *,
    config: BrowserExportConfig,
    model,
    resolved_model: str,
    artifact_config: dict[str, Any],
    output_dir: Path,
    payload_paths: list[Path],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    motion_rep = model.motion_rep
    autoencoder = model.autoencoder
    nfpt = int(model.num_frames_per_token)
    max_frames = config.max_tokens * nfpt
    generation_frames = int(model.gen_horizon_len)
    generation_tokens = generation_frames // nfpt
    history_frames = max_frames - generation_frames
    history_tokens = config.max_tokens - generation_tokens

    files = {path.relative_to(output_dir).as_posix(): _file_record(path) for path in sorted(payload_paths)}
    parents = [int(value) for value in motion_rep.skeleton.joint_parents.detach().cpu().tolist()]
    levels = [int(value) for value in autoencoder.quantizer._levels.detach().cpu().tolist()]

    diffusion = model.diffusion
    alphas_cumprod = diffusion.alphas_cumprod.detach().cpu()
    alphas_cumprod_prev = diffusion.alphas_cumprod_prev.detach().cpu()

    manifest: dict[str, Any] = {
        "format": BROWSER_PACK_FORMAT,
        "schema_version": BROWSER_PACK_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": config.model_id,
            "variant": "MiniLM Core40 text-only",
            "ardy_model": resolved_model,
            "minilm_artifact_fingerprint": artifact_config.get("artifact_fingerprint"),
        },
        "files": files,
        "tokenizer": {
            "directory": "tokenizer",
            "max_length": config.max_prompt_tokens,
            "model_id": artifact_config["base_model"],
        },
        "graphs": _graph_contracts(),
        "dimensions": {
            "fps": int(motion_rep.fps),
            "num_frames_per_token": nfpt,
            "max_tokens": config.max_tokens,
            "max_frames": max_frames,
            "generation_tokens": generation_tokens,
            "generation_frames": generation_frames,
            "history_tokens": history_tokens,
            "history_frames": history_frames,
            "root_features_per_frame": int(motion_rep.motion_root_dim),
            "nframe_root_dim": int(model.denoiser.nframe_root_dim),
            "latent_dim": int(model.denoiser.latent_embedding_dim),
            "hybrid_dim": int(model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim),
            "motion_dim": int(motion_rep.motion_rep_dim),
            "body_dim": int(motion_rep.body_dim),
            "text_condition_dim": int(artifact_config["output_dim"]),
            "num_joints": int(motion_rep.skeleton.nbjoints),
        },
        "generation": {
            "min_frames": generation_frames,
            "max_frames": config.max_output_frames,
            "default_cfg_weight": 2.0,
            "denoising_steps": int(diffusion.num_base_steps),
            "window_policy": (
                "Generate 40 frames per step. For continuation, retain and recenter the preceding 40 frames."
            ),
        },
        "diffusion": {
            "timesteps": list(range(int(diffusion.num_base_steps) - 1, -1, -1)),
            "betas": _tensor_values(diffusion.betas),
            "alphas_cumprod": _tensor_values(alphas_cumprod),
            "alphas_cumprod_prev": _tensor_values(alphas_cumprod_prev),
            "sampler": "deterministic-ddim-eta-0",
        },
        "recenter": {
            "root_mean": _tensor_values(motion_rep.global_root_stats.mean),
            "root_std": _tensor_values(motion_rep.global_root_stats.std_eps),
            "position_indices": [0, 1, 2],
            "heading_indices": [3, 4],
            "translation_axes": [0, 2],
        },
        "latent_quantization": {
            "levels": levels,
            "mean": _tensor_values(autoencoder.post_quantization_stats.mean),
            "std": _tensor_values(autoencoder.post_quantization_stats.std_eps),
            "clamp": [-1.0, 1.0],
            "rounding": "round(clamp(x,-1,1)*(levels//2))/(levels//2)",
        },
        "skeleton": {
            "name": motion_rep.skeleton.name,
            "root_index": int(motion_rep.skeleton.root_idx),
            "parents": parents,
            "joint_names": list(motion_rep.skeleton.bone_order_names),
            "neutral_joints": _tensor_values(motion_rep.skeleton.neutral_joints),
        },
        "stats": {
            "motion": _stats_payload(motion_rep.stats),
            "global_root": _stats_payload(motion_rep.global_root_stats),
            "local_root": _stats_payload(motion_rep.local_root_stats),
            "body": _stats_payload(motion_rep.body_stats),
            "post_quantization": _stats_payload(autoencoder.post_quantization_stats),
        },
        "motion_layout": {name: [int(value.start), int(value.stop)] for name, value in motion_rep.slice_dict.items()},
        "runtime": {
            "onnx_opset": config.opset,
            "batch_size": 1,
            "text_only": True,
            "constraints_supported": False,
            "motion_correction_included": False,
            "global_translation_y_must_be_zero": True,
        },
        "notices": [
            "Licensed by NVIDIA Corporation under the NVIDIA Open Model License.",
            ("The specialized text encoder is based on sentence-transformers/all-MiniLM-L6-v2."),
            (
                "Review THIRD_PARTY_MODELS_AND_DATA.md and all source "
                "model/data terms before distributing this model pack."
            ),
        ],
        "license_notices": [
            {
                "component": "ARDY source code",
                "license": "Apache-2.0",
                "notice": ("Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."),
            },
            {
                "component": "ARDY model",
                "license": "NVIDIA Open Model License",
                "notice": ("Licensed by NVIDIA Corporation under the NVIDIA Open Model License."),
            },
            {
                "component": "sentence-transformers/all-MiniLM-L6-v2",
                "license": "Apache-2.0",
                "notice": ("The specialized text encoder is based on sentence-transformers/all-MiniLM-L6-v2."),
            },
            {
                "component": "specialized MiniLM condition weights",
                "license": "review-required-before-redistribution",
                "notice": (
                    "This local export contains trained weights. Review "
                    "THIRD_PARTY_MODELS_AND_DATA.md and all source model/data "
                    "terms before distributing the model pack."
                ),
            },
        ],
    }
    if verification is not None:
        manifest["verification"] = verification
    return manifest


def export_browser_model_pack(config: BrowserExportConfig) -> Path:
    """Export and validate the three-graph Core40 browser model pack."""
    _validate_config(config)
    device = _resolve_device(config.device)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_name(
        config.model,
        checkpoints_dir=(str(config.checkpoints_dir) if config.checkpoints_dir is not None else None),
    )
    model = load_model(
        resolved_model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=(str(config.checkpoints_dir) if config.checkpoints_dir is not None else None),
    )
    model.eval()

    if int(model.gen_horizon_len) % int(model.num_frames_per_token) != 0:
        raise ValueError("ARDY generation horizon must be divisible by num_frames_per_token.")
    generation_tokens = int(model.gen_horizon_len // model.num_frames_per_token)
    if config.max_tokens != 2 * generation_tokens:
        raise ValueError(
            "The browser v1 contract requires exactly one 40-frame history "
            "window plus one 40-frame generation window; expected "
            f"max_tokens={2 * generation_tokens}, got {config.max_tokens}."
        )

    student, tokenizer, artifact_config = MotionConditionStudent.from_artifact(config.minilm_artifact)
    if resolved_model not in artifact_config["compatible_ardy_models"]:
        raise ValueError(
            f"MiniLM artifact is not compatible with {resolved_model!r}; "
            f"declared compatibility is "
            f"{artifact_config['compatible_ardy_models']}."
        )
    if int(artifact_config["output_dim"]) != 2048:
        raise ValueError(
            f"Browser v1 requires a 2048-dimensional direct root/body condition, got {artifact_config['output_dim']}."
        )
    if config.max_prompt_tokens > int(artifact_config["max_length"]):
        raise ValueError(
            f"max_prompt_tokens={config.max_prompt_tokens} exceeds the "
            f"artifact maximum {artifact_config['max_length']}."
        )

    # Eager attention plus an explicit 4-D additive mask gives a stable
    # opset-17 graph across Transformers 5.x.
    student.backbone.set_attn_implementation("eager")
    student.to(device=device, dtype=torch.float32).eval()

    modules: dict[str, torch.nn.Module] = {
        "text_encoder": BrowserMiniLMEncoder(student).to(device).eval(),
        "denoiser": BrowserTextCFGDenoiser(model.denoiser.model).to(device).eval(),
        "decoder": BrowserMotionDecoder(
            model.autoencoder,
            model.motion_rep,
        )
        .to(device)
        .eval(),
    }
    dummy_inputs = {
        "text_encoder": _make_text_dummy(
            tokenizer,
            config.max_prompt_tokens,
            device,
        ),
        "denoiser": _make_denoiser_dummy(
            model,
            config.max_tokens,
            int(artifact_config["output_dim"]),
            device,
        ),
        "decoder": _make_decoder_dummy(
            model,
            config.max_tokens,
            device,
        ),
    }

    graph_paths = {
        "text_encoder": output_dir / "text_encoder.onnx",
        "denoiser": output_dir / "denoiser.onnx",
        "decoder": output_dir / "decoder.onnx",
    }
    _export_graph(
        modules["text_encoder"],
        dummy_inputs["text_encoder"],
        graph_paths["text_encoder"],
        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],
        output_names=["text_conditions"],
        dynamic_axes={
            "input_ids": {1: "sequence"},
            "attention_mask": {1: "sequence"},
            "token_type_ids": {1: "sequence"},
        },
        opset=config.opset,
    )
    _export_graph(
        modules["denoiser"],
        dummy_inputs["denoiser"],
        graph_paths["denoiser"],
        input_names=[
            "cfg_weight",
            "x",
            "history_len",
            "generation_len",
            "history_mask",
            "generation_mask",
            "history_token_mask",
            "generation_token_mask",
            "text_conditions",
            "timestep",
            "first_heading_angle",
        ],
        output_names=["pred_x0"],
        opset=config.opset,
    )
    _export_graph(
        modules["decoder"],
        dummy_inputs["decoder"],
        graph_paths["decoder"],
        input_names=[
            "hybrid_tokens",
            "motion_pad_mask",
            "global_translation",
        ],
        output_names=["normalized_motion", "posed_joints"],
        opset=config.opset,
    )

    _check_onnx_models(list(graph_paths.values()))
    verification = _verify_numeric(graph_paths, modules, dummy_inputs) if config.verify else None

    tokenizer_paths = _copy_tokenizer_files(
        config.minilm_artifact,
        output_dir,
    )
    payload_paths = list(graph_paths.values()) + tokenizer_paths
    manifest = _build_manifest(
        config=config,
        model=model,
        resolved_model=resolved_model,
        artifact_config=artifact_config,
        output_dir=output_dir,
        payload_paths=payload_paths,
        verification=verification,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "BROWSER_PACK_FORMAT",
    "BROWSER_PACK_SCHEMA_VERSION",
    "BrowserExportConfig",
    "export_browser_model_pack",
]
