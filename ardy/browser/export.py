# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Build the compressed ONNX model files consumed by the browser runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
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

from .precision import (
    MIXED_FP16_POLICY_VERSION,
    MixedPrecisionReport,
    convert_browser_onnx_to_mixed_fp16,
)
from .wrappers import (
    BrowserMiniLMEncoder,
    BrowserMotionDecoder,
    BrowserTextCFGDenoiser,
)

BROWSER_MODEL_FILES_FORMAT = "ardy-browser-model-files"
BROWSER_MODEL_FILES_SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "ardy-minilm-core40-browser-v1"
FP16_VARIANT_DIRECTORY = "fp16"
FP32_VARIANT_DIRECTORY = "fp32"
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
        "local_rotations": 2e-3,
        "global_rotations": 2e-3,
        "root_positions": 1e-3,
        "foot_contacts": 0.0,
        "global_root_heading": 1e-3,
    },
}
_MIXED_PRECISION_LIMITS = {
    "text_encoder": {
        "text_conditions": {
            "rmse": 2e-2,
            "minimum_cosine_similarity": 0.9999,
        },
    },
    "denoiser": {
        "pred_x0": {
            "rmse": 0.0,
            "minimum_cosine_similarity": 1.0,
            "require_exact": True,
        },
    },
    "decoder": {
        "normalized_motion": {"rmse": 6e-2},
        "posed_joints": {"rmse": 2e-2},
        "local_rotations": {"rmse": 8e-2},
        "global_rotations": {"rmse": 8e-2},
        "root_positions": {"rmse": 1e-6},
        "foot_contacts": {"minimum_agreement": 1.0},
        "global_root_heading": {"rmse": 1e-6},
    },
}
_LOCAL_CHECKPOINT_FILES = (
    "config.yaml",
    "denoiser.safetensors",
    "tokenizer.safetensors",
)


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
    """Inputs and fixed dimensions for one browser model-family export."""

    output_directory: Path
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


def _local_checkpoint_identity(
    checkpoints_dir: Path | None,
    resolved_model: str,
) -> dict[str, Any] | None:
    """Bind a local export to its complete ARDY checkpoint payload without paths."""

    if checkpoints_dir is None:
        return None
    checkpoint_dir = checkpoints_dir / resolved_model
    files: dict[str, dict[str, Any]] = {}
    for filename in _LOCAL_CHECKPOINT_FILES:
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"ARDY checkpoint payload is missing required file: {path}"
            )
        files[filename] = _file_record(path)
    unsigned = {
        "format": "ardy-local-checkpoint-files",
        "format_version": 1,
        "files": files,
    }
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **unsigned,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _model_revision(
    *,
    resolved_model: str,
    minilm_artifact_fingerprint: Any,
    checkpoint_identity: dict[str, Any] | None,
) -> str:
    """Derive an immutable revision from both sets of exported weights."""

    if not isinstance(minilm_artifact_fingerprint, str):
        raise TypeError("MiniLM artifact fingerprint must be a string.")
    checkpoint_fingerprint = (
        checkpoint_identity.get("fingerprint")
        if isinstance(checkpoint_identity, dict)
        else None
    )
    for label, fingerprint in (
        ("MiniLM artifact", minilm_artifact_fingerprint),
        ("ARDY checkpoint", checkpoint_fingerprint),
    ):
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(f"{label} fingerprint must be a lowercase SHA-256.")

    identity = {
        "format": "ardy-browser-model-identity",
        "schema_version": 1,
        "ardy_model": resolved_model,
        "ardy_checkpoint_fingerprint": checkpoint_fingerprint,
        "minilm_artifact_fingerprint": minilm_artifact_fingerprint,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if config.checkpoints_dir is None:
        raise ValueError(
            "checkpoints_dir is required to derive an immutable ARDY checkpoint revision."
        )
    if not config.checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"ARDY checkpoints directory not found: {config.checkpoints_dir}"
        )

    output_directory = config.output_directory
    if output_directory.is_symlink() or (
        output_directory.exists() and not output_directory.is_dir()
    ):
        raise NotADirectoryError(
            f"Browser model output must be a directory: {output_directory}"
        )

    protected_directories = [
        config.minilm_artifact.resolve(),
        config.checkpoints_dir.resolve(),
    ]
    resolved_destination = output_directory.resolve()
    if resolved_destination.parent == resolved_destination:
        raise ValueError("Browser model output cannot replace a filesystem root.")
    for protected in protected_directories:
        if (
            resolved_destination == protected
            or resolved_destination in protected.parents
            or protected in resolved_destination.parents
        ):
            raise ValueError(
                "Browser model output must not overlap model input "
                f"directories: {output_directory}"
            )


def _specialize_denoiser_position_tables(
    denoiser: torch.nn.Module,
    *,
    num_timesteps: int,
    max_motion_tokens: int,
) -> None:
    """Trim non-learned lookup tables to the browser runtime's reachable indices."""
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive.")
    if max_motion_tokens <= 0:
        raise ValueError("max_motion_tokens must be positive.")

    for block_name in ("root_model", "body_model"):
        block = getattr(denoiser, block_name)
        if block.positional_encoding_mode != "learned_prefix_zero_at_first_generation":
            raise ValueError(
                "Browser export requires learned_prefix_zero_at_first_generation "
                f"positional encoding, got {block.positional_encoding_mode!r} for {block_name}."
            )

        timestep_table = block.sequence_pos_encoder.pe
        if block.embed_timestep.sequence_pos_encoder is not block.sequence_pos_encoder:
            raise ValueError(f"Timestep positional encoder is not shared for {block_name}.")
        if timestep_table.ndim != 3 or timestep_table.shape[0] != 1:
            raise ValueError(f"Unexpected timestep positional table shape for {block_name}: {timestep_table.shape}")
        if timestep_table.shape[1] < num_timesteps:
            raise ValueError(
                f"Timestep positional table for {block_name} has only {timestep_table.shape[1]} rows; "
                f"{num_timesteps} are required."
            )
        # TimestepEmbedder holds the same PositionalEncoding module, so replacing
        # its non-persistent buffer specializes both references.
        block.sequence_pos_encoder.pe = timestep_table[:, :num_timesteps].clone()

        motion_encoder = block.motion_token_embedding
        motion_table = motion_encoder.pe
        if motion_table.ndim != 2:
            raise ValueError(f"Unexpected motion positional table shape for {block_name}: {motion_table.shape}")
        if motion_encoder.max_len < max_motion_tokens:
            raise ValueError(
                f"Motion positional table for {block_name} supports only "
                f"|index| < {motion_encoder.max_len}; |index| < {max_motion_tokens} is required."
            )

        # PositionalEncodingNegativeIndex stores [0..N-1, -(N-1)..-1].
        # Preserve the exact original values while retaining every index the
        # fixed browser window can reach: -history_tokens through max_tokens-1.
        positive = motion_table[:max_motion_tokens]
        negative = motion_table[-(max_motion_tokens - 1) :] if max_motion_tokens > 1 else motion_table[:0]
        motion_encoder.pe = torch.cat((positive, negative), dim=0).clone()
        motion_encoder.max_len = max_motion_tokens


def _write_deterministic_gzip(
    source: Path,
    destination: Path,
) -> dict[str, Any]:
    """Compress one regular file with reproducible gzip level 9 settings."""

    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"Browser model source is not a regular file: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with (
        source.open("rb") as input_file,
        destination.open("wb") as raw_output,
    ):
        compressor = zlib.compressobj(
            level=9,
            method=zlib.DEFLATED,
            wbits=31,
            memLevel=9,
            strategy=zlib.Z_DEFAULT_STRATEGY,
        )
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
            raw_output.write(compressor.compress(chunk))
        raw_output.write(compressor.flush())
    destination.chmod(0o644)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _write_model_files_directory(
    *,
    source_directory: Path,
    payload_paths: list[Path],
    manifest: dict[str, Any],
    output_directory: Path,
) -> Path:
    """Write a compressed manifest and one gzip transport per asset."""

    if manifest.get("format") != BROWSER_MODEL_FILES_FORMAT:
        raise ValueError(
            f"Browser model format must be {BROWSER_MODEL_FILES_FORMAT!r}."
        )
    if manifest.get("schema_version") != BROWSER_MODEL_FILES_SCHEMA_VERSION:
        raise ValueError(
            "Browser model schema_version must be "
            f"{BROWSER_MODEL_FILES_SCHEMA_VERSION}."
        )
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise TypeError("Browser model manifest requires a model object.")
    revision = model.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 64
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("Browser model revision must be a lowercase SHA-256.")

    if output_directory.exists():
        if not output_directory.is_dir() or output_directory.is_symlink():
            raise NotADirectoryError(
                f"Browser model output must be a directory: {output_directory}"
            )
        if any(output_directory.iterdir()):
            raise FileExistsError(
                f"Browser model staging directory is not empty: {output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)

    expected_records = manifest.get("files")
    if not isinstance(expected_records, dict):
        raise TypeError("Browser model manifest requires a files object.")

    relative_paths: dict[str, Path] = {}
    for source in payload_paths:
        try:
            relative_path = source.relative_to(source_directory).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Browser model source is outside its working directory: {source}"
            ) from error
        if relative_path in relative_paths:
            raise ValueError(f"Duplicate browser model asset: {relative_path}")
        relative_paths[relative_path] = source

    if set(relative_paths) != set(expected_records):
        raise ValueError(
            "Browser model assets do not match manifest files; "
            f"assets={sorted(relative_paths)}, records={sorted(expected_records)}."
        )

    finalized_manifest = copy.deepcopy(manifest)
    for relative_path, source in sorted(relative_paths.items()):
        transport_path = f"{relative_path}.gz"
        compressed_path = output_directory / transport_path
        raw_record = _write_deterministic_gzip(source, compressed_path)
        if raw_record != expected_records[relative_path]:
            raise RuntimeError(
                f"Browser model asset changed after manifest creation: {relative_path}"
            )
        finalized_manifest["files"][relative_path] = {
            **raw_record,
            "transport": {
                "path": transport_path,
                "compression": "gzip",
                **_file_record(compressed_path),
            },
        }

    raw_manifest_path = output_directory / "model.json"
    raw_manifest_path.write_text(
        json.dumps(
            finalized_manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    compressed_manifest_path = output_directory / "model.json.gz"
    _write_deterministic_gzip(
        raw_manifest_path,
        compressed_manifest_path,
    )
    raw_manifest_path.unlink()
    return compressed_manifest_path


def _unused_sibling_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _publish_directory_set(
    staged_directories: list[tuple[Path, Path]],
) -> None:
    """Publish prepared directories together, rolling back ordinary failures."""

    if not staged_directories:
        raise ValueError("At least one staged directory is required.")
    destinations = [
        destination.resolve() for _, destination in staged_directories
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Directory publication destinations must be distinct.")
    for staged, destination in staged_directories:
        if not staged.is_dir() or staged.is_symlink():
            raise FileNotFoundError(
                f"Staged browser model directory is missing: {staged}"
            )
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise NotADirectoryError(
                f"Browser model destination is not a directory: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

    backups: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for _, destination in staged_directories:
            if destination.exists() or destination.is_symlink():
                backup = _unused_sibling_path(destination, ".rollback")
                os.replace(destination, backup)
                backups[destination] = backup
        for staged, destination in staged_directories:
            os.replace(staged, destination)
            published.append((staged, destination))
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        for staged, destination in reversed(published):
            try:
                if destination.exists() or destination.is_symlink():
                    os.replace(destination, staged)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for destination, backup in reversed(tuple(backups.items())):
            try:
                if backup.exists() or backup.is_symlink():
                    os.replace(backup, destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "Browser model publication failed and rollback was incomplete: "
                f"{rollback_errors}"
            ) from error
        raise
    else:
        for backup in backups.values():
            shutil.rmtree(backup)


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
    text_conditions: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    nfpt = int(model.num_frames_per_token)
    max_frames = max_tokens * nfpt
    generation_frames = int(model.gen_horizon_len)
    generation_tokens = generation_frames // nfpt
    history_frames = 0

    history_mask = torch.zeros(1, max_frames, device=device)
    generation_mask = torch.zeros(1, max_frames, device=device)
    generation_mask[:, :generation_frames] = 1
    history_token_mask = torch.zeros(1, max_tokens, device=device)
    generation_token_mask = torch.zeros(1, max_tokens, device=device)
    generation_token_mask[:, :generation_tokens] = 1

    generator = torch.Generator(device=device)
    generator.manual_seed(20260728)
    sample = torch.zeros(
        1,
        max_tokens,
        model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim,
        device=device,
    )
    sample[:, :generation_tokens] = torch.randn(
        1,
        generation_tokens,
        sample.shape[-1],
        generator=generator,
        device=device,
    )
    return (
        torch.tensor([2.0], dtype=torch.float32, device=device),
        sample,
        torch.tensor([history_frames], dtype=torch.int64, device=device),
        torch.tensor([generation_frames], dtype=torch.int64, device=device),
        history_mask,
        generation_mask,
        history_token_mask,
        generation_token_mask,
        text_conditions.detach().to(device=device, dtype=torch.float32),
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
    denoiser: torch.nn.Module,
    denoiser_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    nfpt = int(model.num_frames_per_token)
    max_frames = max_tokens * nfpt
    generation_tokens = int(model.gen_horizon_len) // nfpt
    hybrid = denoiser_inputs[1].clone()
    with torch.no_grad():
        for timestep in range(int(model.diffusion.num_base_steps) - 1, -1, -1):
            step_inputs = list(denoiser_inputs)
            step_inputs[1] = hybrid
            step_inputs[9] = torch.tensor(
                [timestep],
                dtype=torch.int64,
                device=device,
            )
            prediction = denoiser(*step_inputs)
            current_generation = hybrid[:, :generation_tokens]
            predicted_generation = prediction[:, :generation_tokens]
            next_generation = model.sampler(
                current_generation,
                predicted_generation,
                step_inputs[9],
            )
            hybrid = hybrid.clone()
            hybrid[:, :generation_tokens] = next_generation

    # Browser generation requantizes latent history before decoding.  Keeping
    # the verification input on the FSQ grid also avoids meaningless
    # PyTorch-vs-ONNX differences at a rounding boundary.
    root_tokens = hybrid[:, :, : model.denoiser.nframe_root_dim]
    latent_tokens = model.autoencoder.requantize(hybrid[:, :, model.denoiser.nframe_root_dim :])
    hybrid = torch.cat((root_tokens, latent_tokens), dim=-1)
    motion_pad_mask = torch.zeros(
        1,
        max_frames,
        dtype=torch.float32,
        device=device,
    )
    motion_pad_mask[:, : int(model.gen_horizon_len)] = 1
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
            "ONNX validation requires the `onnx` dependency. Run the exporter with `uv run --with onnx ...`."
        ) from error
    for path in paths:
        onnx.checker.check_model(str(path))


def _run_ort(
    path: Path,
    inputs: dict[str, torch.Tensor],
    expected_outputs: tuple[str, ...],
    *,
    disable_graph_optimizations: bool = False,
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
    if disable_graph_optimizations:
        # CPU graph fusions such as com.microsoft.Gelu do not provide FP16
        # kernels. Disabling optimizer-only fusions validates the portable
        # ONNX primitive graph that WebGPU consumes.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = min(8, max(1, torch.get_num_threads()))
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual_input_names = {item.name for item in session.get_inputs()}
    expected_input_names = set(inputs)
    if actual_input_names != expected_input_names:
        raise RuntimeError(
            f"ONNX input contract mismatch for {path.name}: expected "
            f"{sorted(expected_input_names)}, got {sorted(actual_input_names)}"
        )
    actual_output_names = tuple(item.name for item in session.get_outputs())
    if actual_output_names != expected_outputs:
        raise RuntimeError(
            f"ONNX output contract mismatch for {path.name}: expected {expected_outputs}, got {actual_output_names}"
        )
    feeds = {name: value.detach().to(device="cpu").numpy() for name, value in inputs.items()}
    values = session.run(None, feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), values)}


def _max_abs(reference: torch.Tensor, actual: np.ndarray) -> float:
    expected = reference.detach().to(device="cpu", dtype=torch.float32).numpy()
    return float(np.max(np.abs(expected - actual)))


def _verify_mixed_numeric(
    reference_paths: dict[str, Path],
    mixed_paths: dict[str, Path],
    dummy_inputs: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, Any]:
    """Compare mixed graphs with the already verified FP32 ONNX exports."""
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
        "decoder": (
            "normalized_motion",
            "posed_joints",
            "local_rotations",
            "global_rotations",
            "root_positions",
            "foot_contacts",
            "global_root_heading",
        ),
    }
    results: dict[str, Any] = {
        "backend": "onnxruntime-cpu-optimizations-disabled",
        "reference": "fp32-onnx-export",
        "limits": _MIXED_PRECISION_LIMITS,
        "outputs": {},
    }

    for graph_name in ("text_encoder", "denoiser", "decoder"):
        feeds = dict(zip(input_names[graph_name], dummy_inputs[graph_name]))
        reference = _run_ort(
            reference_paths[graph_name],
            feeds,
            output_names[graph_name],
            disable_graph_optimizations=True,
        )
        mixed = _run_ort(
            mixed_paths[graph_name],
            feeds,
            output_names[graph_name],
            disable_graph_optimizations=True,
        )
        graph_results: dict[str, Any] = {}
        failures: dict[str, Any] = {}
        for output_name in output_names[graph_name]:
            expected = reference[output_name]
            actual = mixed[output_name]
            limits = _MIXED_PRECISION_LIMITS[graph_name][output_name]
            if expected.dtype == np.bool_:
                agreement = float(np.mean(expected == actual))
                metrics = {"agreement": agreement}
                if agreement < limits["minimum_agreement"]:
                    failures[output_name] = {
                        "minimum_agreement": limits["minimum_agreement"],
                        "actual": agreement,
                    }
            else:
                expected_float = expected.astype(np.float32, copy=False)
                actual_float = actual.astype(np.float32, copy=False)
                if not np.isfinite(actual_float).all():
                    raise RuntimeError(
                        f"Mixed-FP16 numeric verification produced non-finite {graph_name}.{output_name}."
                    )
                difference = actual_float - expected_float
                rmse = float(np.sqrt(np.mean(np.square(difference))))
                maximum = float(np.max(np.abs(difference)))
                denominator = float(
                    np.linalg.norm(expected_float.reshape(-1)) * np.linalg.norm(actual_float.reshape(-1))
                )
                cosine = (
                    1.0
                    if denominator == 0 and np.array_equal(expected_float, actual_float)
                    else 0.0
                    if denominator == 0
                    else float(
                        np.clip(
                            np.sum(expected_float * actual_float) / denominator,
                            -1.0,
                            1.0,
                        )
                    )
                )
                metrics = {
                    "max_abs_error": maximum,
                    "rmse": rmse,
                    "cosine_similarity": cosine,
                }
                if limits.get("require_exact") and not np.array_equal(expected, actual):
                    failures.setdefault(output_name, {})["exact_equality"] = {
                        "required": True,
                        "max_abs_error": maximum,
                    }
                if rmse > limits["rmse"]:
                    failures.setdefault(output_name, {})["rmse"] = {
                        "limit": limits["rmse"],
                        "actual": rmse,
                    }
                minimum_cosine = limits.get("minimum_cosine_similarity")
                if minimum_cosine is not None and cosine < minimum_cosine:
                    failures.setdefault(output_name, {})["cosine_similarity"] = {
                        "minimum": minimum_cosine,
                        "actual": cosine,
                    }
            graph_results[output_name] = metrics
        if failures:
            raise RuntimeError(f"Mixed-FP16 numeric verification failed for {graph_name}: {failures}")
        results["outputs"][graph_name] = graph_results
    return results


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
        "decoder": (
            "normalized_motion",
            "posed_joints",
            "local_rotations",
            "global_rotations",
            "root_positions",
            "foot_contacts",
            "global_root_heading",
        ),
    }

    results: dict[str, Any] = {
        "backend": "onnxruntime-cpu",
        "reference": "pytorch-fp32-mha-fastpath-disabled",
        "max_abs_error": {},
        "max_abs_error_limit": _NUMERIC_ERROR_LIMITS,
    }
    for graph_name in (
        "text_encoder",
        "denoiser",
        "decoder",
    ):
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
            output_names[graph_name],
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
                "localRotations": "local_rotations",
                "globalRotations": "global_rotations",
                "rootPositions": "root_positions",
                "footContacts": "foot_contacts",
                "globalRootHeading": "global_root_heading",
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
    precision_reports: dict[str, MixedPrecisionReport],
    checkpoint_identity: dict[str, Any] | None,
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
    source_onnx_bytes = sum(report.source_size_bytes for report in precision_reports.values())
    mixed_onnx_bytes = sum(report.output_size_bytes for report in precision_reports.values())
    artifact_fingerprint = artifact_config.get("artifact_fingerprint")
    revision = _model_revision(
        resolved_model=resolved_model,
        minilm_artifact_fingerprint=artifact_fingerprint,
        checkpoint_identity=checkpoint_identity,
    )

    manifest: dict[str, Any] = {
        "format": BROWSER_MODEL_FILES_FORMAT,
        "schema_version": BROWSER_MODEL_FILES_SCHEMA_VERSION,
        "model": {
            "id": config.model_id,
            "revision": revision,
            "variant": "MiniLM Core40 interactive",
            "ardy_model": resolved_model,
            "minilm_artifact_fingerprint": artifact_fingerprint,
            **(
                {"ardy_checkpoint": checkpoint_identity}
                if checkpoint_identity is not None
                else {}
            ),
        },
        "files": files,
        "tokenizer": {
            "directory": "tokenizer",
            "max_length": config.max_prompt_tokens,
            "model_id": artifact_config["base_model"],
        },
        "graphs": _graph_contracts(),
        "precision": {
            "format": "mixed-fp16",
            "policy_version": MIXED_FP16_POLICY_VERSION,
            "public_io_dtype": "float32",
            "required_webgpu_features": ["shader-f16"],
            "toolchain": {
                dependency: metadata.version(dependency)
                for dependency in ("onnx", "onnxruntime", "torch")
            },
            "source_onnx_bytes": source_onnx_bytes,
            "mixed_onnx_bytes": mixed_onnx_bytes,
            "saved_onnx_bytes": source_onnx_bytes - mixed_onnx_bytes,
            "saved_onnx_fraction": 1 - mixed_onnx_bytes / source_onnx_bytes,
            "graphs": {graph_name: report.to_dict() for graph_name, report in sorted(precision_reports.items())},
        },
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
        "capabilities": {
            "text_conditioning": True,
            "initial_root_transform": True,
            "detailed_motion_outputs": True,
            "motion_correction": False,
        },
        "runtime": {
            "onnx_opset": config.opset,
            "batch_size": 1,
            "contract_revision": 3,
            "required_webgpu_features": ["shader-f16"],
            "text_only": True,
            "detailed_motion_outputs": True,
            "motion_correction_included": False,
            "global_translation_y_must_be_zero": False,
        },
        "notices": [
            "Licensed by NVIDIA Corporation under the NVIDIA Open Model License.",
            ("The specialized text encoder is based on sentence-transformers/all-MiniLM-L6-v2."),
            (
                "Review THIRD_PARTY_MODELS_AND_DATA.md and all source "
                "model/data terms before distributing these model files."
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
                    "terms before distributing these model files."
                ),
            },
        ],
    }
    if verification is not None:
        manifest["verification"] = verification
    return manifest


def _build_fp32_payload(
    *,
    candidate_manifest: dict[str, Any],
    output_dir: Path,
    graph_paths: dict[str, Path],
    tokenizer_paths: list[Path],
    verification: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[Path]]:
    """Build the browser-loadable FP32 variant from the exported graphs."""
    expected_graphs = set(candidate_manifest["graphs"])
    if set(graph_paths) != expected_graphs:
        raise ValueError(
            "FP32 reference graphs must match the candidate graph contract; "
            f"expected {sorted(expected_graphs)}, got {sorted(graph_paths)}."
        )

    fp32_tokenizer_dir = output_dir / "tokenizer"
    fp32_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    fp32_tokenizer_paths: list[Path] = []
    for source in tokenizer_paths:
        destination = fp32_tokenizer_dir / source.name
        shutil.copy2(source, destination)
        fp32_tokenizer_paths.append(destination)

    payload_paths = list(graph_paths.values()) + fp32_tokenizer_paths
    manifest = copy.deepcopy(candidate_manifest)
    manifest["files"] = {path.relative_to(output_dir).as_posix(): _file_record(path) for path in sorted(payload_paths)}

    graph_precision: dict[str, Any] = {}
    for graph_name, graph_path in sorted(graph_paths.items()):
        expected_model_path = candidate_manifest["graphs"][graph_name]["model"]
        relative_path = graph_path.relative_to(output_dir).as_posix()
        if relative_path != expected_model_path:
            raise ValueError(
                f"FP32 reference graph {graph_name!r} must be stored as {expected_model_path!r}, got {relative_path!r}."
            )
        graph_precision[graph_name] = {
            "model": relative_path,
            **_file_record(graph_path),
        }

    candidate_precision = candidate_manifest["precision"]
    fp32_onnx_bytes = sum(record["size_bytes"] for record in graph_precision.values())
    manifest["precision"] = {
        "format": "fp32",
        "public_io_dtype": "float32",
        "required_webgpu_features": [],
        "toolchain": copy.deepcopy(candidate_precision["toolchain"]),
        "onnx_bytes": fp32_onnx_bytes,
        "graphs": graph_precision,
    }

    manifest["runtime"]["required_webgpu_features"] = []
    if verification is None:
        manifest.pop("verification", None)
    else:
        manifest["verification"] = {"fp32_export": verification}

    return manifest, payload_paths


def _export_browser_model_files_working_directory(
    config: BrowserExportConfig,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    list[Path],
    dict[str, Any],
    list[Path],
]:
    """Export and validate the three browser graphs in a temporary directory."""
    device = _resolve_device(config.device)

    resolved_model = resolve_model_name(
        config.model,
        checkpoints_dir=(str(config.checkpoints_dir) if config.checkpoints_dir is not None else None),
    )
    checkpoint_identity = _local_checkpoint_identity(
        config.checkpoints_dir,
        resolved_model,
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
            "The Core40 browser contract requires exactly one 40-frame history "
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

    _specialize_denoiser_position_tables(
        model.denoiser.model,
        num_timesteps=int(model.diffusion.num_base_steps),
        max_motion_tokens=config.max_tokens,
    )

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
    text_dummy = _make_text_dummy(
        tokenizer,
        config.max_prompt_tokens,
        device,
    )
    with torch.no_grad():
        text_conditions = modules["text_encoder"](*text_dummy)
    denoiser_dummy = _make_denoiser_dummy(
        model,
        config.max_tokens,
        text_conditions,
        device,
    )
    dummy_inputs = {
        "text_encoder": text_dummy,
        "denoiser": denoiser_dummy,
        "decoder": _make_decoder_dummy(
            model,
            config.max_tokens,
            device,
            modules["denoiser"],
            denoiser_dummy,
        ),
    }

    graph_paths = {
        "text_encoder": output_dir / "text_encoder.onnx",
        "denoiser": output_dir / "denoiser.onnx",
        "decoder": output_dir / "decoder.onnx",
    }
    fp32_dir = output_dir / ".fp32"
    fp32_dir.mkdir()
    fp32_graph_paths = {graph_name: fp32_dir / graph_path.name for graph_name, graph_path in graph_paths.items()}
    _export_graph(
        modules["text_encoder"],
        dummy_inputs["text_encoder"],
        fp32_graph_paths["text_encoder"],
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
        fp32_graph_paths["denoiser"],
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
        fp32_graph_paths["decoder"],
        input_names=[
            "hybrid_tokens",
            "motion_pad_mask",
            "global_translation",
        ],
        output_names=[
            "normalized_motion",
            "posed_joints",
            "local_rotations",
            "global_rotations",
            "root_positions",
            "foot_contacts",
            "global_root_heading",
        ],
        opset=config.opset,
    )

    _check_onnx_models(list(fp32_graph_paths.values()))
    fp32_verification = _verify_numeric(fp32_graph_paths, modules, dummy_inputs) if config.verify else None
    precision_reports = {
        graph_name: convert_browser_onnx_to_mixed_fp16(
            fp32_graph_paths[graph_name],
            graph_paths[graph_name],
            graph_name=graph_name,
        )
        for graph_name in ("text_encoder", "denoiser", "decoder")
    }
    _check_onnx_models(list(graph_paths.values()))
    mixed_verification = (
        _verify_mixed_numeric(
            fp32_graph_paths,
            graph_paths,
            dummy_inputs,
        )
        if config.verify
        else None
    )
    verification = (
        {
            "fp32_export": fp32_verification,
            "mixed_precision": mixed_verification,
        }
        if config.verify
        else None
    )

    tokenizer_paths = _copy_tokenizer_files(
        config.minilm_artifact,
        output_dir,
    )
    final_checkpoint_identity = _local_checkpoint_identity(
        config.checkpoints_dir,
        resolved_model,
    )
    if final_checkpoint_identity != checkpoint_identity:
        raise RuntimeError(
            "ARDY checkpoint files changed while the browser models were exported."
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
        precision_reports=precision_reports,
        checkpoint_identity=checkpoint_identity,
    )
    fp32_manifest, fp32_payload_paths = _build_fp32_payload(
        candidate_manifest=manifest,
        output_dir=fp32_dir,
        graph_paths=fp32_graph_paths,
        tokenizer_paths=tokenizer_paths,
        verification=fp32_verification,
    )
    return (
        manifest,
        payload_paths,
        fp32_manifest,
        fp32_payload_paths,
    )


def _stage_directory(destination: Path, label: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=f".{label}.stage",
            dir=destination.parent,
        )
    )


def export_browser_model_files(config: BrowserExportConfig) -> Path:
    """Export reproducible FP16 and FP32 Core40 browser model variants."""

    _validate_config(config)
    with tempfile.TemporaryDirectory(prefix="ardy-browser-export-") as temporary_directory:
        working_directory = Path(temporary_directory)
        (
            manifest,
            payload_paths,
            fp32_manifest,
            fp32_payload_paths,
        ) = _export_browser_model_files_working_directory(
            config,
            working_directory,
        )
        family_stage = _stage_directory(
            config.output_directory,
            "family",
        )
        try:
            fp16_stage = family_stage / FP16_VARIANT_DIRECTORY
            _write_model_files_directory(
                source_directory=working_directory,
                payload_paths=payload_paths,
                manifest=manifest,
                output_directory=fp16_stage,
            )
            fp32_stage = family_stage / FP32_VARIANT_DIRECTORY
            _write_model_files_directory(
                source_directory=working_directory / ".fp32",
                payload_paths=fp32_payload_paths,
                manifest=fp32_manifest,
                output_directory=fp32_stage,
            )
            _publish_directory_set(
                [(family_stage, config.output_directory)]
            )
        finally:
            if family_stage.exists():
                shutil.rmtree(family_stage)
    return config.output_directory


__all__ = [
    "BROWSER_MODEL_FILES_FORMAT",
    "BROWSER_MODEL_FILES_SCHEMA_VERSION",
    "FP16_VARIANT_DIRECTORY",
    "FP32_VARIANT_DIRECTORY",
    "BrowserExportConfig",
    "export_browser_model_files",
]
