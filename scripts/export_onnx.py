#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export denoiser and autoencoder decoder as ONNX models, then build TRT engines.

Mirrors the ONNX-TRT path of the interactive demo (see
``scripts/interactive_demo/loading.py``): it loads a released model with
``load_model``, exports the CFG+denoiser wrapper and the autoencoder decoder to
ONNX, and builds the TRT engines the demo later loads.

Usage:
    python scripts/export_onnx.py [--model NAME] [--checkpoints_dir DIR]
                                  [--output_dir DIR] [--num_text_tokens N]
                                  [--max_tokens N] [--opset N]
                                  [--skip_trt] [--skip_verify]
"""

import argparse
import contextlib
import logging
import os
import threading
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Released model to export by default; a short key ("core"/"g1"/"soma") or a
# full folder name, resolved by load_model / the model registry.
DEFAULT_MODEL = "core"


# ──────────────────────────────────────────────────────────────────────────────
# Dummy input generation
# ──────────────────────────────────────────────────────────────────────────────
def make_denoiser_dummy_inputs(
    wrapper,
    num_tokens=3,
    num_text_tokens=1,
    device="cuda:0",
):
    """Create dummy inputs matching the denoiser wrapper's forward() signature."""
    nfpt = wrapper.num_frames_per_token
    num_frames = num_tokens * nfpt
    dim_token = wrapper.nframe_root_dim + wrapper.latent_embedding_dim
    motion_rep_dim = wrapper.motion_rep.motion_rep_dim
    llm_dim = wrapper.llm_shape[-1]

    # 1 history token, rest generation, 0 future
    history_frames = nfpt
    gen_frames = num_frames - history_frames

    return {
        "cfg_weight_text": torch.tensor([3.5], device=device),
        "cfg_weight_cstr": torch.tensor([1.5], device=device),
        "x": torch.randn(1, num_tokens, dim_token, device=device),
        "history_len": torch.tensor([history_frames], device=device, dtype=torch.int64),
        "generation_len": torch.tensor([gen_frames], device=device, dtype=torch.int64),
        "future_len": torch.tensor([0], device=device, dtype=torch.int64),
        "history_mask": torch.cat(
            [
                torch.ones(1, history_frames, device=device),
                torch.zeros(1, gen_frames, device=device),
            ],
            dim=1,
        ),
        "generation_mask": torch.cat(
            [
                torch.zeros(1, history_frames, device=device),
                torch.ones(1, gen_frames, device=device),
            ],
            dim=1,
        ),
        "future_mask": torch.zeros(1, num_frames, device=device),
        "history_token_mask": torch.cat(
            [
                torch.ones(1, 1, device=device),
                torch.zeros(1, num_tokens - 1, device=device),
            ],
            dim=1,
        ),
        "generation_token_mask": torch.cat(
            [
                torch.zeros(1, 1, device=device),
                torch.ones(1, num_tokens - 1, device=device),
            ],
            dim=1,
        ),
        "future_token_mask": torch.zeros(1, num_tokens, device=device),
        "text_feat": torch.randn(1, num_text_tokens, llm_dim, device=device),
        "text_feat_pad_mask": torch.ones(1, num_text_tokens, device=device),
        "timesteps": torch.tensor([0], device=device, dtype=torch.int64),
        "first_heading_angle": torch.zeros(1, device=device),
        "motion_mask": torch.zeros(1, num_frames, motion_rep_dim, device=device),
        "observed_motion": torch.zeros(1, num_frames, motion_rep_dim, device=device),
    }


def make_decoder_dummy_inputs(autoencoder, num_tokens=3, device="cuda:0"):
    """Create dummy inputs for the autoencoder decoder export."""
    nfpt = autoencoder._num_frames_per_token
    num_frames = num_tokens * nfpt
    latent_dim = autoencoder._latent_embedding_dim
    # external_cond_dim: get from decoder's stored attribute
    ext_cond_dim = autoencoder.decoder._external_cond_dim

    return {
        "latent_tokens": torch.randn(1, num_tokens, latent_dim, device=device),
        "external_cond": torch.randn(1, num_frames, ext_cond_dim, device=device),
        "motion_pad_mask": torch.ones(1, num_frames, device=device),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for ONNX export compatibility
# ──────────────────────────────────────────────────────────────────────────────
# torch.onnx.export is not thread-safe: it toggles a process-global flag and is
# not reentrant, so concurrent exports (e.g. two viser clients connecting at
# once, each running load_model in the thread pool) must run one at a time.
_ONNX_EXPORT_LOCK = threading.Lock()


@contextlib.contextmanager
def _onnx_export_mode():
    """Set up global state for a single ONNX trace, serialized across threads.

    Disables the TransformerEncoderLayer fast path, and resets the ``GLOBALS.in_onnx_export`` flag
    in case a previous export died mid-way.

    Holds ``_ONNX_EXPORT_LOCK`` for the whole trace: ``torch.onnx.export`` toggles the process-
    global ``GLOBALS.in_onnx_export`` flag (which we also reset here), so two concurrent exports
    clobber each other and one fails its internal ``assert GLOBALS.in_onnx_export``. Serializing
    prevents that race.
    """
    with _ONNX_EXPORT_LOCK:
        # Try the newer PyTorch 2.3+ API
        if hasattr(torch.backends, "mha"):
            prev_fastpath = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            mha_backend = torch.backends.mha
        # Fallback to the older internal API (PyTorch < 2.3)
        elif hasattr(torch._C, "_dispatch_mha"):
            prev_fastpath = torch._C._dispatch_mha.get_fastpath_enabled()
            torch._C._dispatch_mha.set_fastpath_enabled(False)
            mha_backend = torch._C._dispatch_mha
        else:
            mha_backend = None

        # Reset the global flag in case a previous export failed mid-way
        # (safe now that exports are serialized by _ONNX_EXPORT_LOCK).
        try:
            from torch.onnx._internal.torchscript_exporter.utils import GLOBALS

            GLOBALS.in_onnx_export = False
        except (ImportError, AttributeError):
            pass

        try:
            yield
        finally:
            if mha_backend is not None:
                mha_backend.set_fastpath_enabled(prev_fastpath)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export
# ──────────────────────────────────────────────────────────────────────────────
def export_denoiser_onnx(
    denoiser,
    model_cfg,
    output_path,
    num_tokens=3,
    num_text_tokens=1,
    opset=17,
):
    """Export the CFG + denoiser wrapper to ONNX.

    The exported model takes B=1 inputs plus cfg_weights, internally does separated CFG (triples the
    batch), runs the denoiser, and returns the CFG-combined B=1 output.
    """
    device = next(denoiser.parameters()).device
    dummy = make_denoiser_dummy_inputs(denoiser, num_tokens, num_text_tokens, device)

    # The CFG model takes cfg_weight_text / cfg_weight_cstr as separate tensor
    # inputs, so it traces directly to ONNX with two weight graph inputs.
    export_module = denoiser
    export_module.eval()

    input_names = list(dummy.keys())
    output_names = ["output"]

    # Only token-count dims are dynamic (batch is always 1)
    dynamic_axes = {}
    for name in input_names:
        t = dummy[name]
        if t.ndim >= 2:
            if "token" in name or name == "x":
                dynamic_axes[name] = {1: "num_tokens"}
            elif "text" in name:
                pass  # text is always cropped to model's internal max
            elif name in (
                "history_mask",
                "generation_mask",
                "future_mask",
                "motion_mask",
                "observed_motion",
            ):
                dynamic_axes[name] = {1: "num_frames"}
    dynamic_axes["output"] = {1: "num_tokens"}

    log.info("Exporting CFG+denoiser ONNX to %s ...", output_path)
    args = tuple(dummy[k] for k in input_names)

    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            export_module,
            args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    log.info("CFG+denoiser ONNX exported: %s", output_path)


def export_decoder_onnx(autoencoder, output_path, num_tokens=3, opset=17):
    """Export the autoencoder decoder to ONNX."""
    device = next(autoencoder.parameters()).device
    dummy = make_decoder_dummy_inputs(autoencoder, num_tokens, device)

    export_module = autoencoder
    export_module.eval()

    input_names = list(dummy.keys())
    output_names = ["output"]

    dynamic_axes = {
        "latent_tokens": {0: "batch", 1: "num_tokens"},
        "external_cond": {0: "batch", 1: "num_frames"},
        "motion_pad_mask": {0: "batch", 1: "num_frames"},
        "output": {0: "batch", 1: "num_frames"},
    }

    log.info("Exporting decoder ONNX to %s ...", output_path)
    args = tuple(dummy[k] for k in input_names)

    with _onnx_export_mode(), torch.no_grad():
        torch.onnx.export(
            export_module,
            args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    log.info("Decoder ONNX exported: %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# TRT engine building
# ──────────────────────────────────────────────────────────────────────────────
def build_trt_engine(
    onnx_path,
    engine_path,
    max_tokens=64,
    fp16=True,
):
    """Build a TRT engine from an ONNX model.

    Args:
        max_tokens: number of motion tokens. The engine is built for this fixed
            size (min == opt == max); inputs are padded to it at runtime.
        fp16: build with FP16 kernels (faster/smaller). Set False for an FP32
            engine that matches PyTorch more closely (useful to check whether a
            PyTorch-vs-TRT difference is just precision).
    """
    t0 = time.time()

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    log.info("Building TRT engine (%s): %s", "FP16" if fp16 else "FP32", engine_path)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Attention reshapes are fixed to the trace-time (max) token count.
    # All inputs are padded to this size at runtime, so the profile is
    # fixed to the max values.
    nfpt = 4  # num_frames_per_token
    fixed_token_sizes = (max_tokens, max_tokens, max_tokens)
    batch_sizes = (1, 1, 1)
    frame_sizes = tuple(t * nfpt for t in fixed_token_sizes)

    # Map input names to (min, opt, max) per dynamic dim.
    # Only list entries for dims that are actually -1 in the ONNX graph.
    dim_map = {
        "x": (fixed_token_sizes,),
        "history_mask": (frame_sizes,),
        "generation_mask": (frame_sizes,),
        "future_mask": (frame_sizes,),
        "history_token_mask": (fixed_token_sizes,),
        "generation_token_mask": (fixed_token_sizes,),
        "future_token_mask": (fixed_token_sizes,),
        "motion_mask": (frame_sizes,),
        "observed_motion": (frame_sizes,),
        "latent_tokens": (batch_sizes, fixed_token_sizes),
        "external_cond": (batch_sizes, frame_sizes),
        "motion_pad_mask": (batch_sizes, frame_sizes),
    }

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        ndynamic = sum(1 for s in shape if s == -1)
        if ndynamic == 0:
            continue  # skip static inputs

        sizes_for_dims = dim_map.get(inp.name)
        if sizes_for_dims is None:
            # Fallback: use frame_sizes for unknown dynamic dims
            sizes_for_dims = tuple(frame_sizes for _ in range(ndynamic))

        dyn_idx = 0
        min_s, opt_s, max_s = list(shape), list(shape), list(shape)
        for j, s in enumerate(shape):
            if s == -1:
                min_s[j] = sizes_for_dims[dyn_idx][0]
                opt_s[j] = sizes_for_dims[dyn_idx][1]
                max_s[j] = sizes_for_dims[dyn_idx][2]
                dyn_idx += 1
        profile.set_shape(inp.name, tuple(min_s), tuple(opt_s), tuple(max_s))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    log.info("TRT engine built in %.1fs: %s", time.time() - t0, engine_path)


def engine_path(output_dir, kind, max_tokens, fp16=True):
    """Path of a built TRT engine — the single source of truth for engine names.

    ``kind`` is ``"denoiser"`` or ``"decoder"``. Token size and precision are
    encoded in the filename so FP16 and FP32 engines (and different token sizes)
    can coexist in one directory and be loaded deliberately, e.g.
    ``denoiser_tok64_fp16.trt`` / ``decoder_tok64_fp32.trt``.

    NOTE: engine plan files only deserialize with the exact TensorRT version
    that built them, which is why pyproject.toml pins tensorrt-cu12. If that
    pin is ever bumped, cached ``engines/`` dirs must be cleared (or this name
    must start embedding the TRT version) or loading will fail.
    """
    precision = "fp16" if fp16 else "fp32"
    return os.path.join(output_dir, f"{kind}_tok{max_tokens}_{precision}.trt")


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────
def verify_denoiser(denoiser, engine_path, num_frames_per_token, num_tokens=3, num_text_tokens=1):
    """Compare the PyTorch CFG+denoiser against its TRT engine.

    ``denoiser`` is ``model.denoiser`` — the separated-CFG wrapper that was exported (it takes the
    two cfg-weight tensors as inputs and does the CFG batching internally), so it is used directly
    as the PyTorch reference and as the source of attribute/shape info for ``TRTCFGDenoiser``.
    """
    from ardy.model.trt import TRTCFGDenoiser

    device = next(denoiser.parameters()).device
    dummy = make_denoiser_dummy_inputs(denoiser, num_tokens, num_text_tokens, device)
    # forward() takes exactly these inputs, in this order (dummy is ordered to
    # match); pass positionally since TRTCFGDenoiser names the motion arg
    # ``token_seq_t`` while the wrapper names it ``x``.
    args = tuple(dummy.values())

    with torch.no_grad():
        ref = denoiser(*args)

    trt_cfg = TRTCFGDenoiser(
        engine_path,
        denoiser,
        num_frames_per_token=num_frames_per_token,
        num_tokens=num_tokens,
        num_text_tokens=num_text_tokens,
    )
    with torch.no_grad():
        trt_out = trt_cfg(*args)

    abs_diff = (ref - trt_out).abs()
    max_diff = abs_diff.max().item()
    # mean alongside max: a large max with a small mean points at a few FP16
    # outliers rather than a broadly wrong engine.
    log.info("Denoiser abs diff — max: %.6f, mean: %.6f", max_diff, abs_diff.mean().item())
    if max_diff > 0.1:
        log.warning("Denoiser output differs significantly!")
    return max_diff


def verify_decoder(autoencoder, engine_path, num_frames_per_token, num_tokens=3):
    """Compare the PyTorch autoencoder decoder against its TRT engine.

    The exported module is ``autoencoder`` itself (its ``forward`` is ``detokenize``), which returns
    a dict of motion features (e.g. ``root`` and ``body``). ONNX emits one graph output per dict
    entry, so the engine has one output per entry; each is compared against the matching PyTorch
    tensor, in order. The dummy inputs are already built at the fixed (max) trace size, so no
    padding is needed and each engine output shape equals its reference's.
    """
    from ardy.model.trt import TRTEngine

    device = next(autoencoder.parameters()).device
    dummy = make_decoder_dummy_inputs(autoencoder, num_tokens, device)

    # detokenize re-quantizes its latent input (round onto the FSQ grid). Random
    # latents sit near rounding boundaries where FP16 (engine) and FP32 (PyTorch)
    # pick different grid points, which explodes the diff and is not
    # representative — real FSQ tokens land exactly on the grid. Snap the dummy
    # latents to the grid first so the comparison reflects the decoder network.
    if getattr(autoencoder, "encode_with_quantization", False):
        with torch.no_grad():
            dummy["latent_tokens"] = autoencoder.requantize(dummy["latent_tokens"])

    with torch.no_grad():
        ref_dict = autoencoder(
            dummy["latent_tokens"],
            dummy["external_cond"],
            dummy["motion_pad_mask"] > 0.5,
        )
    # dict insertion order matches the ONNX graph output order.
    ref_names = list(ref_dict)
    ref_tensors = [ref_dict[k] for k in ref_names]

    engine = TRTEngine(engine_path)
    if len(engine.output_names) != len(ref_tensors):
        raise RuntimeError(
            f"Engine has {len(engine.output_names)} outputs "
            f"({engine.output_names}) but detokenize returned {len(ref_tensors)} "
            f"tensors ({ref_names})."
        )
    output_shapes = {name: tuple(ref.shape) for name, ref in zip(engine.output_names, ref_tensors)}
    results = engine.infer(
        output_shapes,
        latent_tokens=dummy["latent_tokens"].float(),
        external_cond=dummy["external_cond"].float(),
        motion_pad_mask=dummy["motion_pad_mask"].float(),
    )

    max_diff = 0.0
    for out_name, ref_name, ref in zip(engine.output_names, ref_names, ref_tensors):
        abs_diff = (ref - results[out_name]).abs()
        log.info(
            "Decoder '%s' (%s) abs diff — max: %.6f, mean: %.6f",
            ref_name,
            out_name,
            abs_diff.max().item(),
            abs_diff.mean().item(),
        )
        max_diff = max(max_diff, abs_diff.max().item())
    log.info("Decoder max abs diff: %.6f", max_diff)
    if max_diff > 0.1:
        log.warning("Decoder output differs significantly!")
    return max_diff


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def resolve_model_dir(model_name, checkpoints_dir):
    """Local folder for a model: ``<checkpoints_dir>/<full_name>`` when a checkpoints dir is given,
    otherwise the (already downloaded) Hugging Face snapshot dir.

    Mirrors ``scripts/interactive_demo/common.resolve_model_dir``.
    """
    from ardy.model.registry import hf_repo_id, resolve_model_name

    full_name = resolve_model_name(model_name)
    if checkpoints_dir:
        return os.path.join(checkpoints_dir, full_name)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_repo_id(full_name), local_files_only=True)


def main():
    parser = argparse.ArgumentParser(description="Export denoiser and decoder as ONNX/TRT")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Released model to export: a short key (core/g1/soma) or a full folder name.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default=None,
        help="Local dir holding released model folders. When unset, falls back "
        "to the CHECKPOINTS_DIR env var; if neither is set the model is "
        "downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for ONNX/TRT files (default: <model_dir>/engines, "
        "the same location the interactive demo loads engines from).",
    )
    parser.add_argument("--num_text_tokens", type=int, default=1, help="Num text tokens")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Max number of motion tokens. The engine is built for this fixed "
        "size; inputs are padded to it at runtime.",
    )
    parser.add_argument("--skip_trt", action="store_true", help="Skip TRT engine building")
    parser.add_argument("--skip_verify", action="store_true", help="Skip verification")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Build FP32 engines instead of FP16. Slower/larger but matches "
        "PyTorch closely — use to check whether a PyTorch-vs-TRT difference is "
        "just FP16 precision.",
    )
    args = parser.parse_args()

    # Imported here (not at module import time) so that the interactive demo can
    # import the export/build helpers from this module without pulling in the
    # full model-loading stack.
    from ardy.model.load_model import load_model

    device = "cuda:0"
    checkpoints_dir = args.checkpoints_dir or os.environ.get("CHECKPOINTS_DIR")

    if args.output_dir is None:
        args.output_dir = os.path.join(resolve_model_dir(args.model, checkpoints_dir), "engines")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    # No text encoder is needed: the denoiser takes the text features as a raw
    # tensor input, so skip loading the (large) encoder for export.
    log.info("Loading model %r ...", args.model)
    model, model_cfg = load_model(
        args.model,
        device=device,
        return_config=True,
        text_encoder=False,
        checkpoints_dir=checkpoints_dir,
    )
    num_frames_per_token = model.denoiser.num_frames_per_token
    log.info("Model loaded (num_frames_per_token=%d).", num_frames_per_token)

    # ── Export ONNX ──
    # Trace at the max token size — attention reshapes are fixed to this value,
    # and inputs are padded to it at runtime.
    max_tokens = args.max_tokens

    denoiser_onnx = os.path.join(args.output_dir, "denoiser.onnx")
    export_denoiser_onnx(
        model.denoiser,
        model_cfg,
        denoiser_onnx,
        num_tokens=max_tokens,
        num_text_tokens=args.num_text_tokens,
        opset=args.opset,
    )

    decoder_onnx = os.path.join(args.output_dir, "decoder.onnx")
    export_decoder_onnx(
        model.autoencoder,
        decoder_onnx,
        num_tokens=max_tokens,
        opset=args.opset,
    )

    # ── Build TRT engines ──
    # Precision is encoded in the filename so FP16/FP32 engines can coexist.
    fp16 = not args.fp32
    denoiser_trt = engine_path(args.output_dir, "denoiser", max_tokens, fp16)
    decoder_trt = engine_path(args.output_dir, "decoder", max_tokens, fp16)
    if not args.skip_trt:
        build_trt_engine(denoiser_onnx, denoiser_trt, max_tokens=max_tokens, fp16=fp16)
        build_trt_engine(decoder_onnx, decoder_trt, max_tokens=max_tokens, fp16=fp16)

        # ── Verify ──
        # A failed check is only a diagnostic — the engines are already built —
        # so log and continue rather than abort the run.
        if not args.skip_verify:
            log.info("Verifying denoiser TRT engine...")
            try:
                verify_denoiser(
                    model.denoiser,
                    denoiser_trt,
                    num_frames_per_token,
                    num_tokens=max_tokens,
                    num_text_tokens=args.num_text_tokens,
                )
            except Exception:
                log.warning("Denoiser verification failed.", exc_info=True)
            log.info("Verifying decoder TRT engine...")
            try:
                verify_decoder(
                    model.autoencoder,
                    decoder_trt,
                    num_frames_per_token,
                    num_tokens=max_tokens,
                )
            except Exception:
                log.warning("Decoder verification failed.", exc_info=True)
    else:
        log.info("Skipping TRT engine build (--skip_trt)")

    log.info("Done! Output files in %s", args.output_dir)


if __name__ == "__main__":
    main()
