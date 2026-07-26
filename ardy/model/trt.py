# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine runtime wrappers with PyTorch-compatible interfaces.

Provides drop-in replacements for the denoiser wrapper and autoencoder decoder that run inference
via pre-built TRT engines instead of PyTorch.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTEngine:
    """Loads a serialized TensorRT engine and runs inference.

    Manages device memory for engine I/O bindings and provides a simple ``infer(output_shapes,
    **inputs) -> dict[str, Tensor]`` interface.
    """

    def __init__(self, engine_path: str) -> None:
        if trt is None:
            raise ImportError("tensorrt is required for TRT inference. Install with: pip install tensorrt")
        self.logger = trt.Logger(trt.Logger.WARNING)
        engine_path = str(engine_path)
        log.info("Loading TRT engine from %s", engine_path)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine: {engine_path}\n"
                f"The engine was built with a different TensorRT version. "
                f"Rebuild it with the currently installed version "
                f"({trt.__version__}) by running scripts/export_onnx.py."
            )
        self.context = self.engine.create_execution_context()

        # Catalogue bindings and their static shapes
        self.input_names = []
        self.output_names = []
        self.binding_dtypes = {}
        self.input_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            self.binding_dtypes[name] = dtype
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = tuple(shape)
            else:
                self.output_names.append(name)
        log.info(
            "TRT engine loaded: %d inputs (%s), %d outputs (%s)",
            len(self.input_names),
            ", ".join(self.input_names),
            len(self.output_names),
            ", ".join(self.output_names),
        )

    def infer(
        self,
        output_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        **inputs: torch.Tensor,
    ) -> dict:
        """Run inference with the given named inputs.

        Args:
            output_shapes: Optional expected shape per output tensor. Any output
                not listed here (or all of them, if None) has its shape resolved
                from the execution context after the input shapes are set — which
                works for graphs with multiple or auto-named outputs.
            stream: CUDA stream to use. If None, uses the current stream.
            **inputs: Named input tensors (must be contiguous, on CUDA).

        Returns:
            Dictionary mapping output tensor names to GPU tensors.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        output_shapes = output_shapes or {}

        # Set input shapes and addresses
        for name in self.input_names:
            tensor = inputs[name]
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                inputs[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate outputs. Use the caller-provided shape when given, otherwise
        # ask the context for the shape it resolved from the input shapes above.
        outputs = {}
        for name in self.output_names:
            shape = output_shapes.get(name)
            if shape is None:
                shape = tuple(self.context.get_tensor_shape(name))
            np_dtype = self.binding_dtypes[name]
            torch_dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
            out = torch.empty(shape, dtype=torch_dtype, device="cuda")
            outputs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        # Execute
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


def _pad_to(tensor: torch.Tensor, target: int, dim: int) -> torch.Tensor:
    """Zero-pad ``tensor`` along ``dim`` to ``target`` size."""
    cur = tensor.shape[dim]
    if cur >= target:
        return tensor
    pad_sizes = [0] * (2 * tensor.ndim)
    # F.pad uses reversed dim order: last dim first
    pad_idx = 2 * (tensor.ndim - 1 - dim)
    pad_sizes[pad_idx + 1] = target - cur
    return F.pad(tensor, pad_sizes)


class TRTCFGDenoiser(nn.Module):
    """Drop-in replacement for AutoLatentClassifierFreeGuidedModel using a TRT engine.

    The TRT engine includes both separated CFG batching and the denoiser. It takes B=1 inputs +
    cfg_weights and returns the CFG-combined B=1 output.

    Because the ONNX graph has fixed attention reshapes, all inputs are padded to the trace-time
    sizes (``num_tokens`` and ``num_text_tokens``) before calling the engine, and the output is
    sliced back to the original size.
    """

    def __init__(
        self,
        engine_path: str,
        denoiser: nn.Module,
        num_frames_per_token: int,
        num_tokens: int,
        num_text_tokens: int,
    ) -> None:
        """
        Args:
            engine_path: Path to the serialized CFG+denoiser TRT engine.
            denoiser: The original denoiser (or infer wrapper) for attribute access.
            num_tokens: The fixed num_tokens the ONNX was traced with.
            num_text_tokens: The fixed num_text_tokens the ONNX was traced with.
        """
        super().__init__()
        self._engine = TRTEngine(engine_path)
        self._denoiser = denoiser
        self._num_tokens = num_tokens
        self._num_text_tokens = num_text_tokens
        self._nfpt = num_frames_per_token
        self._num_frames = num_tokens * self._nfpt
        self._dim_token = denoiser.nframe_root_dim + denoiser.latent_embedding_dim
        self._motion_rep_dim = denoiser.motion_rep.motion_rep_dim
        # Expose model attribute so denoising_step can access backbone dims
        self.model = denoiser

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._denoiser, name)

    def forward(
        self,
        cfg_weight_text: torch.Tensor,
        cfg_weight_cstr: torch.Tensor,
        token_seq_t: torch.Tensor,
        history_len: torch.Tensor,
        generation_len: torch.Tensor,
        future_len: torch.Tensor,
        history_mask: torch.Tensor,
        generation_mask: torch.Tensor,
        future_mask: torch.Tensor,
        history_token_mask: torch.Tensor,
        generation_token_mask: torch.Tensor,
        future_token_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_feat_pad_mask: torch.Tensor,
        timesteps: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
        observed_motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        actual_num_tokens = token_seq_t.shape[1]
        if actual_num_tokens > self._num_tokens:
            # TensorRT's Python API only logs oversize setInputShape errors and
            # then runs with the engine's stale shapes, returning a truncated
            # output that crashes far downstream — so refuse loudly here.
            raise ValueError(
                f"Generation window of {actual_num_tokens} tokens "
                f"({actual_num_tokens * self._nfpt} frames) exceeds the "
                f"{self._num_tokens}-token ({self._num_frames}-frame) capacity "
                f"this TRT engine was built with. Reduce the window "
                f"(history/future crop) or rebuild the engine with more tokens."
            )
        T = self._num_tokens  # padded token count
        F_pad = self._num_frames  # padded frame count
        T_text = self._num_text_tokens
        device = token_seq_t.device
        mrep = self._motion_rep_dim

        if first_heading_angle is None:
            first_heading_angle = torch.zeros(1, device=device, dtype=torch.float32)
        if motion_mask is None:
            motion_mask = torch.zeros(1, F_pad, mrep, device=device)
        if observed_motion is None:
            observed_motion = torch.zeros(1, F_pad, mrep, device=device)

        # Pad all inputs to the fixed trace-time sizes
        x_pad = _pad_to(token_seq_t, T, dim=1)
        hmask_pad = _pad_to(history_mask, F_pad, dim=1)
        gmask_pad = _pad_to(generation_mask, F_pad, dim=1)
        fmask_pad = _pad_to(future_mask, F_pad, dim=1)
        htmask_pad = _pad_to(history_token_mask, T, dim=1)
        gtmask_pad = _pad_to(generation_token_mask, T, dim=1)
        ftmask_pad = _pad_to(future_token_mask, T, dim=1)
        text_pad = _pad_to(text_feat, T_text, dim=1)
        tpmask_pad = _pad_to(text_feat_pad_mask, T_text, dim=1)
        mm_pad = _pad_to(motion_mask, F_pad, dim=1)
        om_pad = _pad_to(observed_motion, F_pad, dim=1)

        output_shapes = {"output": (1, T, self._dim_token)}
        results = self._engine.infer(
            output_shapes,
            cfg_weight_text=cfg_weight_text.float(),
            cfg_weight_cstr=cfg_weight_cstr.float(),
            x=x_pad.float(),
            history_len=history_len.long(),
            generation_len=generation_len.long(),
            future_len=future_len.long(),
            history_mask=hmask_pad.float(),
            generation_mask=gmask_pad.float(),
            future_mask=fmask_pad.float(),
            history_token_mask=htmask_pad.float(),
            generation_token_mask=gtmask_pad.float(),
            future_token_mask=ftmask_pad.float(),
            text_feat=text_pad.float(),
            text_feat_pad_mask=tpmask_pad.float(),
            timesteps=timesteps.long(),
            first_heading_angle=first_heading_angle.float(),
            motion_mask=mm_pad.float(),
            observed_motion=om_pad.float(),
        )
        # Slice back to actual size
        return results["output"][:, :actual_num_tokens, :]


# FSQVAETransformer.detokenize splits the raw decoder output into these dict
# keys (in this order) depending on decoder_output_feature_mode. The decoder
# ONNX therefore has one graph output per key, in the same order, so the engine
# outputs line up with these positionally. Kept in sync with fsq.py.
_DECODER_OUTPUT_KEYS = {
    "pose": ("root", "body"),
    "body": ("body",),
    "root": ("root",),
}


class TRTDecoder(nn.Module):
    """Drop-in replacement for the autoencoder decoder using a TRT engine.

    Pads inputs to the trace-time sizes (num_tokens / num_frames) to match the fixed attention
    reshapes in the ONNX graph, runs the engine, and slices each output back to the real frame
    count. Returns one tensor per graph output, in graph order (``detokenize`` splits its result
    into a dict, so the decoder graph has one output per dict entry — e.g. root and body).
    """

    def __init__(
        self,
        engine_path: str,
        decoder_output_dim: Optional[int] = None,
        num_tokens: int = 8,
        num_frames_per_token: int = 4,
    ) -> None:
        super().__init__()
        self._engine = TRTEngine(engine_path)
        # decoder_output_dim is accepted for backward compatibility but no longer
        # needed: output shapes are resolved from the engine at inference time.
        self._num_tokens = num_tokens
        self._num_frames = num_tokens * num_frames_per_token

    def forward(
        self,
        latent_tokens: torch.Tensor,
        external_cond: torch.Tensor,
        motion_pad_mask: torch.Tensor,
    ) -> list:
        actual_num_frames = motion_pad_mask.shape[1]
        T = self._num_tokens
        F_pad = self._num_frames

        # Pad to trace-time sizes
        lt_pad = _pad_to(latent_tokens, T, dim=1)
        ec_pad = _pad_to(external_cond, F_pad, dim=1)
        mp_pad = _pad_to(motion_pad_mask, F_pad, dim=1)

        results = self._engine.infer(
            latent_tokens=lt_pad.float(),
            external_cond=ec_pad.float(),
            motion_pad_mask=mp_pad.float(),
        )
        # One tensor per graph output, in graph order, sliced back to actual size.
        return [results[name][:, :actual_num_frames, :] for name in self._engine.output_names]


class TRTAutoencoder(nn.Module):
    """Wraps the original autoencoder but replaces detokenize() with a TRT decoder."""

    def __init__(self, trt_decoder: TRTDecoder, autoencoder: nn.Module) -> None:
        super().__init__()
        self._trt_decoder = trt_decoder
        self._autoencoder = autoencoder
        # The decoder graph emits one output per detokenize() dict key; recover
        # those keys (in order) so detokenize() below returns the same dict the
        # PyTorch autoencoder does.
        mode = autoencoder.decoder_output_feature_mode
        if mode not in _DECODER_OUTPUT_KEYS:
            raise ValueError(
                f"Unsupported decoder_output_feature_mode {mode!r}; expected one of {list(_DECODER_OUTPUT_KEYS)}."
            )
        self._output_keys = _DECODER_OUTPUT_KEYS[mode]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._autoencoder, name)

    def detokenize(
        self,
        token_embeddings: torch.Tensor,
        external_cond: torch.Tensor = None,
        motion_pad_mask: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Route through the TRT decoder, returning the same dict as the PyTorch ``detokenize``
        (e.g. ``{"root": ..., "body": ...}``)."""
        outputs = self._trt_decoder(token_embeddings, external_cond, motion_pad_mask)
        if len(outputs) != len(self._output_keys):
            raise RuntimeError(
                f"Decoder engine produced {len(outputs)} outputs but "
                f"decoder_output_feature_mode implies keys {self._output_keys}."
            )
        return dict(zip(self._output_keys, outputs))
