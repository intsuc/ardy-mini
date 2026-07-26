# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model loading utilities: checkpoints, registry, env, and Hydra-based instantiation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

from .registry import (
    ARDY_MODELS,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    TMR_MODELS,
)


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return environment variable value, or default if unset/empty."""
    return os.environ.get(name) or default


def instantiate_from_dict(
    cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
):
    """Instantiate an object from a config dict (e.g. from OmegaConf.to_container).

    The dict must contain _target_ with a fully qualified class path. Nested configs are
    instantiated recursively.
    """
    if overrides:
        cfg = {**cfg, **overrides}
    conf = OmegaConf.create(cfg)
    return instantiate(conf)


def load_checkpoint_state_dict(ckpt_path: Union[str, Path]) -> dict:
    """Load a state dict from a checkpoint file.

    If the checkpoint is a dict with a 'state_dict' key (e.g. PyTorch Lightning),
    that is returned; otherwise the whole checkpoint is treated as the state dict.

    Args:
        ckpt_path: Path to the checkpoint file.

    Returns:
        state_dict suitable for model.load_state_dict().
    """
    ckpt_path = str(ckpt_path)

    if ckpt_path.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    return {key: val.detach().cpu() for key, val in state_dict.items()}


__all__ = [
    "get_env_var",
    "instantiate_from_dict",
    "ARDY_MODELS",
    "TMR_MODELS",
    "AVAILABLE_MODELS",
    "MODEL_NAMES",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "load_checkpoint_state_dict",
]
