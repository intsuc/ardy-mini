# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import glob
import hashlib
import json
import os
import pickle
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import viser
from einops import repeat
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ardy.constraints import (
    TYPE_TO_CLASS,
    FullBodyConstraintSet,
    Root2DConstraintSet,
)
from ardy.model.load_model import load_model, load_text_encoder
from ardy.model.registry import (
    DEFAULT_HORIZON,
    MODELS,
    MODELS_BY_SKELETON,
    hf_repo_id,
    parse_model_name,
    resolve_model_name,
)
from ardy.motion_rep import ArdyMotionRep
from ardy.postprocess import post_process_motion
from ardy.skeleton import (
    CoreSkeleton27,
    G1Skeleton34,
    SkeletonBase,
    SOMASkeleton30,
    SOMASkeleton77,
    batch_rigid_transform,
)
from ardy.skeleton.transforms import global_rots_to_local_rots
from ardy.tools import seed_everything
from ardy.viz.viser_utils import (
    Character,
    EEJointsKeyframeSet,
    FullbodyKeyframeSet,
    RootKeyframe2DSet,
    VelocityArrowMesh,
)

# Repo root, resolved from this file's location (scripts/interactive_demo/common.py).
# Use this for filesystem paths to repo assets so they stay correct regardless of
# where the importing module lives in the package tree.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# When CHECKPOINTS_DIR is set, models are discovered/loaded from that local
# folder; when unset, the released models are listed and pulled from Hugging Face.
CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR")
DEFAULT_MODEL_DIR = "ARDY-Core-RP-20FPS-Horizon40"


def scan_checkpoint_dirs(base_dir: str = "checkpoints") -> list[str]:
    """Return sorted list of names in base_dir whose folders contain model weights directly — either
    training ``.ckpt`` files or exported ``.safetensors`` files."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and (
            glob.glob(os.path.join(base_dir, name, "*.ckpt"))
            or glob.glob(os.path.join(base_dir, name, "*.safetensors"))
        )
    )


def available_models() -> list[str]:
    """Model choices for the dropdown.

    If CHECKPOINTS_DIR is set, discover model folders inside it. Otherwise, list the released models
    from the registry (downloaded from Hugging Face on load).
    """
    if CHECKPOINTS_DIR:
        return scan_checkpoint_dirs(CHECKPOINTS_DIR)
    return list(MODELS.values())


def models_by_skeleton() -> dict[str, dict[str, str]]:
    """Group the available models for the two-step picker in the Model tab:
    ``skeleton -> {horizon label -> model name}``.

    Skeleton keys are ``"core"``/``"g1"``/``"soma"`` with horizon labels like
    ``"40"``, sorted numerically. Local folders that don't follow the released
    naming scheme are grouped under ``"other"``, with the folder name itself as
    the label.
    """
    grouped: dict[str, dict[str, str]] = {}
    for name in available_models():
        parsed = parse_model_name(name)
        if parsed:
            skeleton, horizon = parsed
            grouped.setdefault(skeleton, {})[horizon] = name
        else:
            grouped.setdefault("other", {})[name] = name
    if not grouped:
        # Nothing found (e.g. empty CHECKPOINTS_DIR) — offer the released models.
        for skeleton, by_horizon in MODELS_BY_SKELETON.items():
            for horizon, folder in by_horizon.items():
                grouped.setdefault(skeleton, {})[horizon] = folder
    return {
        skeleton: {
            str(label): options[label]
            for label in (sorted(options) if skeleton == "other" else sorted(options, reverse=True))
        }
        for skeleton, options in grouped.items()
    }


def skeleton_supports_constraint_sampling(skeleton) -> bool:
    """Whether ``skeleton`` has a companion bones-seed dataset for constraint sampling / random
    motion file loading.

    The Core skeleton does not.
    """
    return not isinstance(skeleton, CoreSkeleton27)


def resolve_model_dir(model_name: str) -> str:
    """Local folder for a model: ``CHECKPOINTS_DIR/<name>`` when set, otherwise the (already
    downloaded, cached) Hugging Face snapshot dir."""
    full_name = resolve_model_name(model_name, checkpoints_dir=CHECKPOINTS_DIR)
    if CHECKPOINTS_DIR:
        return os.path.join(CHECKPOINTS_DIR, full_name)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_repo_id(full_name), local_files_only=True)


def is_g1_skeleton(skeleton) -> bool:
    """True when ``skeleton`` belongs to the G1 robot family (34 joints)."""
    return isinstance(skeleton, G1Skeleton34) or (getattr(skeleton, "nbjoints", None) == 34)


DEFAULT_REPLAN_TRIGGER_THRESH = 4
DEFAULT_REPLAN_BUFFER_SIZE = 1
DEFAULT_HISTORY_CROP_LENGTH = 4

# Default text prompt (initial GUI value) and the Prompt List preset buttons
# (gui/text.py). Also used by run_demo.py to prewarm the text-embedding
# cache at startup.
DEFAULT_PROMPT = "A person is walking."
PRESET_PROMPTS = [
    "A person is walking.",
    "A person jumps backwards.",
    "A person side steps to the right.",
    "A person is walking backwards.",
    "A person is kicking with their right leg.",
    "A person is standing.",
    "A young lady walks forward elegantly.",
    "A person bows down and then stands upright.",
    "A ballet dancer, performs a forward, turn joining feet, in a repeating loop",
    "a performer gives high bow, with arms to the side, right leg crossed behind the left",
]

LIGHT_THEME = dict(
    floor=(220, 220, 220),
    grid=(180, 180, 180),
)

DARK_THEME = dict(
    floor=(40, 40, 40),
    grid=(90, 90, 90),
)

INFINITE_FRAME_IDX = 99999
TIMELINE_WINDOW_BEFORE = 20
TIMELINE_WINDOW_AFTER = 200

ARROW_KEYS = {"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}

TARGET_VELOCITY_UPDATE_INTERVAL = 4
TARGET_VELOCITY_GOAL_FRAME_INTERVAL = 10
VELOCITY_TRANSITION_DURATION = 2.0  # seconds - duration for velocity to smoothly transition to target

# VelocityArrowMesh scales arrow length as magnitude / 4.0, so the "velocity"
# fed to it for the start-direction marker is scaled up to reach this length.
START_DIRECTION_MARKER_LENGTH = 0.5


@dataclass
class GuiElements:
    """GUI elements for the demo."""

    gui_skeleton_dropdown: viser.GuiDropdownHandle[str]
    gui_horizon_dropdown: viser.GuiDropdownHandle[str]
    gui_chosen_model_md: viser.GuiInputHandle
    gui_load_model_button: viser.GuiInputHandle
    gui_skeleton_name_text: viser.GuiInputHandle[str]
    gui_model_fps: viser.GuiInputHandle[int]
    gui_frame_idx_input: viser.GuiInputHandle[int]
    gui_play_pause_button: viser.GuiInputHandle
    gui_next_frame_button: viser.GuiInputHandle
    gui_prev_frame_button: viser.GuiInputHandle
    gui_actual_fps: viser.GuiInputHandle[float]
    gui_current_time: viser.GuiInputHandle[float]
    gui_compile_mode: viser.GuiInputHandle[str]
    gui_text_encoder_mode: viser.GuiInputHandle[str]
    gui_diffusion_steps_slider: viser.GuiInputHandle[int]
    gui_num_samples: viser.GuiInputHandle[int]
    gui_history_crop_length: viser.GuiInputHandle[int]
    gui_future_crop_length: viser.GuiInputHandle[int]
    gui_replan_buffer_size: viser.GuiInputHandle[int]
    gui_replan_trigger_thresh: viser.GuiInputHandle[int]
    gui_enable_auto_replan_checkbox: viser.GuiInputHandle[bool]
    gui_cfg_text_weight: viser.GuiInputHandle[float]
    gui_cfg_constraint_weight: viser.GuiInputHandle[float]
    gui_prompt_text: viser.GuiInputHandle[str]
    gui_active_prompt_label: viser.GuiInputHandle
    gui_seed: viser.GuiInputHandle[int]
    gui_use_target_velocity_checkbox: viser.GuiInputHandle[bool]
    gui_target_root_velocity: viser.GuiInputHandle  # 2D vector (x, z) for target root velocity
    gui_use_target_heading_checkbox: viser.GuiInputHandle[bool]
    gui_restart_button: viser.GuiInputHandle
    gui_load_seq_button: viser.GuiInputHandle
    gui_random_motion_button: viser.GuiInputHandle
    gui_waypoint_mode_checkbox: viser.GuiInputHandle[bool]
    gui_dense_root_checkbox: viser.GuiInputHandle[bool]
    gui_root_file_path: viser.GuiInputHandle[str]
    gui_load_root_button: viser.GuiInputHandle
    gui_save_root_button: viser.GuiInputHandle
    gui_scene_file_path: viser.GuiInputHandle[str]
    gui_mesh_transform_dropdown: viser.GuiInputHandle[str]
    gui_load_mesh_button: viser.GuiInputHandle
    gui_scene_translation_x: viser.GuiInputHandle[float]
    gui_scene_translation_y: viser.GuiInputHandle[float]
    gui_scene_translation_z: viser.GuiInputHandle[float]
    gui_waypoint_interval: viser.GuiInputHandle[int]
    gui_max_keyframe_num: viser.GuiInputHandle[int]
    gui_motion_file_path: viser.GuiInputHandle[str]
    gui_constraint_fullbody_checkbox: viser.GuiInputHandle[bool]
    gui_constraint_hands_checkbox: viser.GuiInputHandle[bool]
    gui_constraint_feet_checkbox: viser.GuiInputHandle[bool]
    gui_constraint_hands_feet_checkbox: viser.GuiInputHandle[bool]
    gui_constraint_2d_waypoints_checkbox: viser.GuiInputHandle[bool]
    gui_constraint_2d_trajectory_checkbox: viser.GuiInputHandle[bool]
    gui_viz_skeleton_checkbox: viser.GuiInputHandle[bool]
    gui_viz_foot_contacts_checkbox: viser.GuiInputHandle[bool]
    gui_viz_ref_motion_checkbox: viser.GuiInputHandle[bool]
    gui_viz_skinned_mesh_checkbox: viser.GuiInputHandle[bool]
    gui_viz_skinned_mesh_opacity_slider: viser.GuiInputHandle[float]
    gui_viz_hand_orientations_checkbox: viser.GuiInputHandle[bool]
    gui_viz_hide_distant_constraints_checkbox: viser.GuiInputHandle[bool]
    gui_show_timeline_checkbox: viser.GuiInputHandle[bool]
    gui_show_start_direction_checkbox: viser.GuiInputHandle[bool]
    gui_viz_auto_camera_checkbox: viser.GuiInputHandle[bool]
    gui_viz_camera_type_dropdown: viser.GuiInputHandle[str]
    gui_viz_camera_fov_slider: viser.GuiInputHandle[float]
    gui_viz_camera_position: viser.GuiInputHandle[tuple[float, float, float]]
    gui_viz_camera_look_at: viser.GuiInputHandle[tuple[float, float, float]]
    gui_viz_camera_up: viser.GuiInputHandle[tuple[float, float, float]]
    gui_dark_mode_checkbox: viser.GuiInputHandle[bool]
    gui_enable_postprocess_checkbox: viser.GuiInputHandle[bool]
    gui_postprocess_root_margin: viser.GuiInputHandle[float]
    gui_postprocess_contact_threshold: viser.GuiInputHandle[float]


@dataclass
class ClientSession:
    """Per-client session data."""

    client: viser.ClientHandle
    gui_elements: GuiElements

    # Model and motion representation (per-client)
    model: Optional[object] = None
    motion_rep: Optional[object] = None
    motion_rep_infer: Optional[object] = None
    model_fps: int = 30
    num_frames_per_token: int = 4
    gen_horizon_len: int = 20
    max_window_len: int = 300  # per-step window budget in frames; set on model load
    dataset: Optional[object] = None
    mesh_mode: str = "core_skin"  # Mesh visualization mode for character creation

    # Motion state
    motion_tensor: Optional[torch.Tensor] = None
    joints_pos: Optional[torch.Tensor] = None
    joints_rot: Optional[torch.Tensor] = None
    foot_contacts: Optional[torch.Tensor] = None
    root_velocities: Optional[torch.Tensor] = None  # [N, T, 3] root joint velocities (x, y, z)
    characters: dict = field(default_factory=dict)
    target_velocity_arrow: Optional[object] = None  # VelocityArrowMesh for target velocity visualization

    # Initial body transform (for generation)
    init_global_translation: Optional[np.ndarray] = None  # [3] initial body translation, float32
    init_first_heading_angle: Optional[float] = None  # Initial heading angle in radians
    init_transform_gizmo: Optional[object] = None  # Transform control gizmo
    start_direction_marker: Optional[object] = None  # Blue VelocityArrowMesh showing initial position + facing

    # Playback state
    frame_idx: int = -1
    max_frame_idx: int = -1
    playing: bool = False
    cur_time: float = -1.0
    playback_fps: int = 30

    # Camera state for smoothing
    camera_position: Optional[np.ndarray] = None
    camera_look_at: Optional[np.ndarray] = None
    camera_forward_direction: Optional[np.ndarray] = None  # Smoothed character forward direction
    camera_position_buffer: list = field(default_factory=list)  # Temporal buffer for smoothing
    camera_last_update_frame: int = -1  # Track last frame when camera was updated

    # Text prompt
    text_embedding: Optional[torch.Tensor] = None

    # Waypoints (now managed by 2D Root constraint)
    waypoint_mode: bool = False
    click_plane: Optional[object] = None

    # Constraints and timeline
    constraints: dict = field(default_factory=dict)
    timeline_data: Optional[dict] = None
    edit_mode: bool = False

    # Threading locks
    replan_lock: threading.Lock = field(default_factory=threading.Lock)
    characters_lock: threading.Lock = field(default_factory=threading.Lock)
    motion_tensor_lock: threading.Lock = field(default_factory=threading.Lock)

    # Playback thread control
    playback_thread: Optional[threading.Thread] = None
    stop_playback: bool = False

    # Hand orientation gizmos (dict of character_name -> dict of joint_name -> gizmo)
    hand_gizmos: dict = field(default_factory=dict)

    # Loaded scene mesh handle
    loaded_scene_mesh_handle: Optional[object] = None

    # Reference motion (loaded from file for constraint sampling)
    ref_character: Optional[object] = None
    ref_joints_pos: Optional[torch.Tensor] = None  # [T, J, 3]
    ref_joints_rot: Optional[torch.Tensor] = None  # [T, J, 3, 3]

    # MujocoQposConverter for Mujoco/CSV motion import (G1 only)
    mujoco_converter: Optional[object] = None


def add_checkerboard(
    client,
    grid_size=10,
    square_size=0.5,
    plane_thickness=0.01,
    color1=(230, 230, 230),
    color2=(40, 40, 40),
):
    """Add a checkerboard floor to the scene."""
    offset = -grid_size * square_size / 2.0 + square_size / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                color = color1
            else:
                color = color2

            position = (
                i * square_size + offset,
                -plane_thickness / 2.0,
                j * square_size + offset,
            )

            client.scene.add_box(
                name=f"/checkerboard/cell_{i}_{j}",
                dimensions=(square_size, plane_thickness, square_size),
                color=color,
                position=position,
            )
