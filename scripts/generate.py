# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cmd line API for generating motions with Ardy.

Generated files are written under outputs/ unless --output contains a path.

Examples:
    python scripts/generate.py "A person walks in a circle." --checkpoints_dir checkpoints
    python scripts/generate.py "A person jumps." --model soma --num_samples 4 --seed 0 --output jump
    python scripts/generate.py "A person walks forward." --model g1 --duration 8.0
"""

import argparse
import os

import numpy as np
import torch

from ardy.constraints import load_constraints_lst
from ardy.model import DEFAULT_MODEL, load_model
from ardy.model.loading import get_env_var
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.postprocess import post_process_motion
from ardy.skeleton import SOMASkeleton30
from ardy.tools import seed_everything, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Cmd line API for generating motions with Ardy")
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the motion to generate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model nickname (core/g1/soma, optionally with horizon like core8) or full folder name.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=None,
        help="Number of diffusion steps, at most the model's schedule length (num_base_steps, the default).",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Saved constraint list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output stem name: with one sample writes a single file per format (e.g. test.npz, test.csv); with multiple samples creates a folder and writes test_00.npz, test_01.npz, ... inside it. Used for NPZ and CSV. Bare names are written under outputs/; pass a path (e.g. ./test or results/test) to write elsewhere.",
    )
    parser.add_argument(
        "--history_frames",
        type=int,
        default=None,
        help="History frames visible to each autoregressive step (multiple of the model's token size). Default: the longest history that fits the model's trained 10s window; smaller values adapt faster but transition more abruptly.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Don't apply motion post-processing to reduce foot skating (ignored for G1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        nargs="+",
        default=[2.0, 2.0],
        help="CFG scale(s): one float (text weight only) or two floats [text_weight, constraint_weight] (default: 2.0 2.0).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Local dir holding released model folders. Falls back to the CHECKPOINTS_DIR env var; if neither is set the model is downloaded from Hugging Face.",
    )
    return parser.parse_args()


def _default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """Longest history that, together with the generation horizon, fits the model's trained window.

    ARDY is trained on windows of at most 10 s. Without history cropping the attention window grows
    with the output, and generations longer than the trained window degrade into jitter — so each
    autoregressive step must see at most this many history frames (same budget as the interactive
    demo).
    """
    max_window_len = (int(10 * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


def _resolve_output_base(path: str, default_dir: str = "outputs") -> str:
    """Place bare output names under ``default_dir``; honor explicit (relative or absolute)
    paths."""
    if os.path.dirname(path):
        return path
    return os.path.join(default_dir, path)


def _single_file_path(path: str, ext: str) -> str:
    """Return path for a single output file (no folder).

    Adds ext if missing; creates parent dirs if any.
    """
    if not path.endswith(ext):
        path = path.rstrip(os.sep) + ext
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_dir_and_path(path: str, default_base: str, ext: str):
    """Create output folder from path and return (dir_path, path_for_file_with_suffix, base_name).

    If path has an extension, folder name is the path stem; else the path is the folder name.
    base_name is the folder basename for _00, _01, ... when n_samples > 1.
    """
    folder = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.basename(folder.rstrip(os.sep))
    return folder, os.path.join(folder, default_base + ext), base_name


def _select_sample(output: dict, index: int, n_samples: int) -> dict:
    """Extract sample ``index`` from a batched output dict (non-batched entries kept as-is)."""
    return {
        k: (v[index] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples else v)
        for k, v in output.items()
    }


def save_motion_npz(path: str, motion_dict: dict, fps: float, text: str) -> None:
    """Save a motion output dict to ``.npz`` along with fps and the prompt."""
    arrays = {k: np.asarray(v) for k, v in motion_dict.items()}
    arrays["fps"] = np.asarray(fps)
    arrays["text"] = np.asarray(text)
    np.savez(path, **arrays)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args = parse_args()

    if args.num_samples < 1:
        raise ValueError(f"--num_samples must be >= 1, got {args.num_samples}.")

    if len(args.cfg_weight) == 1:
        cfg_weight = float(args.cfg_weight[0])
    elif len(args.cfg_weight) == 2:
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
    else:
        raise ValueError("--cfg_weight expects one float (text) or two floats (text, constraint).")

    # Load model
    checkpoints_dir = args.checkpoints_dir or get_env_var("CHECKPOINTS_DIR")
    resolved_model = resolve_model_name(args.model, checkpoints_dir=checkpoints_dir)
    model = load_model(resolved_model, device=device, checkpoints_dir=checkpoints_dir)
    print(f"Loaded model: {resolved_model}")

    fps = model.motion_rep.fps
    num_frames = int(args.duration * fps)
    text = args.prompt.strip()
    print(f"Will generate '{text}' with {num_frames} frames ({args.duration}s at {fps} fps)")

    # The diffusion schedule can only be subsampled: asking for more steps than
    # num_base_steps indexes past the timestep map (CUDA device-side assert).
    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = args.diffusion_steps if args.diffusion_steps is not None else num_base_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(
            f"--diffusion_steps must be between 1 and {num_base_steps} "
            f"(this model's num_base_steps); got {diffusion_steps}."
        )
    print(f"Using {diffusion_steps} denoising steps")

    # Cap the history each autoregressive step sees, so long generations stay
    # within the trained window (unbounded history degrades into jitter).
    patch = model.num_frames_per_token
    history_frames = args.history_frames
    if history_frames is None:
        history_frames = _default_history_frames(fps, model.gen_horizon_len, patch)
    elif history_frames < patch or history_frames % patch != 0:
        raise ValueError(f"--history_frames must be a positive multiple of {patch} (this model's token size).")
    print(f"Using {history_frames} history frames per autoregressive step")

    # Load constraints
    if args.constraints:
        constraint_lst = load_constraints_lst(args.constraints, model.skeleton)
    else:
        constraint_lst = []

    if constraint_lst:
        print(f"Using {len(constraint_lst)} set of constraints")
        for constraint in constraint_lst:
            print(f"    {type(constraint).__name__} on frames {constraint.frame_indices.tolist()}")
        max_frame_idx = max(int(c.frame_indices.max()) for c in constraint_lst)
        if max_frame_idx >= num_frames:
            raise ValueError(
                f"Constraint frame index {max_frame_idx} exceeds the motion length "
                f"({num_frames} frames = {args.duration}s at {fps} fps); increase --duration."
            )

    if args.seed is not None:
        seed_everything(args.seed)

    # Build the generation inputs (batch of identical prompts/lengths)
    num_samples = args.num_samples
    texts = [text] * num_samples
    lengths = torch.tensor([num_frames] * num_samples, device=device)
    pad_mask = length_to_mask(lengths)
    first_heading_angle = torch.zeros(num_samples, device=device)  # facing +Z

    observed_motion, motion_mask = None, None
    if constraint_lst:
        observed_motion, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
            constraint_lst,
            lengths,
            to_normalize=True,
            device=device,
        )

    with torch.no_grad():
        motion = model(
            texts,
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=first_heading_angle,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=cfg_weight,
            crop_history_length=history_frames,
        )
        output = model.motion_rep.inverse(motion, is_normalized=True)

    # G1: postprocessing is disabled (does not work well for this model).
    use_postprocess = "g1" not in resolved_model.lower() and not args.no_postprocess
    if use_postprocess:
        corrected = post_process_motion(
            output["local_rot_mats"],
            output["root_positions"],
            output["foot_contacts"],
            model.skeleton,
            constraint_lst=constraint_lst or None,
        )
        output.update(corrected)

    # Convert SOMA output to somaskel77 for external API
    if isinstance(model.skeleton, SOMASkeleton30):
        output = model.skeleton.output_to_SOMASkeleton77(output)

    output = to_numpy(output)

    n_samples = int(output["posed_joints"].shape[0])
    # Parse the output stem once; all formats (NPZ, CSV) use this base name.
    output_base = _resolve_output_base(args.output)

    # Save the NPZ output
    if n_samples == 1:
        npz_path = _single_file_path(output_base, ".npz")
        print(f"Saving the npz output to {npz_path}")
        save_motion_npz(npz_path, _select_sample(output, 0, n_samples), fps, text)
    else:
        out_dir, _, base_name = _output_dir_and_path(output_base, "motion", ".npz")
        print(f"Saving the npz output to {out_dir}/ ({base_name}_00.npz ...)")
        for i in range(n_samples):
            save_motion_npz(
                os.path.join(out_dir, f"{base_name}_{i:02d}.npz"),
                _select_sample(output, i, n_samples),
                fps,
                text,
            )

    # Save the CSV output (MuJoCo qpos) for G1
    if "g1" in resolved_model.lower():
        from ardy.exports.mujoco import MujocoQposConverter

        converter = MujocoQposConverter(model.skeleton)
        qpos = converter.dict_to_qpos(output, device)
        if n_samples == 1:
            csv_path = _single_file_path(output_base, ".csv")
            print(f"Saving the csv output to {csv_path}")
            converter.save_csv(qpos, csv_path)
        else:
            out_dir, _, base_name = _output_dir_and_path(output_base, "qpos", ".csv")
            print(f"Saving the csv output to {out_dir}/ ({base_name}_00.csv ...)")
            converter.save_csv(qpos, os.path.join(out_dir, base_name + ".csv"))


if __name__ == "__main__":
    main()
