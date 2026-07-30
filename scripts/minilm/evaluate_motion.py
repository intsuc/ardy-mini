# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Compare paired ARDY rollouts using teacher and distilled text conditions.

The comparison uses identical prompts, seeds, diffusion settings, and ARDY
checkpoint. Teacher LLM2Vec embeddings are read from the offline cache, so the
large teacher does not coexist with the motion model. This does not alter the
condition: the cache stores the production wrapper's float32 4096-vector.

The public ARDY repository does not ship the proprietary Rigplay test split or
its TMR evaluator. These metrics therefore quantify fidelity to the original
encoder (paired root/joint/velocity/contact errors), not paper-comparable FID
or R-precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ardy.minilm_teacher_cache import (
    TeacherCache,
    load_teacher_cache,
    validate_artifact_teacher_cache_fingerprint,
)
from ardy.model import MiniLMArdyEncoder, load_model
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.tools import seed_everything, to_numpy

TEACHER_DIM = 4096
TARGET_DIM = 2048
VALID_SPLITS = {"train", "val", "test"}
DEFAULT_PROMPT_MANIFEST = Path(
    "artifacts/data/prompts-core40-timeline.jsonl"
)
DEFAULT_SOURCES = ["overview_description", "events.description"]
DEFAULT_NUM_PROMPTS = 64
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_STUDENT_DTYPE = "float32"
PROMPT_SELECTION_ALGORITHM = "evenly_spaced_numpy_linspace_v1"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REPEATABILITY_OUTPUT_KEYS = (
    "root_positions",
    "posed_joints",
    "global_root_heading",
    "foot_contacts",
)
SUMMARY_CONTEXT_KEYS = {
    "prompt_index",
    "seed",
    "seed_a",
    "seed_b",
    "text",
    # Generation timings exclude text encoding and are diagnostic only. They
    # must not be folded into the motion-fidelity summary.
    "teacher_generation_seconds",
    "student_generation_seconds",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=DEFAULT_PROMPT_MANIFEST,
        help="Prepared prompt JSONL used to validate cache lineage",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help=(
            "Prompt-manifest source values to evaluate: overview_description and/or events.description"
        ),
    )
    parser.add_argument("--student-path", default="artifacts/minilm-ardy-core40")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--model", default="core")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=DEFAULT_DIFFUSION_STEPS,
    )
    parser.add_argument("--cfg-weight", type=float, nargs=2, default=[2.0, 2.0])
    parser.add_argument("--output", default="artifacts/evaluation/motion_metrics.json")
    parser.add_argument("--sample-dir", default="outputs/minilm-motion-comparison")
    parser.add_argument("--save-samples", type=int, default=3)
    parser.add_argument(
        "--student-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=DEFAULT_STUDENT_DTYPE,
    )
    parser.add_argument(
        "--repeatability-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "generate the first teacher case twice and require bitwise-equal "
            "motion outputs"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or e.g. cuda:0",
    )
    return parser.parse_args(argv)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_prompt_sha256(texts: list[str]) -> str:
    """Hash the ordered selected prompt bodies without publishing them."""

    encoded = json.dumps(
        texts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_determinism() -> dict[str, Any]:
    """Configure and describe the exact motion-evaluation determinism policy."""

    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace is None:
        os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    elif configured_workspace != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be "
            f"{DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r}, got "
            f"{configured_workspace!r}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device={requested!r} requested, but CUDA is unavailable"
        )
    return requested


def assert_exact_repeatability(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> list[str]:
    """Require exact equality for motion arrays used by public metrics."""

    compared: list[str] = []
    for key in REPEATABILITY_OUTPUT_KEYS:
        if key not in first or key not in second:
            raise ValueError(
                f"repeatability output is missing required key {key!r}"
            )
        left = np.asarray(first[key])
        right = np.asarray(second[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"repeatability check failed for {key}: "
                f"{left.shape}/{left.dtype} != {right.shape}/{right.dtype}"
            )
        if np.issubdtype(left.dtype, np.floating) and (
            not np.isfinite(left).all() or not np.isfinite(right).all()
        ):
            raise RuntimeError(
                f"repeatability output {key} contains NaN or infinity"
            )
        if not np.array_equal(left, right):
            raise RuntimeError(
                f"repeatability check failed for {key}: arrays differ"
            )
        compared.append(key)
    return compared


def student_teacher_checkpoint_sha256(artifact_config: dict[str, Any]) -> str:
    """Read the checkpoint hash from the artifact's canonical cache lineage."""

    metadata = artifact_config.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Student artifact metadata must be an object")
    teacher_lineage = metadata.get("teacher_cache_lineage")
    if not isinstance(teacher_lineage, dict):
        raise TypeError("Student artifact is missing teacher-cache lineage")
    checkpoint_hash = teacher_lineage.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_hash)
    ):
        raise ValueError(
            "Student artifact teacher_cache_lineage.checkpoint_sha256 is invalid"
        )
    return checkpoint_hash


def _read_prompt_manifest(
    prompt_manifest: Path,
    *,
    teacher_metadata: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    if not prompt_manifest.is_file():
        raise FileNotFoundError(f"Prompt manifest not found: {prompt_manifest}")
    manifest_sha256 = sha256_file(prompt_manifest)
    expected_sha256 = teacher_metadata.get("input_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("Teacher-cache metadata has no valid input_sha256")
    if manifest_sha256 != expected_sha256:
        raise ValueError(
            "Prompt-manifest SHA-256 does not match teacher-cache input_sha256: "
            f"manifest={manifest_sha256}, teacher_cache={expected_sha256}"
        )

    records: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    with prompt_manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(f"{prompt_manifest}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{prompt_manifest}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"{prompt_manifest}:{line_number}: row must be an object")
            text = value.get("text")
            source = value.get("source")
            split = value.get("split")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{prompt_manifest}:{line_number}: text must be a non-empty string")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{prompt_manifest}:{line_number}: source must be a non-empty string")
            if split not in VALID_SPLITS:
                raise ValueError(
                    f"{prompt_manifest}:{line_number}: split must be one of {sorted(VALID_SPLITS)}, got {split!r}"
                )
            if text in seen_texts:
                raise ValueError(f"{prompt_manifest}:{line_number}: duplicate prompt text {text!r}")
            seen_texts.add(text)
            records.append({"text": text, "source": source, "split": split})

    if len(records) != teacher_metadata["count"]:
        raise ValueError(
            f"Prompt manifest contains {len(records)} rows, but teacher cache declares {teacher_metadata['count']}"
        )
    return records, manifest_sha256


def load_test_embeddings(
    cache_dir: str | Path | TeacherCache,
    *,
    prompt_manifest: str | Path | None = None,
    sources: list[str] | None = None,
) -> tuple[
    list[tuple[str, torch.Tensor]],
    dict[str, Any],
    dict[str, Any],
]:
    cache = (
        cache_dir
        if isinstance(cache_dir, TeacherCache)
        else load_teacher_cache(
            cache_dir,
            expected_teacher_dim=TEACHER_DIM,
            expected_target_dim=TARGET_DIM,
            keep_teacher_embeddings=True,
            keep_targets=False,
        )
    )
    metadata = cache.metadata
    manifest_records: list[dict[str, str]] | None = None
    manifest_sha256: str | None = None
    if sources is not None and prompt_manifest is None:
        raise ValueError("--sources requires --prompt-manifest")
    if sources is not None and (
        not sources
        or not all(isinstance(source, str) and source.strip() for source in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("--sources must contain unique, non-empty source names")
    if prompt_manifest is not None:
        manifest_records, manifest_sha256 = _read_prompt_manifest(
            Path(prompt_manifest),
            teacher_metadata=metadata,
        )
    requested_sources = None if sources is None else list(sources)
    requested_source_set = None if requested_sources is None else set(requested_sources)

    records: list[tuple[str, torch.Tensor]] = []
    observed_count = 0
    observed_manifest_sources = (
        set()
        if manifest_records is None
        else {record["source"] for record in manifest_records}
    )
    for shard in cache.shards:
        embeddings = shard.teacher_embeddings
        if embeddings is None:
            raise RuntimeError("strict teacher-cache loader did not retain embeddings")
        for local_index, (text, split, embedding) in enumerate(
            zip(shard.texts, shard.splits, embeddings, strict=True)
        ):
            source = None
            if manifest_records is not None:
                manifest_record = manifest_records[observed_count + local_index]
                if manifest_record["text"] != text or manifest_record["split"] != split:
                    raise ValueError(
                        "Prompt manifest does not match teacher-cache text/split at "
                        f"global row {observed_count + local_index}: "
                        f"manifest={(manifest_record['text'], manifest_record['split'])!r}, "
                        f"cache={(text, split)!r}"
                    )
                source = manifest_record["source"]
            if split == "test" and (requested_source_set is None or source in requested_source_set):
                # Clone the row so retaining a test example does not retain the
                # complete shard storage.
                records.append((text, embedding.clone()))
        observed_count += len(shard.texts)

    if observed_count != metadata["count"]:
        raise ValueError(
            f"Teacher-cache manifest declares {metadata['count']} examples, but shards contain {observed_count}"
        )
    if requested_source_set is not None:
        unknown_sources = requested_source_set - observed_manifest_sources
        if unknown_sources:
            raise ValueError(f"Requested sources are absent from prompt manifest: {sorted(unknown_sources)}")
    if not records:
        if requested_sources is None:
            raise ValueError("Teacher cache contains no test examples")
        raise ValueError(f"No test examples match requested sources: {requested_sources}")
    selection_scope = {
        "prompt_manifest": (None if prompt_manifest is None else str(Path(prompt_manifest).resolve())),
        "prompt_manifest_sha256": manifest_sha256,
        "source_filter": requested_sources,
        "eligible_test_prompts": len(records),
    }
    return records, metadata, selection_scope


def select_evenly(records: list[tuple[str, torch.Tensor]], count: int) -> list[tuple[str, torch.Tensor]]:
    if count < 1:
        raise ValueError("num_prompts must be at least 1")
    if count >= len(records):
        return records
    indices = np.linspace(0, len(records) - 1, num=count, dtype=int)
    return [records[int(index)] for index in indices]


def validate_seeds(seeds: list[int]) -> list[tuple[int, int]]:
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for diversity normalization")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return list(combinations(seeds, 2))


def validate_basic_args(args: argparse.Namespace) -> None:
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1")
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("--duration must be finite and positive")
    if args.save_samples < 0:
        raise ValueError("--save-samples must be non-negative")
    if len(args.cfg_weight) != 2 or not all(math.isfinite(weight) for weight in args.cfg_weight):
        raise ValueError("--cfg-weight must contain two finite values")
    diffusion_steps = getattr(args, "diffusion_steps", None)
    if diffusion_steps is not None and diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be positive")
    if getattr(args, "student_dtype", DEFAULT_STUDENT_DTYPE) not in {
        "float32",
        "float16",
        "bfloat16",
    }:
        raise ValueError("--student-dtype is invalid")
    prompt_manifest = getattr(args, "prompt_manifest", None)
    sources = getattr(args, "sources", None)
    if sources is not None:
        if prompt_manifest is None:
            raise ValueError("--sources requires --prompt-manifest")
        if (
            not sources
            or not all(isinstance(source, str) and source.strip() for source in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("--sources must contain unique, non-empty source names")


def generate(
    model,
    text: str,
    text_feature: torch.Tensor,
    seed: int,
    duration: float,
    diffusion_steps: int,
    cfg_weight: tuple[float, float],
) -> tuple[dict[str, np.ndarray], float]:
    device = torch.device(model.device)
    num_frames = round(duration * model.motion_rep.fps)
    lengths = torch.tensor([num_frames], device=device)
    pad_mask = length_to_mask(lengths)
    first_heading = torch.zeros(1, device=device)
    text_feature = text_feature.to(device=device)
    text_mask = torch.ones(text_feature.shape[:2], dtype=torch.bool, device=device)
    max_window = (int(10 * model.motion_rep.fps) // model.num_frames_per_token) * model.num_frames_per_token
    history_frames = ((max_window - model.gen_horizon_len) // model.num_frames_per_token) * model.num_frames_per_token

    seed_everything(seed, deterministic=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        motion = model(
            [text],
            num_frames,
            num_denoising_steps=diffusion_steps,
            pad_mask=pad_mask,
            first_heading_angle=first_heading,
            motion_mask=None,
            observed_motion=None,
            cfg_weight=cfg_weight,
            text_feat=text_feature,
            text_pad_mask=text_mask,
            crop_history_length=history_frames,
            progress_bar=lambda values: values,
        )
        output = model.motion_rep.inverse(motion, is_normalized=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    arrays = {
        key: value[0] if hasattr(value, "shape") and value.shape[0] == 1 else value
        for key, value in to_numpy(output).items()
    }
    return arrays, elapsed


def _as_float(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _validate_same_shape(name: str, teacher: np.ndarray, student: np.ndarray) -> None:
    if teacher.shape != student.shape:
        raise ValueError(f"{name} shape mismatch: teacher={teacher.shape}, student={student.shape}")


def _macro_contact_scores(
    teacher_contacts: np.ndarray,
    student_contacts: np.ndarray,
) -> tuple[float, float]:
    f1_scores = []
    iou_scores = []
    for channel in range(teacher_contacts.shape[-1]):
        teacher_channel = teacher_contacts[..., channel]
        student_channel = student_contacts[..., channel]
        true_positive = np.count_nonzero(teacher_channel & student_channel)
        false_positive = np.count_nonzero(~teacher_channel & student_channel)
        false_negative = np.count_nonzero(teacher_channel & ~student_channel)
        positive_union = true_positive + false_positive + false_negative
        if positive_union == 0:
            # A channel with no positive event in either rollout contains no
            # positive-class contact information and is excluded from macro.
            continue
        f1_scores.append(2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative))
        iou_scores.append(true_positive / positive_union)
    return (
        float(np.mean(f1_scores)) if f1_scores else float("nan"),
        float(np.mean(iou_scores)) if iou_scores else float("nan"),
    )


def paired_metrics(teacher: dict, student: dict, *, fps: float) -> dict[str, float]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    teacher_root = _as_float(teacher["root_positions"])
    student_root = _as_float(student["root_positions"])
    teacher_joints = _as_float(teacher["posed_joints"])
    student_joints = _as_float(student["posed_joints"])
    _validate_same_shape("root_positions", teacher_root, student_root)
    _validate_same_shape("posed_joints", teacher_joints, student_joints)
    if (
        teacher_root.ndim != 2
        or teacher_root.shape[-1] != 3
        or teacher_joints.ndim != 3
        or teacher_joints.shape[0] != teacher_root.shape[0]
        or teacher_joints.shape[-1] != 3
        or teacher_root.shape[0] < 2
    ):
        raise ValueError("Expected root_positions [T,3] and posed_joints [T,J,3] with T >= 2")

    root_error = np.linalg.norm(student_root - teacher_root, axis=-1)
    joint_error = np.linalg.norm(student_joints - teacher_joints, axis=-1)
    teacher_local = teacher_joints - teacher_root[:, None, :]
    student_local = student_joints - student_root[:, None, :]
    local_joint_error = np.linalg.norm(student_local - teacher_local, axis=-1)
    teacher_velocity = np.diff(teacher_joints, axis=0) * fps
    student_velocity = np.diff(student_joints, axis=0) * fps
    velocity_error = np.linalg.norm(student_velocity - teacher_velocity, axis=-1)

    flat_teacher = torch.from_numpy(teacher_joints.reshape(1, -1))
    flat_student = torch.from_numpy(student_joints.reshape(1, -1))
    result = {
        "root_ade_m": float(root_error.mean()),
        "root_fde_m": float(root_error[-1]),
        "global_mpjpe_m": float(joint_error.mean()),
        "root_aligned_mpjpe_m": float(local_joint_error.mean()),
        "joint_velocity_error_m_per_s": float(velocity_error.mean()),
        "motion_cosine": float(F.cosine_similarity(flat_teacher, flat_student).item()),
    }

    teacher_heading = _as_float(teacher["global_root_heading"])
    student_heading = _as_float(student["global_root_heading"])
    _validate_same_shape("global_root_heading", teacher_heading, student_heading)
    if teacher_heading.shape != (teacher_root.shape[0], 2):
        raise ValueError("Expected global_root_heading [T,2]")
    valid = (np.linalg.norm(teacher_heading, axis=-1) > 1e-8) & (np.linalg.norm(student_heading, axis=-1) > 1e-8)
    if valid.any():
        dot = np.sum(teacher_heading[valid] * student_heading[valid], axis=-1)
        cross = (
            teacher_heading[valid, 0] * student_heading[valid, 1]
            - teacher_heading[valid, 1] * student_heading[valid, 0]
        )
        result["heading_mae_deg"] = float(np.degrees(np.abs(np.arctan2(cross, dot))).mean())
    else:
        result["heading_mae_deg"] = float("nan")

    if "foot_contacts" in teacher and "foot_contacts" in student:
        teacher_contacts = np.asarray(teacher["foot_contacts"]) > 0.5
        student_contacts = np.asarray(student["foot_contacts"]) > 0.5
        _validate_same_shape("foot_contacts", teacher_contacts, student_contacts)
        if teacher_contacts.ndim != 2:
            raise ValueError("Expected foot_contacts [T,C]")
        result["foot_contact_agreement"] = float(np.mean(teacher_contacts == student_contacts))
        macro_f1, macro_iou = _macro_contact_scores(teacher_contacts, student_contacts)
        result["foot_contact_macro_f1"] = macro_f1
        result["foot_contact_macro_iou"] = macro_iou
    return result


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize an empty metric list")
    keys = sorted(set().union(*(row.keys() for row in rows)) - SUMMARY_CONTEXT_KEYS)
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if len(finite) else float("nan")
    return result


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON ``null``."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_sample(path: Path, output: dict, text: str, seed: int, encoder: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = {
        key: value
        for key, value in output.items()
        if key
        in {
            "posed_joints",
            "root_positions",
            "global_root_heading",
            "foot_contacts",
            "local_rot_mats",
        }
    }
    np.savez_compressed(
        path,
        **selected,
        text=np.asarray(text),
        seed=np.asarray(seed),
        encoder=np.asarray(encoder),
    )


def main(args: argparse.Namespace) -> dict:
    determinism = configure_determinism()
    validate_basic_args(args)
    seed_pairs = validate_seeds(args.seeds)
    device = torch.device(_resolve_device(args.device))
    teacher_cache = load_teacher_cache(
        args.cache_dir,
        expected_teacher_dim=TEACHER_DIM,
        expected_target_dim=TARGET_DIM,
        keep_teacher_embeddings=True,
        keep_targets=False,
    )
    all_test_records, teacher_metadata, prompt_selection = load_test_embeddings(
        teacher_cache,
        prompt_manifest=args.prompt_manifest,
        sources=args.sources,
    )
    test_records = select_evenly(all_test_records, args.num_prompts)
    resolved_model = resolve_model_name(
        args.model,
        checkpoints_dir=args.checkpoints_dir,
    )
    student = MiniLMArdyEncoder(
        args.student_path,
        dtype=getattr(args, "student_dtype", DEFAULT_STUDENT_DTYPE),
        device=str(device),
    )
    student.assert_compatible(resolved_model)
    teacher_cache_fingerprint = validate_artifact_teacher_cache_fingerprint(
        student.artifact_config,
        teacher_cache,
    )
    model = load_model(
        resolved_model,
        device=str(device),
        text_encoder=False,
        checkpoints_dir=args.checkpoints_dir,
    )

    model_dir = Path(args.checkpoints_dir) / resolved_model
    checkpoint_path = model_dir / "denoiser.safetensors"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    student_checkpoint_sha256 = student_teacher_checkpoint_sha256(student.artifact_config)
    expected_checkpoint_hashes = {
        value
        for value in (
            teacher_metadata.get("checkpoint_sha256"),
            student_checkpoint_sha256,
        )
        if value is not None
    }
    if expected_checkpoint_hashes and expected_checkpoint_hashes != {checkpoint_sha256}:
        raise ValueError(
            "Loaded ARDY checkpoint does not match the teacher cache/student artifact: "
            f"loaded={checkpoint_sha256}, expected={sorted(expected_checkpoint_hashes)}"
        )

    num_base_steps = int(model.diffusion.num_base_steps)
    diffusion_steps = num_base_steps if args.diffusion_steps is None else args.diffusion_steps
    if not 1 <= diffusion_steps <= num_base_steps:
        raise ValueError(f"--diffusion-steps must be between 1 and {num_base_steps}; got {diffusion_steps}")
    num_frames = round(args.duration * model.motion_rep.fps)
    if num_frames < 2:
        raise ValueError(f"--duration must produce at least two frames at {model.motion_rep.fps} FPS")

    cases = []
    teacher_outputs: dict[tuple[int, int], dict] = {}
    precomputed_teacher: dict[tuple[int, int], tuple[dict, float]] = {}
    sample_dir = Path(args.sample_dir)
    selected_texts = [text for text, _embedding in test_records]
    selected_digest = selected_prompt_sha256(selected_texts)
    repeatability_enabled = getattr(args, "repeatability_check", True)
    if repeatability_enabled:
        first_text, first_embedding = test_records[0]
        first_seed = args.seeds[0]
        first_feature = first_embedding.reshape(1, 1, TEACHER_DIM)
        first_output, first_seconds = generate(
            model,
            first_text,
            first_feature,
            first_seed,
            args.duration,
            diffusion_steps,
            tuple(args.cfg_weight),
        )
        repeated_output, _repeated_seconds = generate(
            model,
            first_text,
            first_feature,
            first_seed,
            args.duration,
            diffusion_steps,
            tuple(args.cfg_weight),
        )
        compared_keys = assert_exact_repeatability(
            first_output,
            repeated_output,
        )
        precomputed_teacher[(0, first_seed)] = (
            first_output,
            first_seconds,
        )
        repeatability = {
            "enabled": True,
            "exact_equal": True,
            "prompt_index": 0,
            "seed": first_seed,
            "compared_keys": compared_keys,
        }
    else:
        repeatability = {
            "enabled": False,
            "exact_equal": None,
            "prompt_index": None,
            "seed": None,
            "compared_keys": [],
        }

    for prompt_index, (text, teacher_embedding) in enumerate(test_records):
        with torch.inference_mode():
            student_feature, _ = student([text])
        teacher_feature = teacher_embedding.reshape(1, 1, 4096)
        for seed in args.seeds:
            precomputed = precomputed_teacher.get((prompt_index, seed))
            if precomputed is None:
                teacher_output, teacher_seconds = generate(
                    model,
                    text,
                    teacher_feature,
                    seed,
                    args.duration,
                    diffusion_steps,
                    tuple(args.cfg_weight),
                )
            else:
                teacher_output, teacher_seconds = precomputed
            student_output, student_seconds = generate(
                model,
                text,
                student_feature,
                seed,
                args.duration,
                diffusion_steps,
                tuple(args.cfg_weight),
            )
            teacher_outputs[(prompt_index, seed)] = teacher_output
            metrics = paired_metrics(
                teacher_output,
                student_output,
                fps=float(model.motion_rep.fps),
            )
            cases.append(
                {
                    "prompt_index": prompt_index,
                    "text": text,
                    "seed": seed,
                    "teacher_generation_seconds": teacher_seconds,
                    "student_generation_seconds": student_seconds,
                    **metrics,
                }
            )
            print(
                f"[{len(cases)}/{len(test_records) * len(args.seeds)}] "
                f"mpjpe={metrics['global_mpjpe_m']:.4f}m "
                f"root_ade={metrics['root_ade_m']:.4f}m {text!r}",
                flush=True,
            )
            if prompt_index < args.save_samples and seed == args.seeds[0]:
                stem = f"{prompt_index:02d}-seed{seed}"
                save_sample(sample_dir / f"{stem}-teacher.npz", teacher_output, text, seed, "llm2vec")
                save_sample(sample_dir / f"{stem}-student.npz", student_output, text, seed, "minilm")

    diversity_rows = []
    for seed_a, seed_b in seed_pairs:
        for prompt_index, (text, _embedding) in enumerate(test_records):
            metrics = paired_metrics(
                teacher_outputs[(prompt_index, seed_a)],
                teacher_outputs[(prompt_index, seed_b)],
                fps=float(model.motion_rep.fps),
            )
            diversity_rows.append(
                {
                    "prompt_index": prompt_index,
                    "text": text,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    **metrics,
                }
            )

    paired_summary = mean_metrics(cases)
    diversity_summary = mean_metrics(diversity_rows)
    normalized = {}
    for metric in (
        "root_ade_m",
        "root_fde_m",
        "global_mpjpe_m",
        "root_aligned_mpjpe_m",
        "joint_velocity_error_m_per_s",
    ):
        denominator = diversity_summary.get(metric, math.nan)
        normalized[f"{metric}_vs_teacher_diversity"] = (
            paired_summary[metric] / denominator if denominator and np.isfinite(denominator) else float("nan")
        )

    report = {
        "schema_version": 1,
        "scope": {
            "split": "test",
            "requested_model": args.model,
            "resolved_model": resolved_model,
            "checkpoint_dir": str(model_dir.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "student": str(Path(args.student_path).resolve()),
            "student_artifact_fingerprint": student.artifact_config.get("artifact_fingerprint"),
            "teacher_cache": str(teacher_cache.metadata_path.parent),
            "teacher_cache_manifest_sha256": sha256_file(
                teacher_cache.metadata_path
            ),
            "teacher_cache_fingerprint": teacher_cache_fingerprint,
            "prompt_manifest": prompt_selection["prompt_manifest"],
            "prompt_manifest_sha256": prompt_selection["prompt_manifest_sha256"],
            "source_filter": prompt_selection["source_filter"],
            "prompt_selection_algorithm": PROMPT_SELECTION_ALGORITHM,
            "selected_prompt_sha256": selected_digest,
            "eligible_test_prompts": prompt_selection["eligible_test_prompts"],
            "held_out_test_prompts_available": int(teacher_metadata["split_counts"]["test"]),
            "prompts": len(test_records),
            "seeds": args.seeds,
            "teacher_diversity_seed_pairs": len(seed_pairs),
            "duration_seconds": args.duration,
            "fps": float(model.motion_rep.fps),
            "num_frames": num_frames,
            "diffusion_steps": diffusion_steps,
            "cfg_weight": args.cfg_weight,
            "student_dtype": getattr(
                args,
                "student_dtype",
                DEFAULT_STUDENT_DTYPE,
            ),
            "device_requested": args.device,
            "device_resolved": str(device),
            "determinism": determinism,
            "repeatability_check": repeatability,
            "postprocess_applied": False,
            "motion_output_stage": (
                "Raw model.motion_rep.inverse output. post_process_motion is intentionally "
                "not applied, so the metrics isolate encoder-conditioned model fidelity."
            ),
            "generation_timing_note": (
                "Per-case ARDY generation timings exclude text encoding, always run teacher "
                "first, and are diagnostic only. They are excluded from summary metrics and "
                "must not be used for encoder speed comparison."
            ),
            "metric_note": (
                "Paired fidelity to the original encoder; not paper-comparable "
                "FID/R-precision because the Rigplay evaluator is not public."
            ),
        },
        "paired_teacher_student": paired_summary,
        "teacher_seed_diversity": diversity_summary,
        "normalized_by_teacher_diversity": normalized,
        "cases": cases,
        "teacher_diversity_cases": diversity_rows,
    }
    report = json_safe(report)
    output_path = Path(args.output)
    _write_json(output_path, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"cases", "teacher_diversity_cases"}},
            indent=2,
            allow_nan=False,
        )
    )
    return report


if __name__ == "__main__":
    main(parse_args())
