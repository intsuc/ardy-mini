#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Train the all-MiniLM-L6-v2 ARDY root/body condition student.

The input directory is produced by ``cache_teacher.py``. Each shard contains
production LLM2Vec embeddings and the checkpoint-specific, bias-free
``[W_root @ e, W_body @ e]`` target. The resulting artifact is self-contained
for inference but remains specific to the ARDY checkpoint named in its
metadata.

Example:

    uv run python scripts/minilm/train.py \
      --cache-dir artifacts/teacher-core40 \
      --output-dir artifacts/minilm-ardy-core40
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ardy.minilm_teacher_cache import load_teacher_cache, sha256_file
from ardy.model.minilm_encoder import (
    ARDY_CONDITION_DIM,
    POOLING_MODES,
    MotionConditionStudent,
)
from ardy.model.registry import resolve_model_name

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ARDY_MODEL = "ARDY-Core-RP-20FPS-Horizon40"


@dataclass
class CachedExamples:
    texts: list[str]
    splits: list[str]
    targets: torch.Tensor


class ConditionDataset(Dataset):
    def __init__(self, examples: CachedExamples, split: str) -> None:
        indices = [index for index, value in enumerate(examples.splits) if value == split]
        if not indices:
            raise ValueError(f"No {split!r} examples found in teacher cache")
        self.texts = [examples.texts[index] for index in indices]
        self.targets = examples.targets[indices].float()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.targets[index]


class TokenizingCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        texts, targets = zip(*rows)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded, torch.stack(targets), list(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Teacher-cache directory")
    parser.add_argument(
        "--eval-cache-dir",
        default=None,
        help=(
            "optional teacher cache supplying byte-for-byte frozen validation "
            "and test targets while --cache-dir supplies training examples"
        ),
    )
    parser.add_argument("--output-dir", default="artifacts/minilm-ardy-core40")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--ardy-model", default=DEFAULT_ARDY_MODEL)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adapter-dim", type=int, default=768)
    parser.add_argument(
        "--pooling-mode",
        choices=POOLING_MODES,
        default="mean",
        help="Sentence-level pooling over MiniLM token outputs",
    )
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--cosine-weight", type=float, default=0.10)
    parser.add_argument("--relational-weight", type=float, default=0.02)
    parser.add_argument("--train-max-length", type=int, default=128)
    parser.add_argument("--runtime-max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-bf16", action="store_true", help="Disable CUDA BF16 autocast")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_examples(cache_dir: str | Path) -> tuple[CachedExamples, dict]:
    cache = load_teacher_cache(
        cache_dir,
        keep_teacher_embeddings=False,
        keep_targets=True,
    )
    targets = [shard.targets for shard in cache.shards]
    if any(target is None for target in targets):
        raise RuntimeError("strict teacher-cache loader did not retain targets")
    examples = CachedExamples(
        texts=[text for shard in cache.shards for text in shard.texts],
        splits=[split for shard in cache.shards for split in shard.splits],
        targets=torch.cat([target for target in targets if target is not None]),
    )
    if not torch.isfinite(examples.targets).all():
        raise ValueError("Teacher cache targets contain non-finite values")
    return examples, cache.metadata


def validate_training_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "epochs",
        "batch_size",
        "adapter_dim",
        "train_max_length",
        "runtime_max_length",
    )
    for field in positive_int_fields:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be a positive integer")
    if (
        isinstance(args.head_warmup_epochs, bool)
        or not isinstance(args.head_warmup_epochs, int)
        or not 0 <= args.head_warmup_epochs <= args.epochs
    ):
        raise ValueError("--head-warmup-epochs must be between zero and --epochs")
    if isinstance(args.num_workers, bool) or not isinstance(args.num_workers, int) or args.num_workers < 0:
        raise ValueError("--num-workers must be a non-negative integer")

    positive_float_fields = ("backbone_lr", "head_lr")
    for field in positive_float_fields:
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be finite and positive")
    nonnegative_float_fields = (
        "weight_decay",
        "cosine_weight",
        "relational_weight",
    )
    for field in nonnegative_float_fields:
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{field.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.warmup_ratio) or not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be finite and between zero and one")
    if not isinstance(args.base_model, str) or not args.base_model.strip():
        raise ValueError("--base-model must be a non-empty string")
    if not isinstance(args.ardy_model, str) or not args.ardy_model.strip():
        raise ValueError("--ardy-model must be a non-empty string")
    for field in ("cache_dir", "output_dir"):
        value = getattr(args, field)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"--{field.replace('_', '-')} must be a non-empty path")
    if args.eval_cache_dir is not None and (
        not isinstance(args.eval_cache_dir, (str, Path)) or not str(args.eval_cache_dir).strip()
    ):
        raise ValueError("--eval-cache-dir must be a non-empty path when provided")
    if args.pooling_mode not in POOLING_MODES:
        raise ValueError(f"--pooling-mode must be one of {POOLING_MODES}")
    torch.device(args.device)


def resolve_and_validate_teacher_checkpoint(
    ardy_model: str,
    teacher_metadata: dict,
) -> tuple[str, Path]:
    resolved_model = resolve_model_name(ardy_model)
    checkpoint_value = teacher_metadata.get("checkpoint_path")
    checkpoint_hash = teacher_metadata.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Teacher metadata checkpoint_path must be a non-empty string")
    checkpoint_path = Path(checkpoint_value).resolve()
    if checkpoint_path.parent.name != resolved_model:
        raise ValueError(
            f"--ardy-model resolves to {resolved_model!r}, but teacher targets were "
            f"projected with checkpoint folder {checkpoint_path.parent.name!r}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_path}")
    actual_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != actual_hash:
        raise ValueError(
            "Teacher metadata checkpoint SHA-256 does not match the checkpoint file: "
            f"metadata={checkpoint_hash}, actual={actual_hash}"
        )
    return resolved_model, checkpoint_path


def validate_frozen_evaluation_cache(
    training_examples: CachedExamples,
    training_metadata: dict,
    evaluation_examples: CachedExamples,
    evaluation_metadata: dict,
) -> None:
    """Require an evaluation cache with identical teacher identity and val/test text."""

    identity_fields = (
        "base_model_name_or_path",
        "peft_model_name_or_path",
        "checkpoint_sha256",
        "target_keys",
        "target_order",
        "bias_applied",
        "teacher_dim",
        "target_dim",
        "dtype",
        "model_revisions",
    )
    mismatches = [field for field in identity_fields if training_metadata.get(field) != evaluation_metadata.get(field)]
    if mismatches:
        raise ValueError(
            f"training and frozen-evaluation teacher caches have different teacher identities: {mismatches}"
        )

    training_prompt_texts = {
        text
        for text, split in zip(
            training_examples.texts,
            training_examples.splits,
            strict=True,
        )
        if split == "train"
    }
    for split in ("val", "test"):
        training_texts = [
            text
            for text, row_split in zip(
                training_examples.texts,
                training_examples.splits,
                strict=True,
            )
            if row_split == split
        ]
        evaluation_texts = [
            text
            for text, row_split in zip(
                evaluation_examples.texts,
                evaluation_examples.splits,
                strict=True,
            )
            if row_split == split
        ]
        if training_texts != evaluation_texts:
            raise ValueError(f"frozen-evaluation {split} prompt text/order does not match the training cache manifest")
        if len(evaluation_texts) != len(set(evaluation_texts)):
            raise ValueError(f"frozen-evaluation {split} contains duplicate prompt text")
        overlap = training_prompt_texts.intersection(evaluation_texts)
        if overlap:
            raise ValueError(f"training prompts overlap frozen-evaluation {split}: {len(overlap)} prompt(s)")


def make_loader(
    dataset: ConditionDataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TokenizingCollator(tokenizer, max_length),
        generator=generator,
    )


def move_tokens(tokens: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def relational_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[0] < 2:
        return prediction.new_zeros(())
    prediction = F.normalize(prediction, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return F.smooth_l1_loss(prediction @ prediction.T, target @ target.T)


def distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_scale: torch.Tensor,
    cosine_weight: float,
    relational_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    expected_dim = 2 * ARDY_CONDITION_DIM
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != expected_dim:
        raise ValueError(
            f"distillation tensors must share shape [N, {expected_dim}], got "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
        )
    if target_scale.shape != (expected_dim,):
        raise ValueError(f"target_scale must have shape [{expected_dim}], got {tuple(target_scale.shape)}")
    scaled_error = (prediction - target) / target_scale
    regression = F.smooth_l1_loss(
        scaled_error,
        torch.zeros_like(scaled_error),
        beta=0.5,
    )
    root_cosine = (
        1.0
        - F.cosine_similarity(
            prediction[:, :ARDY_CONDITION_DIM],
            target[:, :ARDY_CONDITION_DIM],
            dim=-1,
        ).mean()
    )
    body_cosine = (
        1.0
        - F.cosine_similarity(
            prediction[:, ARDY_CONDITION_DIM:],
            target[:, ARDY_CONDITION_DIM:],
            dim=-1,
        ).mean()
    )
    cosine = 0.5 * (root_cosine + body_cosine)
    relation = relational_loss(prediction, target)
    total = regression + cosine_weight * cosine + relational_weight * relation
    return total, {
        "loss": float(total.detach()),
        "regression": float(regression.detach()),
        "cosine_loss": float(cosine.detach()),
        "root_cosine_loss": float(root_cosine.detach()),
        "body_cosine_loss": float(body_cosine.detach()),
        "relational_loss": float(relation.detach()),
    }


def condition_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, part in (
        ("root", slice(0, ARDY_CONDITION_DIM)),
        ("body", slice(ARDY_CONDITION_DIM, 2 * ARDY_CONDITION_DIM)),
    ):
        pred_part = prediction[:, part]
        target_part = target[:, part]
        result[f"{name}_cosine"] = float(F.cosine_similarity(pred_part, target_part, dim=-1).mean())
        result[f"{name}_rmse"] = float(torch.sqrt(F.mse_loss(pred_part, target_part)))
        result[f"{name}_mae"] = float(F.l1_loss(pred_part, target_part))
        target_rms = torch.sqrt(torch.mean(target_part.square())).clamp_min(1e-9)
        result[f"{name}_nrmse"] = result[f"{name}_rmse"] / float(target_rms)
    return result


@torch.inference_mode()
def evaluate(
    model: MotionConditionStudent,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for tokens, target, _texts in loader:
        tokens = move_tokens(tokens, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = model(**tokens)
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
    return condition_metrics(torch.cat(predictions), torch.cat(targets))


def average_logs(logs: Iterable[dict[str, float]]) -> dict[str, float]:
    logs = list(logs)
    return {key: sum(item[key] for item in logs) / len(logs) for key in logs[0]}


def train(args: argparse.Namespace) -> dict:
    validate_training_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_bf16 = device.type == "cuda" and not args.no_bf16

    examples, teacher_metadata = load_cached_examples(args.cache_dir)
    evaluation_examples = examples
    evaluation_metadata = teacher_metadata
    if args.eval_cache_dir is not None:
        evaluation_examples, evaluation_metadata = load_cached_examples(args.eval_cache_dir)
        validate_frozen_evaluation_cache(
            examples,
            teacher_metadata,
            evaluation_examples,
            evaluation_metadata,
        )
    resolved_ardy_model, teacher_checkpoint = resolve_and_validate_teacher_checkpoint(
        args.ardy_model,
        teacher_metadata,
    )
    train_dataset = ConditionDataset(examples, "train")
    val_dataset = ConditionDataset(evaluation_examples, "val")
    test_dataset = ConditionDataset(evaluation_examples, "test")
    print(
        f"Loaded teacher cache: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = MotionConditionStudent.from_base_model(
        args.base_model,
        adapter_dim=args.adapter_dim,
        condition_dim=ARDY_CONDITION_DIM,
        normalize_embedding=True,
        pooling_mode=args.pooling_mode,
    ).to(device)

    train_loader = make_loader(
        train_dataset,
        tokenizer,
        args.batch_size,
        args.train_max_length,
        True,
        args.num_workers,
        args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        tokenizer,
        args.batch_size,
        args.train_max_length,
        False,
        args.num_workers,
        args.seed,
    )
    test_loader = make_loader(
        test_dataset,
        tokenizer,
        args.batch_size,
        args.train_max_length,
        False,
        args.num_workers,
        args.seed,
    )

    target_std = train_dataset.targets.std(dim=0, unbiased=False)
    if not torch.isfinite(target_std).all():
        raise ValueError("Training target scale contains non-finite values")
    std_floor = float(target_std.median()) * 0.05
    target_scale = target_std.clamp_min(max(std_floor, 1e-6)).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = [
        *model.adapter.parameters(),
        *model.root_head.parameters(),
        *model.body_head.parameters(),
    ]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(args.epochs):
        backbone_trainable = epoch >= args.head_warmup_epochs
        for parameter in backbone_params:
            parameter.requires_grad = backbone_trainable

        model.train()
        epoch_logs = []
        epoch_start = time.perf_counter()
        for step, (tokens, target, _texts) in enumerate(train_loader, start=1):
            tokens = move_tokens(tokens, device)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                prediction = model(**tokens)
                loss, log = distillation_loss(
                    prediction.float(),
                    target.float(),
                    target_scale,
                    args.cosine_weight,
                    args.relational_weight,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_logs.append(log)
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} loss={log['loss']:.5f}",
                    flush=True,
                )

        train_log = average_logs(epoch_logs)
        val_metrics = evaluate(model, val_loader, device, use_bf16)
        score = 0.5 * (val_metrics["root_cosine"] + val_metrics["body_cosine"])
        epoch_result = {
            "epoch": epoch + 1,
            "backbone_trainable": backbone_trainable,
            "seconds": time.perf_counter() - epoch_start,
            "train": train_log,
            "validation": val_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = evaluate(model, val_loader, device, use_bf16)
    test_metrics = evaluate(model, test_loader, device, use_bf16)
    elapsed = time.perf_counter() - started

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target_definition": "[W_root @ LLM2Vec(prompt), W_body @ LLM2Vec(prompt)] (bias excluded)",
        "teacher_cache": str(Path(args.cache_dir).resolve()),
        "evaluation_teacher_cache": (None if args.eval_cache_dir is None else str(Path(args.eval_cache_dir).resolve())),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_metadata": teacher_metadata,
        "evaluation_teacher_metadata": evaluation_metadata,
        "training": {
            "seed": args.seed,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "test_examples": len(test_dataset),
            "epochs": args.epochs,
            "head_warmup_epochs": args.head_warmup_epochs,
            "batch_size": args.batch_size,
            "adapter_dim": args.adapter_dim,
            "pooling_mode": args.pooling_mode,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "cosine_weight": args.cosine_weight,
            "relational_weight": args.relational_weight,
            "train_max_length": args.train_max_length,
            "runtime_max_length": args.runtime_max_length,
            "num_workers": args.num_workers,
            "device": str(device),
            "bf16_autocast": use_bf16,
            "train_batches_per_epoch": len(train_loader),
            "optimizer_updates": total_steps,
            "elapsed_seconds": elapsed,
            "base_model": args.base_model,
            "ardy_model": resolved_ardy_model,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "provenance_notice": (
            "Derived using ARDY checkpoint projections and LLM2Vec teacher outputs. "
            "Review NVIDIA Open Model, Meta Llama, LLM2Vec, and source-corpus licenses "
            "before redistribution."
        ),
    }
    model.save_artifact(
        output_dir,
        tokenizer,
        metadata,
        base_model_name_or_path=args.base_model,
        compatible_ardy_models=[resolved_ardy_model],
        max_length=args.runtime_max_length,
    )
    training_report = {
        "artifact": str(output_dir.resolve()),
        "base_model": args.base_model,
        "ardy_model": resolved_ardy_model,
        "best_validation_score": best_score,
        "validation": validation_metrics,
        "test": test_metrics,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(training_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(training_report, indent=2, sort_keys=True), flush=True)
    return training_report


if __name__ == "__main__":
    train(parse_args())
