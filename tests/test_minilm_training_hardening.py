# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for MiniLM training and teacher-cache validation."""

from __future__ import annotations

import argparse
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ardy.minilm_teacher_cache import load_teacher_cache, sha256_file
from scripts.minilm import evaluate_conditions
from scripts.minilm.train import (
    CachedExamples,
    distillation_loss,
    load_cached_examples,
    resolve_and_validate_teacher_checkpoint,
    validate_frozen_evaluation_cache,
    validate_training_args,
)


class _TeacherCacheFixture:
    @staticmethod
    def write(directory: Path) -> tuple[Path, Path, dict]:
        checkpoint_dir = directory / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "denoiser.safetensors"
        checkpoint_path.write_bytes(b"tiny checkpoint")

        shard_path = directory / "teacher-00000.pt"
        torch.save(
            {
                "texts": ["walk", "jump", "turn"],
                "splits": ["train", "val", "test"],
                "teacher_embeddings": torch.zeros(3, 4096, dtype=torch.float32),
                "targets": torch.zeros(3, 2048, dtype=torch.float32),
            },
            shard_path,
        )
        metadata = {
            "format_version": 1,
            "status": "complete",
            "count": 3,
            "completed_count": 3,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "shard_size": 3,
            "shards": [shard_path.name],
            "shard_sha256": {shard_path.name: sha256_file(shard_path)},
            "teacher_dim": 4096,
            "target_dim": 2048,
            "target_order": ["root", "body"],
            "bias_applied": False,
            "dtype": {
                "teacher_model": "bfloat16",
                "teacher_embeddings": "float32",
                "projection_weights": "float32",
                "targets": "float32",
            },
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return shard_path, metadata_path, metadata

    @staticmethod
    def rewrite_metadata(path: Path, metadata: dict) -> None:
        path.write_text(json.dumps(metadata), encoding="utf-8")

    @staticmethod
    def rewrite_shard(
        shard_path: Path,
        metadata_path: Path,
        metadata: dict,
        payload: dict,
    ) -> None:
        torch.save(payload, shard_path)
        metadata["shard_sha256"][shard_path.name] = sha256_file(shard_path)
        _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)


class TeacherCacheHardeningTests(unittest.TestCase):
    def test_valid_cache_loads_and_checkpoint_matches_resolved_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _shard, _metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
            cache = load_teacher_cache(cache_dir)
            examples, loaded_metadata = load_cached_examples(cache_dir)
            resolved, checkpoint = resolve_and_validate_teacher_checkpoint(
                "core",
                metadata,
            )

        self.assertEqual(resolved, "ARDY-Core-RP-20FPS-Horizon40")
        self.assertEqual(checkpoint.name, "denoiser.safetensors")
        self.assertEqual(len(cache.shards), 1)
        self.assertEqual(examples.splits, ["train", "val", "test"])
        self.assertTrue(torch.isfinite(examples.targets).all())
        self.assertEqual(loaded_metadata, metadata)

    def test_incomplete_count_split_and_hash_fail_closed(self) -> None:
        mutations = (
            ("status", "in_progress", "incomplete"),
            ("completed_count", 2, "incomplete"),
            ("count", 4, "incomplete"),
            (
                "split_counts",
                {"train": 2, "val": 0, "test": 1},
                "observed split counts",
            ),
            (
                "shard_sha256",
                {"teacher-00000.pt": "0" * 64},
                "SHA-256 mismatch",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory)
                _shard, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
                metadata[field] = value
                _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
                with self.assertRaisesRegex(ValueError, message):
                    load_teacher_cache(cache_dir)

    def test_shape_dtype_and_finite_values_fail_closed(self) -> None:
        mutations = {
            "shape": lambda payload: payload.update(targets=torch.zeros(3, 2047, dtype=torch.float32)),
            "dtype": lambda payload: payload.update(teacher_embeddings=torch.zeros(3, 4096, dtype=torch.float16)),
            "finite": lambda payload: payload["targets"].__setitem__(
                (0, 0),
                float("nan"),
            ),
        }
        messages = {
            "shape": "must have shape",
            "dtype": "must be float32",
            "finite": "non-finite",
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory)
                shard_path, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
                payload = torch.load(
                    shard_path,
                    map_location="cpu",
                    weights_only=True,
                )
                mutate(payload)
                _TeacherCacheFixture.rewrite_shard(
                    shard_path,
                    metadata_path,
                    metadata,
                    payload,
                )
                with self.assertRaisesRegex(ValueError, messages[name]):
                    load_teacher_cache(cache_dir)

    def test_manifest_must_match_all_on_disk_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _TeacherCacheFixture.write(cache_dir)
            (cache_dir / "teacher-00001.pt").write_bytes(b"unmanifested")
            with self.assertRaisesRegex(ValueError, "do not exactly match manifest"):
                load_teacher_cache(cache_dir)

    def test_condition_evaluator_consumes_the_strict_cache(self) -> None:
        class _FakeEncoder:
            def __init__(self, **_kwargs) -> None:
                self.artifact_config = {
                    "format_version": 2,
                    "artifact_fingerprint": "a" * 64,
                    "base_model": "tiny",
                    "condition_dim": 1024,
                    "output_dim": 2048,
                    "compatible_ardy_models": ["ARDY-Core-RP-20FPS-Horizon40"],
                    "max_length": 16,
                }

            def __call__(self, texts: list[str]):
                return torch.zeros(len(texts), 1, 2048), [1] * len(texts)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _TeacherCacheFixture.write(cache_dir)
            output_path = cache_dir / "conditions.json"
            args = SimpleNamespace(
                teacher_cache=cache_dir,
                student_path=cache_dir / "student",
                output=output_path,
                split="test",
                device="cpu",
                dtype="float32",
                batch_size=2,
                max_samples=None,
                teacher_dim=4096,
                expected_ardy_model="ARDY-Core-RP-20FPS-Horizon40",
                max_length=None,
            )
            with (
                patch.object(
                    evaluate_conditions,
                    "MiniLMArdyEncoder",
                    _FakeEncoder,
                ),
                redirect_stdout(io.StringIO()),
            ):
                evaluate_conditions.run(args)
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["dataset"]["evaluated_examples"], 1)
        self.assertEqual(report["metrics"]["overall"]["examples"], 1)
        self.assertEqual(len(report["dataset"]["cache_fingerprint"]), 64)


class TrainingValidationTests(unittest.TestCase):
    @staticmethod
    def valid_args() -> argparse.Namespace:
        return argparse.Namespace(
            cache_dir="cache",
            eval_cache_dir=None,
            output_dir="artifact",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            ardy_model="core",
            epochs=2,
            head_warmup_epochs=0,
            batch_size=4,
            adapter_dim=16,
            pooling_mode="mean",
            backbone_lr=1e-4,
            head_lr=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            cosine_weight=0.5,
            relational_weight=0.05,
            train_max_length=32,
            runtime_max_length=64,
            num_workers=0,
            seed=1,
            device="cpu",
            no_bf16=True,
        )

    def test_frozen_evaluation_cache_requires_matching_identity_and_text(self) -> None:
        training = CachedExamples(
            texts=["new train", "validation", "test"],
            splits=["train", "val", "test"],
            targets=torch.zeros(3, 2048),
        )
        evaluation = CachedExamples(
            texts=["old train", "validation", "test"],
            splits=["train", "val", "test"],
            targets=torch.zeros(3, 2048),
        )
        metadata = {
            "base_model_name_or_path": "teacher-base",
            "peft_model_name_or_path": "teacher-adapter",
            "checkpoint_sha256": "a" * 64,
            "target_keys": ["root", "body"],
            "target_order": ["root", "body"],
            "bias_applied": False,
            "teacher_dim": 4096,
            "target_dim": 2048,
            "dtype": {"targets": "float32"},
            "model_revisions": {"base": "revision"},
        }

        validate_frozen_evaluation_cache(
            training,
            metadata,
            evaluation,
            dict(metadata),
        )

        changed_identity = dict(metadata, checkpoint_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "different teacher identities"):
            validate_frozen_evaluation_cache(
                training,
                metadata,
                evaluation,
                changed_identity,
            )

        changed_evaluation = CachedExamples(
            texts=["old train", "different validation", "test"],
            splits=["train", "val", "test"],
            targets=torch.zeros(3, 2048),
        )
        with self.assertRaisesRegex(ValueError, "val prompt text/order"):
            validate_frozen_evaluation_cache(
                training,
                metadata,
                changed_evaluation,
                dict(metadata),
            )

        leaked_training = CachedExamples(
            texts=["validation", "validation", "test"],
            splits=["train", "val", "test"],
            targets=torch.zeros(3, 2048),
        )
        with self.assertRaisesRegex(ValueError, "overlap frozen-evaluation val"):
            validate_frozen_evaluation_cache(
                leaked_training,
                metadata,
                evaluation,
                dict(metadata),
            )

    def test_training_arguments_are_validated(self) -> None:
        validate_training_args(self.valid_args())
        invalid_values = (
            ("epochs", 0),
            ("head_warmup_epochs", 3),
            ("batch_size", -1),
            ("backbone_lr", math.nan),
            ("head_lr", 0.0),
            ("weight_decay", -0.1),
            ("warmup_ratio", math.inf),
            ("cosine_weight", -1.0),
            ("runtime_max_length", 0),
            ("num_workers", -1),
            ("base_model", ""),
        )
        for field, value in invalid_values:
            args = self.valid_args()
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_args(args)

    def test_cosine_loss_is_root_body_macro_average(self) -> None:
        target = torch.ones(2, 2048)
        prediction = target.clone()
        prediction[:, 1024:] = -1.0
        _loss, log = distillation_loss(
            prediction,
            target,
            torch.ones(2048),
            cosine_weight=1.0,
            relational_weight=0.0,
        )
        self.assertAlmostEqual(log["root_cosine_loss"], 0.0)
        self.assertAlmostEqual(log["body_cosine_loss"], 2.0)
        self.assertAlmostEqual(log["cosine_loss"], 1.0)

    def test_checkpoint_folder_and_hash_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _shard, _metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
            with self.assertRaisesRegex(ValueError, "checkpoint folder"):
                resolve_and_validate_teacher_checkpoint("g1", metadata)

            metadata["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_and_validate_teacher_checkpoint("core", metadata)


if __name__ == "__main__":
    unittest.main()
