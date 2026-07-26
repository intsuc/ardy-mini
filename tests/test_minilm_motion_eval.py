# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.minilm.evaluate_motion import (
    json_safe,
    load_test_embeddings,
    mean_metrics,
    paired_metrics,
    sha256_file,
    student_teacher_checkpoint_sha256,
    validate_basic_args,
    validate_seeds,
)


class MotionMetricTests(unittest.TestCase):
    def test_paired_metrics_use_heading_velocity_and_positive_contacts(self):
        teacher_root = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        student_root = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        teacher = {
            "root_positions": teacher_root,
            "posed_joints": teacher_root[:, None, :],
            "global_root_heading": np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            "foot_contacts": np.asarray([[1, 0], [0, 0]], dtype=bool),
        }
        student = {
            "root_positions": student_root,
            "posed_joints": student_root[:, None, :],
            "global_root_heading": np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            "foot_contacts": np.asarray([[1, 0], [1, 0]], dtype=bool),
        }

        metrics = paired_metrics(teacher, student, fps=20.0)

        self.assertAlmostEqual(metrics["root_ade_m"], 0.5)
        self.assertAlmostEqual(metrics["root_fde_m"], 1.0)
        self.assertAlmostEqual(metrics["global_mpjpe_m"], 0.5)
        self.assertAlmostEqual(metrics["root_aligned_mpjpe_m"], 0.0)
        self.assertAlmostEqual(metrics["joint_velocity_error_m_per_s"], 20.0)
        self.assertAlmostEqual(metrics["heading_mae_deg"], 90.0)
        self.assertAlmostEqual(metrics["foot_contact_agreement"], 0.75)
        self.assertAlmostEqual(metrics["foot_contact_macro_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["foot_contact_macro_iou"], 0.5)

    def test_seed_pairs_are_unique_and_complete(self):
        self.assertEqual(validate_seeds([2, 3, 5]), [(2, 3), (2, 5), (3, 5)])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_seeds([2, 2, 3])
        with self.assertRaisesRegex(ValueError, "At least two"):
            validate_seeds([2])

    def test_summary_excludes_generation_timings_and_context(self):
        summary = mean_metrics(
            [
                {
                    "prompt_index": 0,
                    "seed": 1,
                    "teacher_generation_seconds": 100.0,
                    "student_generation_seconds": 1.0,
                    "root_ade_m": 1.0,
                },
                {
                    "prompt_index": 1,
                    "seed": 2,
                    "teacher_generation_seconds": 200.0,
                    "student_generation_seconds": 2.0,
                    "root_ade_m": 3.0,
                },
            ]
        )
        self.assertEqual(summary, {"root_ade_m": 2.0})

    def test_json_safe_replaces_nonfinite_values(self):
        value = {
            "finite": 1.0,
            "values": [float("nan"), np.float64("inf"), np.int64(3)],
        }
        safe = json_safe(value)
        self.assertEqual(safe, {"finite": 1.0, "values": [None, None, 3]})
        json.dumps(safe, allow_nan=False)

    def test_basic_argument_validation(self):
        valid = argparse.Namespace(
            num_prompts=1,
            duration=1.0,
            save_samples=0,
            cfg_weight=[2.0, 2.0],
        )
        validate_basic_args(valid)
        for field, value in (
            ("num_prompts", 0),
            ("duration", math.nan),
            ("duration", 0.0),
            ("save_samples", -1),
            ("cfg_weight", [2.0, math.inf]),
        ):
            invalid = argparse.Namespace(**vars(valid))
            setattr(invalid, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_basic_args(invalid)

    def test_student_checkpoint_hash_supports_v2_and_legacy_metadata(self):
        checkpoint_hash = "a" * 64
        v2 = {
            "format_version": 2,
            "metadata": {"teacher_metadata": {"checkpoint_sha256": checkpoint_hash}},
        }
        legacy = {
            "format_version": 1,
            "teacher_metadata": {"checkpoint_sha256": checkpoint_hash},
        }
        self.assertEqual(
            student_teacher_checkpoint_sha256(v2),
            checkpoint_hash,
        )
        self.assertEqual(
            student_teacher_checkpoint_sha256(legacy),
            checkpoint_hash,
        )
        with self.assertRaisesRegex(TypeError, "missing teacher checkpoint"):
            student_teacher_checkpoint_sha256({"format_version": 2, "metadata": {}})


class TeacherCacheValidationTests(unittest.TestCase):
    def _write_cache(
        self,
        cache_dir: Path,
        *,
        embeddings: torch.Tensor | None = None,
        status: str = "complete",
        prompt_manifest: Path | None = None,
    ) -> tuple[Path, Path]:
        texts = ["walk forward", "jump", "turn left"]
        splits = ["train", "test", "val"]
        if embeddings is None:
            embeddings = torch.zeros(3, 4096, dtype=torch.float32)
        shard = {
            "texts": texts,
            "splits": splits,
            "teacher_embeddings": embeddings,
            "targets": torch.zeros(3, 2048, dtype=torch.float32),
        }
        shard_path = cache_dir / "teacher-00000.pt"
        torch.save(shard, shard_path)
        metadata = {
            "status": status,
            "count": 3,
            "completed_count": 3 if status == "complete" else 0,
            "teacher_dim": 4096,
            "target_dim": 2048,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "shards": [shard_path.name],
            "shard_sha256": {shard_path.name: sha256_file(shard_path)},
        }
        if prompt_manifest is not None:
            metadata["input_sha256"] = sha256_file(prompt_manifest)
        metadata_path = cache_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return shard_path, metadata_path

    def test_complete_cache_is_loaded_and_test_rows_are_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            self._write_cache(cache_dir)
            records, metadata, selection = load_test_embeddings(cache_dir)

        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "jump")
        self.assertEqual(tuple(records[0][1].shape), (4096,))
        self.assertEqual(selection["eligible_test_prompts"], 1)
        self.assertIsNone(selection["source_filter"])

    def test_incomplete_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            self._write_cache(cache_dir, status="in_progress")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_test_embeddings(cache_dir)

    def test_shard_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            shard_path, _metadata_path = self._write_cache(cache_dir)
            with shard_path.open("ab") as output_file:
                output_file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_test_embeddings(cache_dir)

    def test_nonfinite_teacher_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            embeddings = torch.zeros(3, 4096, dtype=torch.float32)
            embeddings[1, 3] = float("nan")
            self._write_cache(cache_dir, embeddings=embeddings)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_test_embeddings(cache_dir)

    def _write_prompt_manifest(
        self,
        path: Path,
        *,
        second_text: str = "jump",
    ) -> None:
        rows = [
            {
                "text": "walk forward",
                "split": "train",
                "source": "content_natural_desc_1",
            },
            {
                "text": second_text,
                "split": "test",
                "source": "content_natural_desc_2",
            },
            {
                "text": "turn left",
                "split": "val",
                "source": "content_technical_description",
            },
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_prompt_manifest_source_filter_is_lineage_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            prompt_manifest = cache_dir / "prompts.jsonl"
            self._write_prompt_manifest(prompt_manifest)
            self._write_cache(cache_dir, prompt_manifest=prompt_manifest)
            expected_manifest_sha256 = sha256_file(prompt_manifest)

            records, _metadata, selection = load_test_embeddings(
                cache_dir,
                prompt_manifest=prompt_manifest,
                sources=["content_natural_desc_2"],
            )

        self.assertEqual([text for text, _embedding in records], ["jump"])
        self.assertEqual(selection["source_filter"], ["content_natural_desc_2"])
        self.assertEqual(selection["eligible_test_prompts"], 1)
        self.assertEqual(
            selection["prompt_manifest_sha256"],
            expected_manifest_sha256,
        )

    def test_prompt_manifest_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            prompt_manifest = cache_dir / "prompts.jsonl"
            self._write_prompt_manifest(prompt_manifest)
            self._write_cache(cache_dir, prompt_manifest=prompt_manifest)
            prompt_manifest.write_text(
                prompt_manifest.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_test_embeddings(cache_dir, prompt_manifest=prompt_manifest)

    def test_prompt_manifest_cache_row_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            prompt_manifest = cache_dir / "prompts.jsonl"
            self._write_prompt_manifest(prompt_manifest, second_text="leap")
            self._write_cache(cache_dir, prompt_manifest=prompt_manifest)

            with self.assertRaisesRegex(ValueError, "text/split"):
                load_test_embeddings(cache_dir, prompt_manifest=prompt_manifest)

    def test_sources_require_manifest_and_known_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            self._write_cache(cache_dir)
            with self.assertRaisesRegex(ValueError, "requires"):
                load_test_embeddings(
                    cache_dir,
                    sources=["content_natural_desc_2"],
                )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_test_embeddings(
                    cache_dir,
                    prompt_manifest=cache_dir / "missing.jsonl",
                    sources=["duplicate", "duplicate"],
                )

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            prompt_manifest = cache_dir / "prompts.jsonl"
            self._write_prompt_manifest(prompt_manifest)
            self._write_cache(cache_dir, prompt_manifest=prompt_manifest)
            with self.assertRaisesRegex(ValueError, "absent"):
                load_test_embeddings(
                    cache_dir,
                    prompt_manifest=prompt_manifest,
                    sources=["does_not_exist"],
                )


if __name__ == "__main__":
    unittest.main()
