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

from ardy.minilm_teacher_cache import (
    TIMELINE_PROMPT_DEDUPLICATION,
    TIMELINE_PROMPT_GROUPING,
    TIMELINE_PROMPT_MAX_CHARACTERS,
    TIMELINE_PROMPT_NORMALIZATION,
    TIMELINE_SPLIT_HASH_NAMESPACE,
    prompt_provenance_sha256,
)
from scripts.minilm.evaluate_motion import (
    assert_exact_repeatability,
    json_safe,
    load_test_embeddings,
    mean_metrics,
    paired_metrics,
    selected_prompt_sha256,
    sha256_file,
    student_teacher_checkpoint_sha256,
    validate_basic_args,
    validate_seeds,
)


def _prompt_provenance(manifest_sha256: str) -> dict:
    return {
        "format": "ardy-minilm-prompt-provenance",
        "format_version": 1,
        "dataset": {
            "repo_id": "nvidia/SEED-Timeline-Annotations",
            "revision": "b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f",
            "filename": "timelines.jsonl",
            "sha256": "379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8",
            "size_bytes": 80_373_523,
            "resolved_from": "hugging_face_hub",
            "owner": "NVIDIA",
            "license": "CC BY 4.0",
            "url": "https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations",
        },
        "preparation": {
            "sources": ["overview_description", "events.description"],
            "normalization": TIMELINE_PROMPT_NORMALIZATION,
            "deduplication": TIMELINE_PROMPT_DEDUPLICATION,
            "max_prompt_characters": TIMELINE_PROMPT_MAX_CHARACTERS,
            "grouping": TIMELINE_PROMPT_GROUPING,
            "split_hash_namespace": TIMELINE_SPLIT_HASH_NAMESPACE,
            "seed": 20260726,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "sample_size": 0,
        },
        "counts": {
            "timeline_rows": 3,
            "recording_families": 3,
            "recording_components": 3,
            "missing_propagation_references": 0,
            "raw_overview_descriptions": 3,
            "raw_event_descriptions": 3,
            "dropped_prompt_too_long": 0,
            "unique_descriptions": 3,
            "unique_before_sampling": 3,
            "written": 3,
            "groups_written": 3,
            "splits": {"train": 1, "val": 1, "test": 1},
            "split_groups": {"train": 1, "val": 1, "test": 1},
            "sources": {"overview_description": 2, "events.description": 1},
        },
        "manifest": {
            "filename": "prompts.jsonl",
            "sha256": manifest_sha256,
        },
    }


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

    def test_selected_prompt_digest_is_order_sensitive(self):
        forward = selected_prompt_sha256(["walk", "turn"])
        reverse = selected_prompt_sha256(["turn", "walk"])

        self.assertEqual(len(forward), 64)
        self.assertNotEqual(forward, reverse)

    def test_repeatability_requires_exact_motion_arrays(self):
        output = {
            "root_positions": np.zeros((2, 3), dtype=np.float32),
            "posed_joints": np.zeros((2, 1, 3), dtype=np.float32),
            "global_root_heading": np.zeros((2, 2), dtype=np.float32),
            "foot_contacts": np.zeros((2, 2), dtype=np.float32),
        }
        self.assertEqual(
            assert_exact_repeatability(output, output),
            [
                "root_positions",
                "posed_joints",
                "global_root_heading",
                "foot_contacts",
            ],
        )
        changed = {key: value.copy() for key, value in output.items()}
        changed["root_positions"][0, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "arrays differ"):
            assert_exact_repeatability(output, changed)

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
            ("diffusion_steps", 0),
            ("student_dtype", "float64"),
        ):
            invalid = argparse.Namespace(**vars(valid))
            setattr(invalid, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_basic_args(invalid)

    def test_student_checkpoint_hash_requires_canonical_cache_lineage(self):
        checkpoint_hash = "a" * 64
        artifact = {
            "format_version": 2,
            "metadata": {"teacher_cache_lineage": {"checkpoint_sha256": checkpoint_hash}},
        }
        self.assertEqual(
            student_teacher_checkpoint_sha256(artifact),
            checkpoint_hash,
        )
        with self.assertRaisesRegex(TypeError, "missing teacher-cache lineage"):
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
        input_sha256 = sha256_file(prompt_manifest) if prompt_manifest is not None else "1" * 64
        corpus_provenance = _prompt_provenance(input_sha256)
        metadata = {
            "format_version": 3,
            "input_path": str(cache_dir / "prompts.jsonl"),
            "input_sha256": input_sha256,
            "input_metadata_sha256": prompt_provenance_sha256(corpus_provenance),
            "corpus_provenance": corpus_provenance,
            "model_name": (
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp + "
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            ),
            "foundation_model_name_or_path": "meta-llama/Meta-Llama-3-8B",
            "foundation_model_revision": "a" * 40,
            "base_model_name_or_path": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
            "base_model_revision": "b" * 40,
            "peft_model_name_or_path": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
            "peft_model_revision": "c" * 40,
            "model_revisions": {
                "foundation_model": "meta-llama/Meta-Llama-3-8B",
                "foundation": "a" * 40,
                "base": "b" * 40,
                "peft": "c" * 40,
            },
            "versions": {
                "torch": "2.9.1",
                "transformers": "5.1.0",
                "peft": "0.18.1",
                "safetensors": "0.7.0",
            },
            "device": {
                "cli_requested": "cpu",
                "env_override": None,
                "requested": "cpu",
                "resolved": "cpu",
            },
            "provenance_status": "recorded",
            "status": status,
            "count": 3,
            "completed_count": 3 if status == "complete" else 0,
            "teacher_dim": 4096,
            "target_dim": 2048,
            "target_keys": [
                "denoiser.backbone.root_model.embed_text.weight",
                "denoiser.backbone.body_model.embed_text.weight",
            ],
            "target_order": ["root", "body"],
            "bias_applied": False,
            "dtype": {
                "teacher_model": "bfloat16",
                "teacher_embeddings": "float32",
                "projection_weights": "float32",
                "targets": "float32",
            },
            "teacher_batch_size": 1,
            "checkpoint_path": str(cache_dir / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40" / "denoiser.safetensors"),
            "checkpoint_sha256": "2" * 64,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "shard_size": 3,
            "shards": [shard_path.name],
            "shard_sha256": {shard_path.name: sha256_file(shard_path)},
        }
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
                "source": "overview_description",
            },
            {
                "text": second_text,
                "split": "test",
                "source": "events.description",
            },
            {
                "text": "turn left",
                "split": "val",
                "source": "overview_description",
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
                sources=["events.description"],
            )

        self.assertEqual([text for text, _embedding in records], ["jump"])
        self.assertEqual(selection["source_filter"], ["events.description"])
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
                    sources=["events.description"],
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
