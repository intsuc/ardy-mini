# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for MiniLM training and teacher-cache validation."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from ardy.minilm_teacher_cache import (
    TIMELINE_PROMPT_DEDUPLICATION,
    TIMELINE_PROMPT_GROUPING,
    TIMELINE_PROMPT_MAX_CHARACTERS,
    TIMELINE_PROMPT_NORMALIZATION,
    TIMELINE_SPLIT_HASH_NAMESPACE,
    load_teacher_cache,
    prompt_provenance_sha256,
    sha256_file,
    teacher_cache_fingerprint,
    validate_artifact_teacher_cache_fingerprint,
    validated_teacher_lineage,
)
from scripts.minilm import cache_teacher, evaluate_conditions
from scripts.minilm import train as train_module
from scripts.minilm.cache_teacher import read_prompt_provenance, read_prompts
from scripts.minilm.train import (
    CachedExamples,
    cosine_lr_multiplier,
    cuda_supports_bf16,
    distillation_loss,
    load_cached_examples,
    require_fresh_output_dir,
    resolve_and_validate_teacher_checkpoint,
    validate_training_args,
)


def _prompt_provenance(manifest_sha256: str = "1" * 64) -> dict:
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
        corpus_provenance = _prompt_provenance()
        metadata = {
            "format_version": 3,
            "input_path": str(directory / "prompts.jsonl"),
            "input_sha256": "1" * 64,
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
            "status": "complete",
            "count": 3,
            "completed_count": 3,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "shard_size": 3,
            "shards": [shard_path.name],
            "shard_sha256": {shard_path.name: sha256_file(shard_path)},
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


class PromptProvenanceInputTests(unittest.TestCase):
    @staticmethod
    def write(directory: Path) -> tuple[Path, Path, dict, str]:
        manifest_path = directory / "prompts.jsonl"
        manifest_path.write_text(
            '{"text":"walk","split":"train"}\n{"text":"jump","split":"val"}\n{"text":"turn","split":"test"}\n',
            encoding="utf-8",
        )
        manifest_sha256 = sha256_file(manifest_path)
        provenance = _prompt_provenance(manifest_sha256)
        metadata_path = directory / "prompts.metadata.json"
        metadata_path.write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path, metadata_path, provenance, manifest_sha256

    def test_valid_sidecar_is_loaded_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, metadata_path, provenance, manifest_sha256 = self.write(Path(directory))
            loaded, metadata_sha256 = read_prompt_provenance(
                metadata_path,
                input_path=manifest_path,
                input_sha256=manifest_sha256,
                count=3,
                split_counts={"train": 1, "val": 1, "test": 1},
            )

        self.assertEqual(loaded, provenance)
        self.assertEqual(metadata_sha256, prompt_provenance_sha256(provenance))

    def test_missing_and_mismatched_sidecars_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, metadata_path, provenance, manifest_sha256 = self.write(root)
            with self.assertRaisesRegex(FileNotFoundError, "sidecar not found"):
                read_prompt_provenance(
                    root / "missing.metadata.json",
                    input_path=manifest_path,
                    input_sha256=manifest_sha256,
                    count=3,
                    split_counts={"train": 1, "val": 1, "test": 1},
                )

            provenance["manifest"]["sha256"] = "0" * 64
            metadata_path.write_text(
                json.dumps(
                    provenance,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                read_prompt_provenance(
                    metadata_path,
                    input_path=manifest_path,
                    input_sha256=manifest_sha256,
                    count=3,
                    split_counts={"train": 1, "val": 1, "test": 1},
                )

    def test_noncanonical_sidecar_encoding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, metadata_path, provenance, manifest_sha256 = self.write(Path(directory))
            metadata_path.write_text(
                json.dumps(provenance, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not in the canonical encoding"):
                read_prompt_provenance(
                    metadata_path,
                    input_path=manifest_path,
                    input_sha256=manifest_sha256,
                    count=3,
                    split_counts={"train": 1, "val": 1, "test": 1},
                )


class PromptManifestValidationTests(unittest.TestCase):
    @staticmethod
    def write(directory: Path, rows: list[dict]) -> Path:
        path = directory / "prompts.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def record(
        text: str,
        split: str,
        group: object,
        source: object = "overview_description",
    ) -> dict:
        return {
            "text": text,
            "split": split,
            "group": group,
            "source": source,
        }

    def test_valid_timeline_manifest_preserves_text_and_split_order(self) -> None:
        rows = [
            self.record("Walk forward.", "train", "walk"),
            self.record(
                "Turn left.",
                "val",
                "turn_left",
                "events.description",
            ),
            self.record("Jump.", "test", "jump"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(Path(directory), rows)
            texts, splits = read_prompts(path)

        self.assertEqual(texts, ["Walk forward.", "Turn left.", "Jump."])
        self.assertEqual(splits, ["train", "val", "test"])

    def test_group_source_and_text_invariants_fail_closed(self) -> None:
        mutations = (
            (
                self.record("Walk.", "train", None),
                "'group' must be a non-empty string",
            ),
            (
                self.record("Walk.", "train", " Walk "),
                "'group' is not in canonical",
            ),
            (
                self.record("Walk.", "train", "walk", "other.source"),
                "'source' must be one of",
            ),
            (
                self.record("Walk  forward.", "train", "walk"),
                "'text' is not in canonical",
            ),
        )
        for row, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = self.write(Path(directory), [row])
                with self.assertRaisesRegex(ValueError, message):
                    read_prompts(path)

    def test_canonical_duplicates_and_cross_split_groups_fail_closed(self) -> None:
        invalid_manifests = (
            (
                [
                    self.record("Walk-forward.", "train", "walk_1"),
                    self.record(
                        "walk forward!",
                        "test",
                        "walk_2",
                        "events.description",
                    ),
                ],
                "duplicate canonical prompt",
            ),
            (
                [
                    self.record("Walk forward.", "train", "walk"),
                    self.record(
                        "Stop walking.",
                        "test",
                        "walk",
                        "events.description",
                    ),
                ],
                "appears in both",
            ),
        )
        for rows, message in invalid_manifests:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = self.write(Path(directory), rows)
                with self.assertRaisesRegex(ValueError, message):
                    read_prompts(path)


class TeacherCacheProducerResumeTests(unittest.TestCase):
    @staticmethod
    def write_complete_cache(
        directory: Path,
    ) -> tuple[Path, Path, Path, bytes]:
        input_path = directory / "prompts.jsonl"
        records = (
            {
                "text": "walk",
                "split": "train",
                "group": "walk",
                "source": "overview_description",
            },
            {
                "text": "jump",
                "split": "val",
                "group": "jump",
                "source": "overview_description",
            },
            {
                "text": "turn",
                "split": "test",
                "group": "turn",
                "source": "events.description",
            },
        )
        input_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
        input_sha256 = sha256_file(input_path)
        corpus_provenance = _prompt_provenance(input_sha256)
        input_metadata_path = directory / "prompts.metadata.json"
        input_metadata_path.write_text(
            json.dumps(
                corpus_provenance,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        checkpoint_path = directory / "denoiser.safetensors"
        checkpoint_path.write_bytes(b"tiny checkpoint")
        output_dir = directory / "teacher"
        output_dir.mkdir()
        shard_path = output_dir / "teacher-00000.pt"
        torch.save(
            {
                "texts": [record["text"] for record in records],
                "splits": [record["split"] for record in records],
                "teacher_embeddings": torch.zeros(
                    3,
                    cache_teacher.TEACHER_DIM,
                    dtype=torch.float32,
                ),
                "targets": torch.zeros(
                    3,
                    cache_teacher.TARGET_DIM,
                    dtype=torch.float32,
                ),
            },
            shard_path,
        )
        identity = cache_teacher._metadata_identity(
            input_path=input_path,
            input_sha256=input_sha256,
            input_metadata_sha256=prompt_provenance_sha256(corpus_provenance),
            corpus_provenance=corpus_provenance,
            count=3,
            split_counts={"train": 1, "val": 1, "test": 1},
            foundation_model=cache_teacher.DEFAULT_FOUNDATION_MODEL,
            foundation_model_revision=(cache_teacher.DEFAULT_FOUNDATION_MODEL_REVISION),
            base_model=cache_teacher.DEFAULT_BASE_MODEL,
            base_model_revision=cache_teacher.DEFAULT_BASE_MODEL_REVISION,
            peft_model=cache_teacher.DEFAULT_PEFT_MODEL,
            peft_model_revision=cache_teacher.DEFAULT_PEFT_MODEL_REVISION,
            checkpoint=checkpoint_path,
            checkpoint_sha256=sha256_file(checkpoint_path),
            root_key=cache_teacher.DEFAULT_ROOT_KEY,
            body_key=cache_teacher.DEFAULT_BODY_KEY,
            shard_size=3,
        )
        metadata = {
            **identity,
            "device": {
                "cli_requested": "cpu",
                "env_override": None,
                "requested": "cpu",
                "resolved": "cpu",
            },
            "model_revisions": {
                "foundation_model": cache_teacher.DEFAULT_FOUNDATION_MODEL,
                "foundation": cache_teacher.DEFAULT_FOUNDATION_MODEL_REVISION,
                "base": cache_teacher.DEFAULT_BASE_MODEL_REVISION,
                "peft": cache_teacher.DEFAULT_PEFT_MODEL_REVISION,
            },
            "versions": {
                "torch": "2.9.1",
                "transformers": "5.1.0",
                "peft": "0.18.1",
                "safetensors": "0.7.0",
            },
            "provenance_status": "recorded",
            "completed_count": 3,
            "elapsed_seconds": 42.0,
            "status": "complete",
            "shards": [shard_path.name],
            "shard_sha256": {shard_path.name: sha256_file(shard_path)},
        }
        metadata_path = output_dir / cache_teacher.METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return input_path, checkpoint_path, output_dir, metadata_path.read_bytes()

    def test_complete_cache_noop_ignores_current_runtime_and_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, checkpoint_path, output_dir, metadata_before = self.write_complete_cache(root)
            stdout = io.StringIO()
            with (
                patch.object(cache_teacher, "inspect_projection_shapes"),
                patch.object(
                    cache_teacher,
                    "_runtime_provenance",
                    side_effect=AssertionError("complete cache must not inspect the current runtime"),
                ) as runtime_provenance,
                redirect_stdout(stdout),
            ):
                result = cache_teacher.main(
                    [
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(output_dir),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--shard-size",
                        "3",
                        "--device",
                        "cuda:7",
                    ]
                )
            metadata_after = (output_dir / cache_teacher.METADATA_FILENAME).read_bytes()

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
        self.assertEqual(metadata_after, metadata_before)
        runtime_provenance.assert_not_called()

    def test_incomplete_cache_resume_keeps_strict_device_comparison(self) -> None:
        previous = {
            "device": {
                "cli_requested": "cpu",
                "env_override": None,
                "requested": "cpu",
                "resolved": "cpu",
            },
            "model_revisions": {
                "foundation_model": cache_teacher.DEFAULT_FOUNDATION_MODEL,
                "foundation": cache_teacher.DEFAULT_FOUNDATION_MODEL_REVISION,
                "base": cache_teacher.DEFAULT_BASE_MODEL_REVISION,
                "peft": cache_teacher.DEFAULT_PEFT_MODEL_REVISION,
            },
            "versions": {
                "torch": "2.9.1",
                "transformers": "5.1.0",
                "peft": "0.18.1",
                "safetensors": "0.7.0",
            },
            "provenance_status": "recorded",
        }
        current = copy.deepcopy(previous)
        current["device"] = {
            "cli_requested": "cuda:0",
            "env_override": None,
            "requested": "cuda:0",
            "resolved": "cuda:0",
            "index": 0,
            "name": "different GPU",
            "capability": [9, 0],
        }

        with self.assertRaisesRegex(ValueError, "device provenance mismatch"):
            cache_teacher._select_resume_provenance(
                previous,
                current,
                has_shards=True,
            )


class TeacherCacheHardeningTests(unittest.TestCase):
    def test_valid_cache_loads_and_checkpoint_matches_resolved_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _shard, _metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
            cache = load_teacher_cache(cache_dir)
            (
                examples,
                loaded_metadata,
                loaded_lineage,
                loaded_fingerprint,
            ) = load_cached_examples(cache_dir)
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
        self.assertEqual(loaded_lineage, validated_teacher_lineage(metadata))
        self.assertEqual(loaded_fingerprint, teacher_cache_fingerprint(cache))

    def test_legacy_and_missing_provenance_fields_are_rejected(self) -> None:
        mutations = (
            ("format_version", 2, "unsupported teacher-cache format_version 2"),
            ("input_metadata_sha256", None, "missing required metadata fields"),
            ("corpus_provenance", None, "missing required metadata fields"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory)
                _shard, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
                if value is None:
                    metadata.pop(field)
                else:
                    metadata[field] = value
                _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
                with self.assertRaisesRegex(ValueError, message):
                    load_teacher_cache(cache_dir)

    def test_provenance_dataset_manifest_and_sidecar_hash_fail_closed(self) -> None:
        mutations = (
            (
                lambda metadata: metadata["corpus_provenance"]["dataset"].update(repo_id="other/dataset"),
                "dataset identity mismatch",
            ),
            (
                lambda metadata: metadata["corpus_provenance"]["manifest"].update(sha256="0" * 64),
                "manifest SHA-256 mismatch",
            ),
            (
                lambda metadata: metadata.update(input_metadata_sha256="0" * 64),
                "does not match corpus_provenance",
            ),
            (
                lambda metadata: metadata["corpus_provenance"]["preparation"].update(
                    normalization="different normalization"
                ),
                "preparation policy mismatch",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory)
                _shard, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
                mutate(metadata)
                _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
                with self.assertRaisesRegex(ValueError, message):
                    load_teacher_cache(cache_dir)

    def test_teacher_identity_contract_fails_closed(self) -> None:
        mutations = (
            (
                lambda metadata: metadata.pop("foundation_model_revision"),
                "missing required metadata fields",
            ),
            (
                lambda metadata: metadata.update(model_name="unrelated"),
                "model_name is inconsistent",
            ),
            (
                lambda metadata: metadata.update(base_model_revision="main"),
                "resolved 40-character",
            ),
            (
                lambda metadata: metadata["model_revisions"].update(base="d" * 40),
                "model_revisions.base is inconsistent",
            ),
            (
                lambda metadata: metadata["versions"].update(torch="unknown"),
                "versions.torch must be known",
            ),
            (
                lambda metadata: metadata["versions"].update(extra="1"),
                "versions must contain exactly",
            ),
            (
                lambda metadata: metadata["device"].update(requested="cuda"),
                "device.requested must equal",
            ),
            (
                lambda metadata: metadata.update(provenance_status="migrated_from_legacy_unverified"),
                "provenance_status must be 'recorded'",
            ),
            (
                lambda metadata: metadata.update(teacher_batch_size=2),
                "teacher_batch_size must be 1",
            ),
            (
                lambda metadata: metadata.update(target_keys=list(reversed(metadata["target_keys"]))),
                "target_keys must be",
            ),
            (
                lambda metadata: metadata["dtype"].pop("projection_weights"),
                "dtype must be exactly",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                cache_dir = Path(directory)
                _shard, metadata_path, original = _TeacherCacheFixture.write(cache_dir)
                metadata = copy.deepcopy(original)
                mutate(metadata)
                _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    load_teacher_cache(cache_dir)

    def test_cache_fingerprint_binds_corpus_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _shard, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
            original = teacher_cache_fingerprint(load_teacher_cache(cache_dir))

            metadata["corpus_provenance"]["preparation"]["seed"] += 1
            metadata["input_metadata_sha256"] = prompt_provenance_sha256(metadata["corpus_provenance"])
            _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
            changed = teacher_cache_fingerprint(load_teacher_cache(cache_dir))

        self.assertNotEqual(original, changed)

    def test_cache_fingerprint_binds_teacher_runtime_and_model_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _shard, metadata_path, metadata = _TeacherCacheFixture.write(cache_dir)
            original = teacher_cache_fingerprint(load_teacher_cache(cache_dir))

            metadata["base_model_revision"] = "d" * 40
            metadata["model_revisions"]["base"] = "d" * 40
            metadata["versions"]["torch"] = "2.9.2"
            _TeacherCacheFixture.rewrite_metadata(metadata_path, metadata)
            changed = teacher_cache_fingerprint(load_teacher_cache(cache_dir))

        self.assertNotEqual(original, changed)

    def test_artifact_must_bind_the_exact_cache_lineage_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _TeacherCacheFixture.write(cache_dir)
            cache = load_teacher_cache(cache_dir)
            fingerprint = teacher_cache_fingerprint(cache)
            artifact_config = {
                "metadata": {
                    "teacher_cache_lineage": validated_teacher_lineage(cache.metadata),
                    "teacher_cache_fingerprint": fingerprint,
                }
            }
            self.assertEqual(
                validate_artifact_teacher_cache_fingerprint(
                    artifact_config,
                    cache,
                ),
                fingerprint,
            )

            artifact_config["metadata"]["teacher_cache_fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "different teacher cache"):
                validate_artifact_teacher_cache_fingerprint(
                    artifact_config,
                    cache,
                )

            artifact_config["metadata"]["teacher_cache_fingerprint"] = fingerprint
            artifact_config["metadata"]["teacher_cache_lineage"]["base_model_revision"] = "d" * 40
            with self.assertRaisesRegex(ValueError, "different teacher-cache lineage"):
                validate_artifact_teacher_cache_fingerprint(
                    artifact_config,
                    cache,
                )

    def test_incomplete_count_split_and_hash_fail_closed(self) -> None:
        mutations = (
            ("status", "in_progress", "incomplete"),
            ("completed_count", 2, "incomplete"),
            ("count", 4, "incomplete"),
            (
                "split_counts",
                {"train": 2, "val": 0, "test": 1},
                "split counts",
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
            teacher_cache_lineage: dict | None = None
            teacher_cache_fingerprint = ""

            def __init__(self, **_kwargs) -> None:
                self.artifact_config = {
                    "format_version": 2,
                    "artifact_fingerprint": "a" * 64,
                    "base_model": "tiny",
                    "condition_dim": 1024,
                    "output_dim": 2048,
                    "compatible_ardy_models": ["ARDY-Core-RP-20FPS-Horizon40"],
                    "max_length": 16,
                    "metadata": {
                        "teacher_cache_lineage": self.teacher_cache_lineage,
                        "teacher_cache_fingerprint": self.teacher_cache_fingerprint,
                    },
                }

            def __call__(self, texts: list[str]):
                return torch.zeros(len(texts), 1, 2048), [1] * len(texts)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            _TeacherCacheFixture.write(cache_dir)
            cache = load_teacher_cache(cache_dir)
            _FakeEncoder.teacher_cache_lineage = validated_teacher_lineage(cache.metadata)
            _FakeEncoder.teacher_cache_fingerprint = teacher_cache_fingerprint(cache)
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
            output_dir="artifact",
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            base_model_revision="1" * 40,
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
            lr_schedule_epochs=None,
            cosine_weight=0.5,
            relational_weight=0.05,
            train_max_length=32,
            runtime_max_length=64,
            num_workers=0,
            seed=1,
            device="cpu",
            no_bf16=True,
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
            ("lr_schedule_epochs", 1),
            ("cosine_weight", -1.0),
            ("runtime_max_length", 0),
            ("num_workers", -1),
            ("seed", -1),
            ("base_model", ""),
            ("base_model_revision", "main"),
        )
        for field, value in invalid_values:
            args = self.valid_args()
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_training_args(args)

    def test_argument_parsing_does_not_probe_cuda_before_determinism_setup(
        self,
    ) -> None:
        with (
            patch(
                "sys.argv",
                ["train.py", "--cache-dir", "teacher-cache"],
            ),
            patch.object(
                train_module.torch.cuda,
                "is_available",
                side_effect=AssertionError("parse_args must not initialize or probe CUDA"),
            ),
        ):
            args = train_module.parse_args()

        self.assertEqual(args.device, "auto")

    def test_cublas_determinism_is_configured_before_cuda_probe(self) -> None:
        previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

        def cuda_available() -> bool:
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
                train_module.DEFAULT_CUBLAS_WORKSPACE_CONFIG,
            )
            return False

        try:
            with patch.object(
                train_module.torch.cuda,
                "is_available",
                side_effect=cuda_available,
            ):
                train_module.seed_everything(7)
        finally:
            if previous is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous

    def test_bf16_capability_is_checked_on_the_selected_cuda_device(self) -> None:
        selected = torch.device("cuda:3")
        with (
            patch.object(
                train_module.torch.cuda,
                "device",
                return_value=nullcontext(),
            ) as device_context,
            patch.object(
                train_module.torch.cuda,
                "is_bf16_supported",
                return_value=True,
            ),
        ):
            supported = cuda_supports_bf16(selected)

        self.assertTrue(supported)
        device_context.assert_called_once_with(selected)

    def test_shared_lr_horizon_supports_a_prefix_fair_epoch_comparison(
        self,
    ) -> None:
        shorter = self.valid_args()
        shorter.epochs = 50
        shorter.lr_schedule_epochs = 100
        longer = self.valid_args()
        longer.epochs = 100
        longer.lr_schedule_epochs = 100
        validate_training_args(shorter)
        validate_training_args(longer)

        batches_per_epoch = 409
        schedule_steps = 100 * batches_per_epoch
        warmup_steps = int(schedule_steps * shorter.warmup_ratio)
        shorter_prefix = [
            cosine_lr_multiplier(
                step,
                warmup_steps=warmup_steps,
                schedule_steps=schedule_steps,
            )
            for step in range(50 * batches_per_epoch)
        ]
        longer_prefix = [
            cosine_lr_multiplier(
                step,
                warmup_steps=warmup_steps,
                schedule_steps=schedule_steps,
            )
            for step in range(50 * batches_per_epoch)
        ]
        self.assertEqual(shorter_prefix, longer_prefix)

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                require_fresh_output_dir(directory)

            fresh = Path(directory) / "candidate"
            self.assertEqual(require_fresh_output_dir(fresh), fresh)

    def test_training_selects_on_validation_without_evaluating_test(self) -> None:
        class _Tokenizer:
            def __call__(
                self,
                texts,
                *,
                padding,
                truncation,
                max_length,
                return_tensors,
            ):
                self.assert_contract(
                    padding,
                    truncation,
                    max_length,
                    return_tensors,
                )
                input_ids = torch.tensor(
                    [[len(text)] for text in texts],
                    dtype=torch.long,
                )
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }

            @staticmethod
            def assert_contract(
                padding,
                truncation,
                max_length,
                return_tensors,
            ):
                if not padding or not truncation or max_length != 32 or return_tensors != "pt":
                    raise AssertionError("unexpected tokenizer contract")

        captured_metadata = {}

        class _Student(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(1, 2)
                self.adapter = nn.Linear(2, 2)
                self.root_head = nn.Linear(2, 1024)
                self.body_head = nn.Linear(2, 1024)

            def forward(self, input_ids, attention_mask):
                del attention_mask
                hidden = self.backbone(input_ids.float())
                shared = self.adapter(hidden)
                return torch.cat(
                    (self.root_head(shared), self.body_head(shared)),
                    dim=-1,
                )

            def save_artifact(
                self,
                output_dir,
                tokenizer,
                metadata,
                **_kwargs,
            ):
                del tokenizer
                captured_metadata.update(metadata)
                artifact_dir = Path(output_dir)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "ardy_minilm_config.json").write_text(
                    json.dumps(
                        {
                            "artifact_fingerprint": "f" * 64,
                            "artifact_files": {
                                "condition_heads.safetensors": {
                                    "size_bytes": 123,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return artifact_dir

        validation_metrics = (
            {"root_cosine": 0.9, "body_cosine": 0.8},
            {"root_cosine": 0.7, "body_cosine": 0.6},
        )
        generator = torch.Generator().manual_seed(9)
        examples = CachedExamples(
            texts=["walk", "run", "jump", "turn"],
            splits=["train", "train", "val", "test"],
            targets=torch.randn(4, 2048, generator=generator),
        )
        student = _Student()

        with tempfile.TemporaryDirectory() as directory:
            args = self.valid_args()
            args.output_dir = str(Path(directory) / "candidate")
            args.cache_dir = str(Path(directory) / "teacher")
            with (
                patch.object(
                    train_module,
                    "load_cached_examples",
                    return_value=(
                        examples,
                        {},
                        {"checkpoint_sha256": "b" * 64},
                        "a" * 64,
                    ),
                ),
                patch.object(
                    train_module,
                    "resolve_and_validate_teacher_checkpoint",
                    return_value=(
                        "ARDY-Core-RP-20FPS-Horizon40",
                        Path("denoiser.safetensors"),
                    ),
                ),
                patch.object(
                    train_module.AutoTokenizer,
                    "from_pretrained",
                    return_value=_Tokenizer(),
                ),
                patch.object(
                    train_module.MotionConditionStudent,
                    "from_base_model",
                    return_value=student,
                ),
                patch.object(
                    train_module,
                    "evaluate",
                    side_effect=validation_metrics,
                ) as evaluate_mock,
                patch.object(
                    train_module,
                    "training_runtime_versions",
                    return_value={"torch": "test"},
                ),
                redirect_stdout(io.StringIO()),
            ):
                report = train_module.train(args)

            persisted_report = json.loads((Path(args.output_dir) / "training_report.json").read_text(encoding="utf-8"))

        self.assertEqual(evaluate_mock.call_count, args.epochs)
        self.assertEqual(report["best_epoch"], 1)
        self.assertEqual(report["artifact_fingerprint"], "f" * 64)
        self.assertEqual(report["artifact_payload_size_bytes"], 123)
        self.assertEqual(report["configuration"]["epochs"], args.epochs)
        self.assertNotIn("test", report)
        self.assertNotIn("test", persisted_report)
        self.assertEqual(
            report["selection"],
            {
                "split": "val",
                "metric": "mean(root_cosine, body_cosine)",
                "test_evaluated": False,
            },
        )
        self.assertNotIn("test_metrics", captured_metadata)
        self.assertEqual(captured_metadata["training"]["test_examples"], 1)
        serialized_metadata = json.dumps(captured_metadata, sort_keys=True)
        self.assertNotIn(directory, serialized_metadata)

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
