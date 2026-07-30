# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for MiniLM benchmark provenance."""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.minilm.benchmark_encoders import (
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_FOUNDATION_MODEL_REVISION,
    DEFAULT_PEFT_MODEL_REVISION,
    _hardware_identity,
    _load_teacher,
    _student_artifact_identity,
    _teacher_identity,
)
from scripts.minilm.cache_teacher import (
    DEFAULT_BASE_MODEL_REVISION as CACHE_BASE_MODEL_REVISION,
)
from scripts.minilm.cache_teacher import (
    DEFAULT_FOUNDATION_MODEL_REVISION as CACHE_FOUNDATION_MODEL_REVISION,
)
from scripts.minilm.cache_teacher import (
    DEFAULT_PEFT_MODEL_REVISION as CACHE_PEFT_MODEL_REVISION,
)
from scripts.minilm.cache_teacher import (
    resolve_pinned_snapshot,
)


class BenchmarkTeacherIdentityTests(unittest.TestCase):
    def test_benchmark_revisions_match_teacher_cache_revisions(self) -> None:
        self.assertEqual(DEFAULT_FOUNDATION_MODEL_REVISION, CACHE_FOUNDATION_MODEL_REVISION)
        self.assertEqual(DEFAULT_BASE_MODEL_REVISION, CACHE_BASE_MODEL_REVISION)
        self.assertEqual(DEFAULT_PEFT_MODEL_REVISION, CACHE_PEFT_MODEL_REVISION)

    def test_cpu_hardware_identity_is_explicit(self) -> None:
        identity = _hardware_identity(None)

        self.assertIsNone(identity["cuda"])
        self.assertIn("platform", identity)
        self.assertIn("machine", identity)
        self.assertIn("logical_cpu_count", identity)

    def test_teacher_identity_contains_every_model_and_revision(self) -> None:
        args = argparse.Namespace(
            foundation_model="foundation/repo",
            foundation_model_revision="a" * 40,
            base_model="base/repo",
            base_model_revision="b" * 40,
            peft_model="peft/repo",
            peft_model_revision="c" * 40,
            llm_dim=4096,
        )

        self.assertEqual(
            _teacher_identity(args),
            {
                "foundation_model": "foundation/repo",
                "foundation_model_revision": "a" * 40,
                "base_model": "base/repo",
                "base_model_revision": "b" * 40,
                "peft_model": "peft/repo",
                "peft_model_revision": "c" * 40,
                "llm_dim": 4096,
            },
        )

    def test_snapshot_resolution_can_be_forced_offline(self) -> None:
        revision = "d" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            with patch(
                "scripts.minilm.cache_teacher.snapshot_download",
                return_value=str(snapshot),
            ) as download:
                resolved = resolve_pinned_snapshot(
                    "owner/repo",
                    revision,
                    allow_patterns=("config.json",),
                    local_files_only=True,
                )

        self.assertEqual(resolved, snapshot)
        download.assert_called_once_with(
            repo_id="owner/repo",
            revision=revision,
            cache_dir=None,
            allow_patterns=["config.json"],
            local_files_only=True,
        )

    def test_direct_script_teacher_load_uses_sibling_cache_module(self) -> None:
        calls: list[tuple[str, str, bool]] = []

        def resolve(repo_id: str, revision: str, **kwargs: object) -> Path:
            calls.append((repo_id, revision, bool(kwargs["local_files_only"])))
            return Path("/snapshots") / revision

        cache_module = types.ModuleType("cache_teacher")
        cache_module.resolve_pinned_snapshot = resolve  # type: ignore[attr-defined]
        encoder = object()
        args = argparse.Namespace(
            foundation_model="foundation/repo",
            foundation_model_revision="a" * 40,
            base_model="base/repo",
            base_model_revision="b" * 40,
            peft_model="peft/repo",
            peft_model_revision="c" * 40,
            dtype="bfloat16",
            llm_dim=4096,
        )
        with (
            patch("scripts.minilm.benchmark_encoders.__package__", ""),
            patch.dict(sys.modules, {"cache_teacher": cache_module}),
            patch(
                "ardy.model.llm2vec.llm2vec_wrapper.LLM2VecEncoder",
                return_value=encoder,
            ) as encoder_type,
        ):
            loaded = _load_teacher(args, "cpu")

        self.assertIs(loaded, encoder)
        self.assertEqual(
            calls,
            [
                ("foundation/repo", "a" * 40, True),
                ("base/repo", "b" * 40, True),
                ("peft/repo", "c" * 40, True),
            ],
        )
        encoder_type.assert_called_once_with(
            base_model_name_or_path=str(Path("/snapshots") / ("b" * 40)),
            peft_model_name_or_path=str(Path("/snapshots") / ("c" * 40)),
            foundation_model_name_or_path=str(Path("/snapshots") / ("a" * 40)),
            dtype="bfloat16",
            llm_dim=4096,
            device="cpu",
        )

    def test_student_identity_contains_artifact_fingerprint_and_file_hashes(self) -> None:
        encoder = argparse.Namespace(
            artifact_config={
                "artifact_fingerprint": "f" * 64,
                "format_version": 2,
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "compatible_ardy_models": ["ARDY-Core-RP-20FPS-Horizon40"],
                "output_dim": 2048,
            }
        )
        args = argparse.Namespace(student_path="artifact")
        with patch(
            "scripts.minilm.benchmark_encoders._artifact_file_hashes",
            return_value={"condition_heads.safetensors": "a" * 64},
        ):
            identity = _student_artifact_identity(encoder, args)

        self.assertEqual(identity["artifact_fingerprint"], "f" * 64)
        self.assertEqual(
            identity["files_sha256"],
            {"condition_heads.safetensors": "a" * 64},
        )


if __name__ == "__main__":
    unittest.main()
