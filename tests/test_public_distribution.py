# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Repository-level checks for the public source distribution."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TRACKED_PREFIXES = (
    "artifacts/",
    "checkpoints/",
    "datasets/",
    "outputs/",
)
MAX_REGULAR_GIT_FILE_BYTES = 50 * 1024 * 1024
MODIFICATION_NOTICE = "Modified by intsuc in 2026"
MODIFIED_UPSTREAM_FILES = (
    ".gitignore",
    "MANIFEST.in",
    "README.md",
    "ardy/model/__init__.py",
    "ardy/model/ardy_model.py",
    "ardy/model/auto_latent_twostage_denoiser.py",
    "ardy/model/backbone.py",
    "ardy/model/llm2vec/llm2vec_wrapper.py",
    "ardy/model/load_model.py",
    "ardy/model/text_encoder_api.py",
    "ardy/model/trt.py",
    "pyproject.toml",
    "scripts/export_onnx.py",
    "scripts/interactive_demo/embedding_cache.py",
    "scripts/interactive_demo/loading.py",
    "scripts/run_text_encoder_server.py",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry]


class PublicDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPOSITORY_ROOT / ".git").exists():
            raise unittest.SkipTest("public-distribution checks require a Git checkout")

    def test_private_and_generated_roots_are_not_tracked(self) -> None:
        leaked = [path for path in tracked_files() if path.startswith(FORBIDDEN_TRACKED_PREFIXES)]
        self.assertEqual(leaked, [])

    def test_regular_git_files_are_below_github_warning_threshold(self) -> None:
        oversized = {
            path: (REPOSITORY_ROOT / path).stat().st_size
            for path in tracked_files()
            if (REPOSITORY_ROOT / path).is_file()
            and (REPOSITORY_ROOT / path).stat().st_size > MAX_REGULAR_GIT_FILE_BYTES
        }
        self.assertEqual(oversized, {})

    def test_local_workspace_path_is_not_published(self) -> None:
        local_path = "/" + "home/intsuc/ardy-mini"
        result = subprocess.run(
            ["git", "grep", "-I", "-n", "--fixed-strings", local_path, "--"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_modified_upstream_files_carry_change_notices(self) -> None:
        missing = [
            path
            for path in MODIFIED_UPSTREAM_FILES
            if MODIFICATION_NOTICE not in (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        ]
        self.assertEqual(missing, [])

    def test_distribution_notices_are_present(self) -> None:
        for path in ("LICENSE", "ATTRIBUTIONS.MD", "NOTICE", "THIRD_PARTY_MODELS_AND_DATA.md"):
            with self.subTest(path=path):
                self.assertTrue((REPOSITORY_ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
