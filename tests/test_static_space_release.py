# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import scripts.prepare_static_space_release as release_module
from scripts.prepare_static_space_release import RELEASE_MARKER, prepare_release


def test_static_space_release_is_allowlisted_and_standalone() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "space"
        prepare_release(output, allow_dirty=True)

        readme = (output / "README.md").read_text(encoding="utf-8")
        assert "sdk: static" in readme
        assert "app_build_command: npm run build" in readme
        assert "cross-origin-embedder-policy: require-corp" in readme
        assert "Built with Meta Llama 3" in readme
        assert (output / RELEASE_MARKER).is_file()
        assert (output / "src" / "entry.tsx").is_file()
        assert (output / "scripts" / "copy-ort-assets.mjs").is_file()
        assert (output / "LICENSE").is_file()
        assert (output / "THIRD_PARTY_MODELS_AND_DATA.md").is_file()
        assert not (output / "dist").exists()
        assert not (output / "e2e").exists()
        assert not (output / "node_modules").exists()


def test_static_space_release_refuses_unknown_existing_output() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "space"
        output.mkdir()
        (output / "unrelated.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(FileExistsError, match="unrecognized"):
            prepare_release(output, allow_dirty=True)

        assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_static_space_release_requires_a_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_module, "_source_is_dirty", lambda: True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "space"
        with pytest.raises(RuntimeError, match="dirty checkout"):
            prepare_release(output)
