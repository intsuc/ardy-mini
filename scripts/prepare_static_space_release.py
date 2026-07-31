# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Prepare the allowlisted source tree for the ARDY Mini Static Space."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPOSITORY_ROOT / "web"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts" / "huggingface" / "space" / "ardy-mini"
RELEASE_MARKER = ".ardy-static-space-release.json"

WEB_FILES = (
    "ONNXRUNTIME_LICENSE.txt",
    "components.json",
    "index.html",
    "package-lock.json",
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
)
WEB_DIRECTORIES = (
    "patches",
    "scripts",
    "src",
)
ROOT_NOTICES = (
    "LICENSE",
    "NOTICE",
    "ATTRIBUTIONS.MD",
    "THIRD_PARTY_MODELS_AND_DATA.md",
)


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _source_commit() -> str | None:
    commit = _git_output("rev-parse", "HEAD")
    return commit if commit is not None and len(commit) == 40 else None


def _source_is_dirty() -> bool:
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    return status is None or bool(status)


def _copy_required_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required Static Space source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _safe_existing_release(output: Path) -> bool:
    return output.is_dir() and (output / RELEASE_MARKER).is_file()


def prepare_release(output: Path, *, allow_dirty: bool = False) -> Path:
    output = output.resolve()
    repository_root = REPOSITORY_ROOT.resolve()
    if output == repository_root or output.parent == repository_root:
        raise ValueError(
            "Choose a dedicated output directory below artifacts/ or outside "
            "the repository root."
        )
    if output.exists() and not _safe_existing_release(output):
        raise FileExistsError(
            f"Refusing to replace unrecognized output directory: {output}"
        )
    source_dirty = _source_is_dirty()
    if source_dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to prepare a publishable Space from a dirty checkout. "
            "Commit the release or pass --allow-dirty for local validation only."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}-",
    ) as temporary_directory:
        stage = Path(temporary_directory)
        _copy_required_file(WEB_ROOT / "space" / "README.md", stage / "README.md")
        for relative_path in WEB_FILES:
            _copy_required_file(WEB_ROOT / relative_path, stage / relative_path)
        for relative_path in WEB_DIRECTORIES:
            source = WEB_ROOT / relative_path
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Required Static Space source is missing: {source}"
                )
            shutil.copytree(source, stage / relative_path)
        for relative_path in ROOT_NOTICES:
            _copy_required_file(REPOSITORY_ROOT / relative_path, stage / relative_path)

        marker = {
            "format": "ardy-static-space-release-v1",
            "source_commit": _source_commit(),
            "source_dirty": source_dirty,
            "space": "intsuc/ardy-mini",
        }
        (stage / RELEASE_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(stage, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Static Space source directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow local staging from uncommitted source; do not publish that tree.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_release(args.output, allow_dirty=args.allow_dirty)
    print(f"Prepared Static Space source: {output}")
    print(f"Validate with: cd {output} && npm ci && npm run build")


if __name__ == "__main__":
    main()
