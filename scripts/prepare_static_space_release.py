# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Build and stage the allowlisted ARDY Mini Static Space deployment."""

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


def _copy_required_directory(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"Required Static Space source is missing: {source}")
    shutil.copytree(source, destination)


def _build_web_distribution(build_root: Path) -> Path:
    for command in (("npm", "ci"), ("npm", "run", "build")):
        try:
            subprocess.run(command, cwd=build_root, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"Static Space web build failed: {' '.join(command)}"
            ) from error
    distribution = build_root / "dist"
    if not (distribution / "index.html").is_file():
        raise RuntimeError("Static Space build did not produce dist/index.html")
    return distribution


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
        temporary_root = Path(temporary_directory)
        build_root = temporary_root / "build"
        release_root = temporary_root / "release"
        build_root.mkdir()
        release_root.mkdir()

        for relative_path in WEB_FILES:
            _copy_required_file(WEB_ROOT / relative_path, build_root / relative_path)
        for relative_path in WEB_DIRECTORIES:
            _copy_required_directory(
                WEB_ROOT / relative_path,
                build_root / relative_path,
            )
        for relative_path in ROOT_NOTICES:
            _copy_required_file(
                REPOSITORY_ROOT / relative_path,
                build_root / relative_path,
            )

        distribution = _build_web_distribution(build_root)
        for source in distribution.iterdir():
            destination = release_root / source.name
            if source.is_symlink():
                raise RuntimeError(f"Static build output contains a symlink: {source}")
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise RuntimeError(f"Unsupported static build output: {source}")

        _copy_required_file(
            WEB_ROOT / "space" / "README.md",
            release_root / "README.md",
        )
        for relative_path in ROOT_NOTICES:
            _copy_required_file(
                REPOSITORY_ROOT / relative_path,
                release_root / relative_path,
            )

        marker = {
            "format": "ardy-static-space-release-v2",
            "source_commit": _source_commit(),
            "source_dirty": source_dirty,
            "space": "intsuc/ardy-mini",
        }
        (release_root / RELEASE_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(release_root, output)
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
    print(f"Prepared prebuilt Static Space deployment: {output}")
    print("The staged index.html is served directly; no hosted build command is used.")


if __name__ == "__main__":
    main()
