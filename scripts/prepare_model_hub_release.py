# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Stage the public Llama 3 ARDY Mini Core40 Browser Model Hub repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from ardy.model_hub_release import ReleaseConfig, stage_model_hub_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and stage the allowlisted FP16/FP32 browser model files, "
            "model card, composite terms, pinned licenses, reports, provenance, and checksums."
        )
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("artifacts/browser/ardy-minilm-core40-browser-v1"),
        help="Exporter-owned model-family directory containing fp16/ and fp32/.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/model-hub/Llama-3-ARDY-Mini-Core40-Browser"),
        help="Ignored staging directory to upload as the Model Hub repository root.",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--reports-directory", type=Path, default=Path("reports"))
    parser.add_argument("--template-directory", type=Path, default=Path("model_hub"))
    parser.add_argument(
        "--license-cache-directory",
        type=Path,
        default=Path("artifacts/model-hub-license-cache"),
        help="Ignored cache for hash-pinned official license texts.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy large transports instead of hard-linking when the staging directory is on the same filesystem.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Permit a dirty tracked worktree for a non-release check; published releases should not use this.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace a prior recognized staging directory for this exact model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = stage_model_hub_release(
        ReleaseConfig(
            model_directory=args.model_directory,
            output_directory=args.output_directory,
            repository_root=args.repository_root,
            template_directory=args.template_directory,
            reports_directory=args.reports_directory,
            license_sources_path=args.template_directory / "LICENSE_SOURCES.json",
            license_cache_directory=args.license_cache_directory,
            use_hardlinks=not args.copy,
            allow_dirty_source=args.allow_dirty_source,
            replace=args.replace,
        )
    )
    print(f"Model Hub release staged: {output}")
    print("Upload only this directory; it contains no teacher weights, datasets, prompts, or source checkpoints.")


if __name__ == "__main__":
    main()
