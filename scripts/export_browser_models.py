# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Export the mixed-FP16 MiniLM Core40 files used by the browser app."""

from __future__ import annotations

import argparse
from pathlib import Path

from ardy.browser import BrowserExportConfig, export_browser_model_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the MiniLM condition encoder, text-conditioned ARDY "
            "denoiser, and structured motion decoder as a mixed-FP16 model "
            "directory for ONNX Runtime Web."
        )
    )
    parser.add_argument(
        "--minilm-artifact",
        type=Path,
        default=Path("artifacts/minilm-ardy-core40"),
        help="Checkpoint-specific trained MiniLM Core40 artifact directory.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing the separately obtained ARDY-Core-RP-20FPS-Horizon40 checkpoint.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/browser/ardy-minilm-core40-browser-v1"),
        help=(
            "Destination for model.json.gz and individually compressed assets "
            "(kept under ignored artifacts/ by default)."
        ),
    )
    parser.add_argument(
        "--fp32-reference-output-directory",
        type=Path,
        default=None,
        help=(
            "Optional destination for matching original-FP32 reference files "
            "used by scripts/evaluate_browser_fp16.py."
        ),
    )
    parser.add_argument(
        "--model",
        default="core",
        help="ARDY model key or full model name; this MiniLM browser artifact supports Core40 only.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Export device: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=128,
        help="Maximum WordPiece length for the supported typo-free English motion prompts.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (minimum 17).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "Skip both CPU numerical comparisons (PyTorch vs FP32 ONNX and "
            "FP32 ONNX vs mixed-FP16 ONNX); ONNX checking and mixed-FP16 "
            "conversion still run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = export_browser_model_files(
        BrowserExportConfig(
            output_directory=args.output_directory,
            fp32_reference_output_directory=(
                args.fp32_reference_output_directory
            ),
            minilm_artifact=args.minilm_artifact,
            checkpoints_dir=args.checkpoints_dir,
            model=args.model,
            max_prompt_tokens=args.max_prompt_tokens,
            opset=args.opset,
            device=args.device,
            verify=not args.skip_verify,
        )
    )
    print(f"Browser model files exported: {output_directory}")
    if args.fp32_reference_output_directory is not None:
        print(
            "FP32 reference model files exported: "
            f"{args.fp32_reference_output_directory}"
        )


if __name__ == "__main__":
    main()
