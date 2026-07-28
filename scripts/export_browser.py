# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Export the MiniLM Core40 ONNX model pack used by the browser demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from ardy.browser import BrowserExportConfig, export_browser_model_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Export MiniLM, text-only ARDY denoiser, and motion decoder graphs for ONNX Runtime Web.")
    )
    parser.add_argument(
        "--minilm-artifact",
        type=Path,
        default=Path("artifacts/minilm-ardy-core40"),
        help="Trained MiniLM ARDY artifact directory.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing ARDY-Core-RP-20FPS-Horizon40.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/browser/core40"),
        help="Destination model-pack directory (kept under ignored artifacts/ by default).",
    )
    parser.add_argument(
        "--model",
        default="core",
        help="ARDY model key or full released model name (browser v1 supports Core40).",
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
        help="Maximum WordPiece sequence length.",
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
        help="Skip ONNX Runtime CPU numeric comparison (ONNX checker still runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = export_browser_model_pack(
        BrowserExportConfig(
            output_dir=args.output_dir,
            minilm_artifact=args.minilm_artifact,
            checkpoints_dir=args.checkpoints_dir,
            model=args.model,
            max_prompt_tokens=args.max_prompt_tokens,
            opset=args.opset,
            device=args.device,
            verify=not args.skip_verify,
        )
    )
    print(f"Browser model pack exported: {manifest_path}")


if __name__ == "__main__":
    main()
