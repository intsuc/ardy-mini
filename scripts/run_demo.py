# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive demo entry point.

The implementation is split across the ``interactive_demo`` package; this module assembles the
mixins into ``InteractiveTimelineDemo`` and exposes the Hydra ``main`` entry point.
"""

import argparse

from interactive_demo.camera import CameraMixin
from interactive_demo.characters import CharactersMixin
from interactive_demo.client import ClientMixin
from interactive_demo.common import *  # noqa: F401,F403
from interactive_demo.constraints import ConstraintsMixin
from interactive_demo.embedding_cache import CachedTextEncoder
from interactive_demo.gen_constraints import GenConstraintsMixin
from interactive_demo.generation import GenerationMixin
from interactive_demo.gui import (
    GuiGenerateMixin,
    GuiIOMixin,
    GuiMixin,
    GuiModelMixin,
    GuiPlaybackMixin,
    GuiTextMixin,
    GuiVisualizeMixin,
)
from interactive_demo.loading import ModelLoadingMixin
from interactive_demo.motion_io import MotionIOMixin
from interactive_demo.playback import PlaybackMixin
from interactive_demo.session_io import SessionIOMixin


class InteractiveTimelineDemo(
    ModelLoadingMixin,
    MotionIOMixin,
    ClientMixin,
    CharactersMixin,
    ConstraintsMixin,
    SessionIOMixin,
    GenerationMixin,
    GenConstraintsMixin,
    CameraMixin,
    GuiMixin,
    GuiPlaybackMixin,
    GuiTextMixin,
    GuiGenerateMixin,
    GuiVisualizeMixin,
    GuiModelMixin,
    GuiIOMixin,
    PlaybackMixin,
):
    def __init__(self, compile_model: bool = True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Built once and reused across all model loads (core / g1 / soma).
        # Wrapped in a disk-backed cache so repeated/preset prompts skip
        # re-encoding, both within a session and across demo restarts.
        self.text_encoder = CachedTextEncoder(self._build_text_encoder())

        # Prewarm the cache for the default prompt and Prompt List presets
        # on a background thread; the server must not wait on this.
        threading.Thread(
            target=self.text_encoder.prewarm,
            args=([DEFAULT_PROMPT, *PRESET_PROMPTS],),
            daemon=True,
        ).start()

        self.compile_model = compile_model

        # Dataset (will be loaded per-client if needed)
        self.max_keyframe_num = 6

        # Motion file cache directory
        self._motion_cache_dir = os.path.join(REPO_ROOT, "datasets", "bones-seed", "cache")
        os.makedirs(self._motion_cache_dir, exist_ok=True)
        self._metadata_df = None  # Lazy-loaded metadata DataFrame
        # Cache of motion-file paths per skeleton family, built once from the
        # metadata CSV so the "Random Motion File" button doesn't need to walk
        # the dataset on every click. Any failure here (missing CSV, missing
        # pandas, malformed file) is logged but does not block the demo —
        # other features keep working without the bones-seed dataset.
        self._bones_seed_paths_by_skeleton: dict[str, list[str]] = {}
        try:
            self._prime_bones_seed_paths()
        except Exception as e:
            print(
                f"[bones-seed] Failed to load motion-path index: {e!r}. "
                "Random Motion File button will be disabled; other features "
                "remain available."
            )

        # Per-client sessions
        self.client_sessions: dict[int, ClientSession] = {}

        # Server setup
        self.server = viser.ViserServer(
            host="0.0.0.0",
            port=2333,
            label="ARDY Interactive Demo",
            enable_camera_keyboard_controls=False,
            show_timeline_arrow_keys=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")

        # Register callbacks for session handling
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

        # Floor setup
        self.floor_len = 20.0

        # Color palette for text prompts (darker colors for good contrast with white text)
        self.prompt_colors = [
            (40, 100, 200),  # Deep blue
            (200, 80, 40),  # Burnt orange
            (40, 150, 60),  # Forest green
            (180, 40, 150),  # Deep magenta
            (150, 120, 40),  # Dark gold
            (60, 120, 180),  # Steel blue
            (180, 60, 80),  # Deep red
            (100, 60, 180),  # Deep purple
            (40, 140, 120),  # Teal
            (160, 60, 120),  # Maroon
        ]

    def get_prompt_color(self, prompt_index: int) -> tuple:
        """Get a color for a prompt based on its index."""
        return self.prompt_colors[prompt_index % len(self.prompt_colors)]


def main() -> None:
    parser = argparse.ArgumentParser(description="ARDY Interactive Demo")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Do not compile the model (initial backend is 'None' instead of 'ONNX-TRT (fp16)').",
    )
    args = parser.parse_args()

    demo = InteractiveTimelineDemo(compile_model=not args.no_compile)
    demo.run()


if __name__ == "__main__":
    main()
