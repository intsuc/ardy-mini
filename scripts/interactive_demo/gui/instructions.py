# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: shared markdown copy.

Single source of truth for the welcome modal, the Instructions tab, and the `h`-key keyboard-
shortcuts modal, so the two shortcut listings can't drift apart.
"""

# Rendered standalone by the `h`-key help modal (gui/io.py) and embedded at
# the end of INSTRUCTIONS_TAB_MD below.
KEYBOARD_SHORTCUTS_MD = (
    "| Key | Action |\n"
    "|:---:|--------|\n"
    "| `Space` | Toggle play / pause |\n"
    "| `j` | Previous frame |\n"
    "| `k` | Next frame |\n"
    "| `r` | Reset auto-follow camera to current frame |\n"
    "| `t` | Toggle target velocity (heading + arrow overlay) |\n"
    "| `Arrow keys` | Steer target velocity (when `t` is enabled) |\n"
    "| `p` | Toggle waypoint control mode |\n"
    "| `z` | Sample kinematic constraints from motion file |\n"
    "| `h` | Show / hide this help |\n"
)

# Shown once per browser in the welcome modal on connect (see client.py's
# on_client_connect); dismissal is persisted client-side via `save_choice`.
QUICK_START_MD = """
##### Camera
Left-drag to rotate · right-drag to pan · scroll to zoom.

##### Playback
`Space` plays/pauses · `j` / `k` step one frame back/forward.
Motion is generated ahead of the playhead automatically while **Auto Replan** is on (Playback tab).

##### Text prompts
Open the **Text** tab, type a prompt (or pick one from the *Prompt List*), then press
**Update Text Prompt**. The new prompt takes effect at the current frame — each prompt
shows as a colored segment on the timeline.

##### Steering
`p` toggles **Waypoint Mode** — click the floor to set where the character should go.
`t` toggles **Target Velocity** — steer with the arrow keys (`↑`/`↓` speed, `←`/`→` turn).

##### Constraints
Click a constraint track in the timeline to pin the character's current pose at that
frame; right-click a marker to remove it. The **Generate** tab can also sample
constraints from a reference motion file.

Press `h` anytime to see all keyboard shortcuts — and see the **Instructions** tab for
the full manual.
"""

# Rendered in the Instructions tab (gui/orchestrator.py).
INSTRUCTIONS_TAB_MD = f"""### What is this demo?

    ARDY generates character motion autoregressively in a streaming fashion: motion is
    planned a window at a time, just ahead of the playhead, and re-planned whenever you
    change the text prompt, place a waypoint, steer with target velocity, or edit
    constraints.

    ### Playback

    - `Space` play/pause, `j`/`k` step, or scrub via the timeline or **Frame Index**
      (Playback tab).
    - **Enable Auto Replan** keeps generating new motion as the playhead approaches the end
      of what exists. Turn it off to inspect a fixed clip.

    ### Text prompts

    - Edit the prompt in the **Text** tab and press **Update Text Prompt**; the
      *Prompt List* folder has ready-made examples.
    - Prompts take effect from the current frame onward and appear as colored segments on
      the timeline. The active prompt is always shown above the tabs.

    ### Steering the character

    - **Waypoints** (`p`): click the floor to drop a target the character will reach a bit
      later (see **Waypoint Interval** in the Generate tab). **Use Dense Root** makes it
      follow a smoothed dense path.
    - **Target velocity** (`t`): drive the character like a gamepad — arrow keys change
      speed and heading; the orange arrow shows the target.

    ### Constraints

    - **Timeline tracks**: click a track (Full-Body, 2D Root, hands/feet) to pin the
      current pose or joints at that frame; right-click a marker to delete it.
    - **Sample from a motion file** (Generate tab, or `z`): loads a reference motion
      (BVH/CSV) and samples the checked constraint types; **Show Reference Motion**
      (Visualize tab) displays it as a red ghost.
    - **Initial Body Transform** (Generate tab): a gizmo to set the starting position and
      heading before restarting.
    - **Restart** regenerates from scratch; **Restart From Now** keeps the past and
      regenerates the future; **Clear All Constraints** removes every keyframe and
      interval.

    ### Visualization

    - Toggle the mesh, skeleton, and foot contacts; enable the follow camera with
      over-the-shoulder or front-facing presets; adjust FOV or set the camera manually.
    - Dark mode lives in the title bar (top right).

    ### Saving & loading (IO tab)

    - **Session**: save and restore motion + prompts + constraints (`.pkl`).
    - **Root constraints**: save/load as JSON.
    - **Scene**: load a `.ply`/`.obj` environment mesh; capture the viewport to a PNG.

    ### Models (Model tab)

    - Pick a **Skeleton** and **Horizon**, then press **Load Model**. Acceleration runs
      via TensorRT by default (or `torch.compile`).
    - Fine-grained sampling knobs (seed, denoising steps, classifier-free guidance,
      replan behavior) live in the collapsed **Advanced** folder.

    ### Keyboard shortcuts
    {KEYBOARD_SHORTCUTS_MD}"""
