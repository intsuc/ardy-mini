# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI orchestrator (per-tab builders live in gui_*.py)."""

from types import SimpleNamespace

from ..common import *  # noqa: F401,F403
from .instructions import INSTRUCTIONS_TAB_MD


class GuiMixin:
    def create_gui(self, client: viser.ClientHandle, constraint_tracks: dict):
        """Create GUI elements for a client."""
        g = SimpleNamespace()
        client_id = client.client_id

        # Setup timeline (check if available)
        timeline = None
        timeline_available = hasattr(client, "timeline")

        default_prompt = DEFAULT_PROMPT
        uuid = None

        if timeline_available:
            try:
                print("Setting up timeline")
                timeline = client.timeline
                uuid = timeline.add_prompt(
                    text=default_prompt,
                    start_frame=0,
                    end_frame=INFINITE_FRAME_IDX,
                    color=self.get_prompt_color(0),
                )
                timeline.set_visible(True)
                timeline.set_current_frame(0)
                # Set initial window: 20 frames before (0 in this case) + 200 frames after
                timeline.set_frame_range(start_frame=0, end_frame=TIMELINE_WINDOW_AFTER)
            except Exception as e:
                print(f"Warning: Could not setup timeline: {e}")
                timeline_available = False
        else:
            print("=" * 60)
            print("⚠️  TIMELINE FEATURE NOT AVAILABLE")
            print("=" * 60)
            print("Your viser version doesn't support timeline features.")
            print("The demo will work with waypoints and keyframes only.")
            print("To enable timeline: install viser from source with timeline support")
            print("=" * 60)

        # Create timeline tracks (only if timeline is available)
        timeline_tracks = {}
        if timeline_available and timeline is not None:
            fullbody_id = timeline.add_track(
                "Full-Body",
                track_type="keyframe",
                color=(219, 148, 86),
                height_scale=0.5,
            )
            timeline_tracks[fullbody_id] = {"name": "Full-Body"}

            root2d_id = timeline.add_track(
                "2D Root",
                track_type="keyframe",
                color=(150, 100, 200),
                height_scale=0.5,
            )
            timeline_tracks[root2d_id] = {"name": "2D Root"}

            lefthand_id = timeline.add_track(
                "Left Hand",
                track_type="keyframe",
                color=(100, 200, 150),
                height_scale=0.5,
            )
            timeline_tracks[lefthand_id] = {"name": "Left Hand"}

            righthand_id = timeline.add_track(
                "Right Hand",
                track_type="keyframe",
                color=(200, 100, 150),
                height_scale=0.5,
            )
            timeline_tracks[righthand_id] = {"name": "Right Hand"}

            leftfoot_id = timeline.add_track(
                "Left Foot",
                track_type="keyframe",
                color=(219, 148, 86),
                height_scale=0.5,
            )
            timeline_tracks[leftfoot_id] = {"name": "Left Foot"}

            rightfoot_id = timeline.add_track(
                "Right Foot",
                track_type="keyframe",
                color=(150, 100, 200),
                height_scale=0.5,
            )
            timeline_tracks[rightfoot_id] = {"name": "Right Foot"}

        # Setup timeline data
        timeline_data = {
            "tracks": timeline_tracks,
            "tracks_ids": {val["name"]: key for key, val in timeline_tracks.items()},
            "keyframes": {},
            "intervals": {},
            "keyframe_update_lock": threading.Lock(),
            "keyframe_move_timers": {},
            "pending_keyframe_moves": {},
            "prompt_uuid_list": [uuid] if uuid is not None else [],
            "prompt_counter": 1,  # Counter for prompt colors (starts at 1 since initial prompt is 0)
        }

        # Active prompt label
        g.gui_active_prompt_label = client.gui.add_markdown("**Active Prompt:** A person is walking.")

        tab_group = client.gui.add_tab_group()

        #
        # Playback tab
        #
        self._build_playback_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_text_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_generate_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_visualize_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_model_tab(client, client_id, tab_group, g, timeline, default_prompt)
        self._build_io_tab(client, client_id, tab_group, g, timeline, default_prompt)

        #
        # Instructions tab
        #
        with tab_group.add_tab("Instructions", viser.Icon.INFO_CIRCLE):
            client.gui.add_markdown(INSTRUCTIONS_TAB_MD)

        gui_elements = GuiElements(**{f: getattr(g, f) for f in GuiElements.__dataclass_fields__})
        return gui_elements, timeline_tracks, timeline_data

    def on_text_prompt_update(self, client_id: int, trigger_replan: bool = True, initial_prompt: bool = False):
        """Update text embedding when prompt changes and update timeline prompts."""
        start_time = time.time()
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        if session.model is None:
            return

        text_prompt = session.gui_elements.gui_prompt_text.value
        text_feat, _ = session.model.text_encoder([text_prompt])
        session.text_embedding = text_feat.to(self.device)

        session.gui_elements.gui_active_prompt_label.content = f"**Active Prompt:** {text_prompt}"

        # Update timeline prompts
        current_frame = max(0, session.frame_idx)
        next_frame = current_frame + 1

        if session.timeline_data is not None and hasattr(client, "timeline"):
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])

            # Update the last prompt to end at current frame
            if len(prompt_uuid_list) > 0:
                last_uuid = prompt_uuid_list[-1]
                try:
                    client.timeline.update_prompt(last_uuid, end_frame=current_frame)
                    print(f"Updated prompt '{last_uuid}' to end at frame {current_frame}")
                except (AttributeError, Exception) as e:
                    print(f"Could not update prompt end frame: {e}")

            # Add new prompt starting from next frame with unique color
            try:
                prompt_counter = session.timeline_data.get("prompt_counter", 1)
                prompt_color = self.get_prompt_color(prompt_counter)

                new_uuid = client.timeline.add_prompt(
                    text=text_prompt,
                    start_frame=0 if initial_prompt else next_frame,
                    end_frame=INFINITE_FRAME_IDX,
                    color=prompt_color,
                )
                prompt_uuid_list.append(new_uuid)
                session.timeline_data["prompt_counter"] = prompt_counter + 1
                print(
                    f"Added new prompt '{new_uuid}' starting at frame {next_frame}: '{text_prompt}' (color: {prompt_color})"
                )
            except (AttributeError, Exception) as e:
                print(f"Could not add new prompt to timeline: {e}")

        session.client.add_notification(
            title="Text prompt updated",
            body=f"New prompt starts at frame {next_frame}",
            auto_close_seconds=3.0,
            color="blue",
        )

        end_time = time.time()
        print(f"Time taken to update text prompt: {end_time - start_time} seconds")

        if trigger_replan:
            threading.Thread(target=self.on_replan_trigger, args=(client_id,), daemon=True).start()
