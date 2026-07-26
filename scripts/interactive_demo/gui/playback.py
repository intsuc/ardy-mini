# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Playback tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiPlaybackMixin:
    def _build_playback_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Playback", viser.Icon.PLAYER_PLAY):
            g.gui_play_pause_button = client.gui.add_button("Play", disabled=False)
            g.gui_next_frame_button = client.gui.add_button(
                "Next Frame", disabled=False, icon=viser.Icon.PLAYER_TRACK_NEXT_FILLED
            )
            g.gui_prev_frame_button = client.gui.add_button(
                "Prev Frame", disabled=True, icon=viser.Icon.PLAYER_TRACK_PREV_FILLED
            )
            g.gui_actual_fps = client.gui.add_number("Actual FPS", initial_value=30.0, step=0.0001, disabled=True)
            g.gui_current_time = client.gui.add_number("Current Time (s)", initial_value=0.0, step=0.01, disabled=True)
            g.gui_model_fps = client.gui.add_number("Native FPS", initial_value=30, disabled=True)
            g.gui_frame_idx_input = client.gui.add_number("Frame Index", initial_value=0, min=0, max=199, step=1)
            g.gui_enable_auto_replan_checkbox = client.gui.add_checkbox("Enable Auto Replan", initial_value=True)

            @g.gui_frame_idx_input.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                target_frame = int(g.gui_frame_idx_input.value)
                # Clamp to valid range
                target_frame = max(0, min(target_frame, session.max_frame_idx))
                if target_frame != session.frame_idx:
                    self.set_frame(client_id, target_frame)
                    # Update button states
                    g.gui_prev_frame_button.disabled = target_frame == 0
                    g.gui_next_frame_button.disabled = target_frame == session.max_frame_idx

            @g.gui_play_pause_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                session.playing = not session.playing
                g.gui_play_pause_button.label = "Pause" if session.playing else "Play"
                g.gui_next_frame_button.disabled = session.playing
                g.gui_prev_frame_button.disabled = session.playing

            @g.gui_next_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.frame_idx < session.max_frame_idx:
                    new_frame = session.frame_idx + 1
                    self.set_frame(client_id, new_frame)
                    g.gui_prev_frame_button.disabled = False
                    if new_frame == session.max_frame_idx:
                        g.gui_next_frame_button.disabled = True

            @g.gui_prev_frame_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.frame_idx > 0:
                    new_frame = session.frame_idx - 1
                    self.set_frame(client_id, new_frame)
                    g.gui_next_frame_button.disabled = False
                    if new_frame == 0:
                        g.gui_prev_frame_button.disabled = True

        #
        # Text tab
        #
