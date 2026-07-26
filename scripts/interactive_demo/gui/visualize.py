# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Visualize tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiVisualizeMixin:
    def _build_visualize_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Visualize", viser.Icon.EYE):
            g.gui_viz_skinned_mesh_checkbox = client.gui.add_checkbox("Show Mesh", initial_value=True)
            g.gui_viz_skinned_mesh_opacity_slider = client.gui.add_slider(
                "Mesh Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
            )
            g.gui_viz_skeleton_checkbox = client.gui.add_checkbox("Show Skeleton", initial_value=False)
            g.gui_viz_foot_contacts_checkbox = client.gui.add_checkbox(
                "Show Foot Contacts",
                initial_value=False,
                hint="Color foot joints purple when predicted to be in contact with the ground",
            )
            g.gui_viz_foot_contacts_checkbox.visible = g.gui_viz_skeleton_checkbox.value
            g.gui_viz_ref_motion_checkbox = client.gui.add_checkbox(
                "Show Reference Motion",
                initial_value=False,
                hint="Show loaded reference motion as a red mesh character",
            )

            @g.gui_viz_ref_motion_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                show = g.gui_viz_ref_motion_checkbox.value
                if session.ref_character is not None:
                    session.ref_character.set_skinned_mesh_visibility(show)
                    if session.ref_character.skeleton_mesh is not None:
                        session.ref_character.skeleton_mesh.set_visibility(False)
                elif show and session.ref_joints_pos is not None:
                    # Create reference character on first toggle
                    self._create_ref_character(client_id)

            g.gui_viz_hand_orientations_checkbox = client.gui.add_checkbox(
                "Show Hand+Foot Orientations",
                initial_value=False,
                hint="Show rotation axes for hand/wrist and foot joints in both EE constraints and generated motion",
            )
            g.gui_viz_hide_distant_constraints_checkbox = client.gui.add_checkbox(
                "Hide Distant Future Constraints",
                initial_value=False,
                hint="Hide constraints beyond future_crop + generate_horizon frames from current frame",
            )
            g.gui_viz_auto_camera_checkbox = client.gui.add_checkbox("Auto Camera Follow", initial_value=False)
            g.gui_viz_camera_type_dropdown = client.gui.add_dropdown(
                "Camera Type",
                options=["Over-the-shoulder", "Front-facing"],
                initial_value="Front-facing",
            )
            g.gui_show_timeline_checkbox = client.gui.add_checkbox(
                "Show Timeline",
                initial_value=True,
                hint="Show/hide the timeline strip at the bottom of the viewer",
            )
            g.gui_show_start_direction_checkbox = client.gui.add_checkbox(
                "Show Starting Direction",
                initial_value=True,
                hint="Show a blue marker at the initial position and facing direction",
            )
            g.gui_show_timeline_arrow_keys_checkbox = client.gui.add_checkbox(
                "Show Timeline Arrow Keys", initial_value=False
            )
            g.gui_arrow_key_position_dropdown = client.gui.add_dropdown(
                "Arrow Key Position",
                options=["bottom_center", "top_center"],
                initial_value="bottom_center",
            )

            # Camera controls
            client.gui.add_markdown("**Camera Control**")
            g.gui_viz_camera_fov_slider = client.gui.add_slider(
                "FOV (degrees)",
                min=20.0,
                max=120.0,
                step=1.0,
                initial_value=50.0,
                hint="Field of view in degrees",
            )
            g.gui_viz_camera_position = client.gui.add_vector3(
                "Position",
                initial_value=(0.0, 2.0, 5.0),
                step=0.1,
                hint="Camera position in world space",
            )
            g.gui_viz_camera_look_at = client.gui.add_vector3(
                "Look at",
                initial_value=(0.0, 1.0, 0.0),
                step=0.1,
                hint="Point the camera is looking at",
            )
            g.gui_viz_camera_up = client.gui.add_vector3(
                "Up direction",
                initial_value=(0.0, 1.0, 0.0),
                step=0.01,
                hint="Camera up direction vector",
            )
            g.gui_capture_camera_button = client.gui.add_button(
                "Capture Current Camera",
                hint="Capture the current camera view to the controls above",
            )
            g.gui_apply_camera_button = client.gui.add_button(
                "Apply Camera Settings",
                hint="Apply the camera settings from the controls above",
            )

            g.gui_camera_file_path = client.gui.add_text(
                "Camera File",
                initial_value=".cache/camera_params.json",
                hint="Path to save/load camera parameters",
            )
            g.gui_save_camera_button = client.gui.add_button(
                "Save Camera Parameters", hint="Save current camera parameters to file"
            )
            g.gui_load_camera_button = client.gui.add_button(
                "Load Camera Parameters", hint="Load camera parameters from file"
            )

            g.gui_dark_mode_checkbox = client.gui.add_checkbox("Dark Mode", initial_value=False)
            # Hidden from the sidebar: it still drives the theme, but the
            # toggle itself renders in the titlebar (see configure_theme).
            g.gui_dark_mode_checkbox.visible = False

            @g.gui_show_timeline_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                client = session.client
                # Same degrade-gracefully guard as gui/orchestrator.py:24-48 —
                # older viser builds without timeline support won't have this.
                if hasattr(client, "timeline"):
                    client.timeline.set_visible(g.gui_show_timeline_checkbox.value)
                else:
                    print("Timeline not available, cannot toggle visibility")

            @g.gui_show_start_direction_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                self._update_start_direction_marker(client_id)

            @g.gui_show_timeline_arrow_keys_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                client = session.client
                show_timeline_arrow_keys = g.gui_show_timeline_arrow_keys_checkbox.value
                arrow_key_position = g.gui_arrow_key_position_dropdown.value
                client.timeline.configure_arrow_key_overlay(
                    enabled=show_timeline_arrow_keys, position=arrow_key_position
                )

            @g.gui_arrow_key_position_dropdown.on_update
            def _(event: viser.GuiEvent) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                client = session.client
                show_timeline_arrow_keys = g.gui_show_timeline_arrow_keys_checkbox.value
                arrow_key_position = g.gui_arrow_key_position_dropdown.value
                client.timeline.configure_arrow_key_overlay(
                    enabled=show_timeline_arrow_keys, position=arrow_key_position
                )

            @g.gui_viz_camera_fov_slider.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                fov_degrees = g.gui_viz_camera_fov_slider.value
                session.client.camera.fov = np.radians(fov_degrees)
                print(f"[Camera] FOV set to {fov_degrees}°")

            @g.gui_viz_camera_position.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                position = g.gui_viz_camera_position.value
                session.client.camera.position = np.array(position, dtype=np.float64)
                print(f"[Camera] Position set to {position}")

            @g.gui_viz_camera_look_at.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                look_at = g.gui_viz_camera_look_at.value
                session.client.camera.look_at = np.array(look_at, dtype=np.float64)
                print(f"[Camera] Look at set to {look_at}")

            @g.gui_viz_camera_up.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                up = g.gui_viz_camera_up.value
                session.client.camera.up_direction = np.array(up, dtype=np.float64)
                print(f"[Camera] Up direction set to {up}")

            @g.gui_capture_camera_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                camera = session.client.camera
                # Capture current camera state to GUI
                g.gui_viz_camera_fov_slider.value = float(np.degrees(camera.fov))
                g.gui_viz_camera_position.value = tuple(float(x) for x in camera.position)
                g.gui_viz_camera_look_at.value = tuple(float(x) for x in camera.look_at)
                g.gui_viz_camera_up.value = tuple(float(x) for x in camera.up_direction)
                print("[Camera] Captured current camera state")
                session.client.add_notification(
                    title="Camera captured",
                    body="Current camera view saved to controls",
                    auto_close_seconds=2.0,
                    color="green",
                )

            @g.gui_apply_camera_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                # Apply all camera settings
                fov_degrees = g.gui_viz_camera_fov_slider.value
                position = g.gui_viz_camera_position.value
                look_at = g.gui_viz_camera_look_at.value
                up = g.gui_viz_camera_up.value

                session.client.camera.fov = np.radians(fov_degrees)
                session.client.camera.position = np.array(position, dtype=np.float64)
                session.client.camera.look_at = np.array(look_at, dtype=np.float64)
                session.client.camera.up_direction = np.array(up, dtype=np.float64)

                print(f"[Camera] Applied settings - FOV: {fov_degrees}°, Pos: {position}, Look at: {look_at}, Up: {up}")
                session.client.add_notification(
                    title="Camera applied",
                    body="Camera settings applied to view",
                    auto_close_seconds=2.0,
                    color="green",
                )

            @g.gui_save_camera_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                filepath = g.gui_camera_file_path.value

                # Get current camera parameters from GUI controls
                camera_params = {
                    "fov_degrees": float(g.gui_viz_camera_fov_slider.value),
                    "position": list(g.gui_viz_camera_position.value),
                    "look_at": list(g.gui_viz_camera_look_at.value),
                    "up": list(g.gui_viz_camera_up.value),
                }

                try:
                    # Create directory if it doesn't exist
                    import os

                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    # Save to JSON file
                    with open(filepath, "w") as f:
                        json.dump(camera_params, f, indent=2)

                    print(f"[Camera] Saved camera parameters to {filepath}")
                    session.client.add_notification(
                        title="Camera saved",
                        body=f"Parameters saved to {filepath}",
                        auto_close_seconds=3.0,
                        color="green",
                    )
                except Exception as e:
                    print(f"[Camera] Error saving camera parameters: {e}")
                    session.client.add_notification(
                        title="Save failed",
                        body=f"Error: {str(e)}",
                        auto_close_seconds=5.0,
                        color="red",
                    )

            @g.gui_load_camera_button.on_click
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                filepath = g.gui_camera_file_path.value

                try:
                    # Load from JSON file
                    with open(filepath, "r") as f:
                        camera_params = json.load(f)

                    # Update GUI controls
                    g.gui_viz_camera_fov_slider.value = float(camera_params["fov_degrees"])
                    g.gui_viz_camera_position.value = tuple(camera_params["position"])
                    g.gui_viz_camera_look_at.value = tuple(camera_params["look_at"])
                    g.gui_viz_camera_up.value = tuple(camera_params["up"])

                    # Apply to camera
                    session.client.camera.fov = np.radians(camera_params["fov_degrees"])
                    session.client.camera.position = np.array(camera_params["position"], dtype=np.float64)
                    session.client.camera.look_at = np.array(camera_params["look_at"], dtype=np.float64)
                    session.client.camera.up_direction = np.array(camera_params["up"], dtype=np.float64)

                    print(f"[Camera] Loaded camera parameters from {filepath}")
                    session.client.add_notification(
                        title="Camera loaded",
                        body=f"Parameters loaded from {filepath}",
                        auto_close_seconds=3.0,
                        color="green",
                    )
                except FileNotFoundError:
                    print(f"[Camera] File not found: {filepath}")
                    session.client.add_notification(
                        title="Load failed",
                        body=f"File not found: {filepath}",
                        auto_close_seconds=5.0,
                        color="red",
                    )
                except Exception as e:
                    print(f"[Camera] Error loading camera parameters: {e}")
                    session.client.add_notification(
                        title="Load failed",
                        body=f"Error: {str(e)}",
                        auto_close_seconds=5.0,
                        color="red",
                    )

            @g.gui_dark_mode_checkbox.on_update
            def _(_) -> None:
                self.configure_theme(
                    client,
                    dark_mode=g.gui_dark_mode_checkbox.value,
                    dark_mode_checkbox_uuid=g.gui_dark_mode_checkbox.uuid,
                )
                if self.client_active(client_id):
                    session = self.client_sessions[client_id]
                    with session.characters_lock:
                        for character in session.characters.values():
                            character.change_theme(g.gui_dark_mode_checkbox.value)

            @g.gui_viz_skeleton_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                g.gui_viz_foot_contacts_checkbox.visible = g.gui_viz_skeleton_checkbox.value
                if not g.gui_viz_skeleton_checkbox.value:
                    g.gui_viz_foot_contacts_checkbox.value = False
                with session.characters_lock:
                    for character in session.characters.values():
                        character.set_skeleton_visibility(g.gui_viz_skeleton_checkbox.value)

            @g.gui_viz_foot_contacts_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                with session.characters_lock:
                    for character in session.characters.values():
                        character.set_show_foot_contacts(g.gui_viz_foot_contacts_checkbox.value)
                if session.frame_idx >= 0:
                    self.set_frame(client_id, session.frame_idx)

            @g.gui_viz_hand_orientations_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]

                if g.gui_viz_hand_orientations_checkbox.value:
                    # Create hand gizmos for all characters
                    self.create_hand_gizmos(client_id)
                    # Update them to current frame
                    if session.frame_idx >= 0:
                        self.update_hand_gizmos(client_id, session.frame_idx)
                else:
                    # Hide hand gizmos
                    self.set_hand_gizmos_visibility(client_id, False)

                # Update EE constraint rotation axes visibility
                if "End-Effectors" in session.constraints:
                    ee_constraint = session.constraints["End-Effectors"]
                    for constraint_frame_idx in ee_constraint.scene_elements.keys():
                        # Only update visibility if the constraint is already visible
                        visibility = constraint_frame_idx >= session.frame_idx if session.frame_idx >= 0 else True
                        ee_constraint.set_keyframe_visibility(
                            constraint_frame_idx,
                            visibility,
                            show_rotation_axes=g.gui_viz_hand_orientations_checkbox.value,
                        )

            @g.gui_viz_hide_distant_constraints_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                # Trigger a frame update to refresh constraint visibility
                if session.frame_idx >= 0:
                    self.set_frame(client_id, session.frame_idx)

            @g.gui_viz_auto_camera_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                # Reset camera state when toggling to allow smooth start
                session.camera_position = None
                session.camera_look_at = None
                session.camera_forward_direction = None
                session.camera_position_buffer.clear()
                session.camera_last_update_frame = -1
                if g.gui_viz_auto_camera_checkbox.value:
                    # Immediately update camera when enabled
                    self.update_camera_follow(client_id, session.frame_idx)

            @g.gui_viz_camera_type_dropdown.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                # Reset camera smoothing when changing camera type
                session.camera_position = None
                session.camera_look_at = None
                session.camera_forward_direction = None
                session.camera_position_buffer.clear()
                if g.gui_viz_auto_camera_checkbox.value:
                    # Immediately update camera with new type
                    self.update_camera_follow(client_id, session.frame_idx)

            @g.gui_viz_skinned_mesh_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                with session.characters_lock:
                    for character in session.characters.values():
                        character.set_skinned_mesh_visibility(g.gui_viz_skinned_mesh_checkbox.value)

            @g.gui_viz_skinned_mesh_opacity_slider.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                with session.characters_lock:
                    for character in session.characters.values():
                        character.set_skinned_mesh_opacity(g.gui_viz_skinned_mesh_opacity_slider.value)

        #
        # Model tab
        #
