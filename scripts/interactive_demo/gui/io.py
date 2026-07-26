# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: IO tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403
from .instructions import KEYBOARD_SHORTCUTS_MD


class GuiIOMixin:
    def _build_io_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("IO", viser.Icon.FILE_DOWNLOAD):
            # Session Export/Load Group
            with client.gui.add_folder("Session (Motion + Prompts + Constraints)", expand_by_default=True):
                # Generate default timestamp-based filename
                default_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                g.gui_session_file_path = client.gui.add_text(
                    "Session File Path",
                    initial_value=f".cache/export/session_{default_timestamp}.pkl",
                )
                g.gui_export_session_button = client.gui.add_button("Export Session")
                g.gui_load_session_button = client.gui.add_button("Load Session")

            # Root Constraints Group
            with client.gui.add_folder("Root Constraints", expand_by_default=True):
                g.gui_root_file_path = client.gui.add_text(
                    "Root File Path", initial_value=".cache/root_constraints.json"
                )
                g.gui_load_root_button = client.gui.add_button("Load Root Constraints")
                g.gui_save_root_button = client.gui.add_button("Save Root Constraints")

            # Scene Group
            with client.gui.add_folder("Scene", expand_by_default=True):
                g.gui_scene_file_path = client.gui.add_text("Scene File Path", initial_value=".cache/room.ply")
                g.gui_mesh_transform_dropdown = client.gui.add_dropdown(
                    "Mesh Transform",
                    options=["No Transform", "Z-up to Y-up"],
                    initial_value="No Transform",
                )
                g.gui_load_mesh_button = client.gui.add_button("Load 3D Mesh (.ply/.obj)")

                # Scene translation controls
                g.gui_scene_translation_x = client.gui.add_slider(
                    "Scene Translation X",
                    min=-20.0,
                    max=20.0,
                    step=0.01,
                    initial_value=0.0,
                    hint="Translate loaded scene along X axis",
                )
                g.gui_scene_translation_y = client.gui.add_slider(
                    "Scene Translation Y",
                    min=-20.0,
                    max=20.0,
                    step=0.01,
                    initial_value=0.0,
                    hint="Translate loaded scene along Y axis",
                )
                g.gui_scene_translation_z = client.gui.add_slider(
                    "Scene Translation Z",
                    min=-20.0,
                    max=20.0,
                    step=0.01,
                    initial_value=0.0,
                    hint="Translate loaded scene along Z axis",
                )

                # Viewport image capture
                g.gui_viewport_capture_path = client.gui.add_text(
                    "Viewport Capture Path",
                    initial_value=".cache/image_export/viewport_capture.png",
                )
                g.gui_capture_width = client.gui.add_number(
                    "Image Width",
                    initial_value=1920,
                    min=320,
                    max=7680,
                    step=1,
                    hint="Width of the captured image in pixels",
                )
                g.gui_capture_height = client.gui.add_number(
                    "Image Height",
                    initial_value=1080,
                    min=240,
                    max=4320,
                    step=1,
                    hint="Height of the captured image in pixels",
                )
                g.gui_capture_viewport_button = client.gui.add_button("Capture Viewport Image")

            @g.gui_export_session_button.on_click
            def _(event: viser.GuiEvent) -> None:
                filepath = g.gui_session_file_path.value
                success = self.export_session(client_id, filepath)
                if event.client:
                    if success:
                        event.client.add_notification(
                            title="Session Exported",
                            body=f"Saved to {filepath}",
                            auto_close_seconds=3.0,
                            color="green",
                        )
                    else:
                        event.client.add_notification(
                            title="Export Failed",
                            body="Failed to export session. Check console for details.",
                            auto_close_seconds=5.0,
                            color="red",
                        )

            @g.gui_load_session_button.on_click
            def _(event: viser.GuiEvent) -> None:
                filepath = g.gui_session_file_path.value

                # Run in separate thread to avoid blocking
                def load_thread():
                    success = self.load_session(client_id, filepath)
                    # Notification is sent inside load_session

                thread = threading.Thread(target=load_thread, daemon=True)
                thread.start()

                if event.client:
                    event.client.add_notification(
                        title="Loading Session",
                        body=f"Loading from {filepath}...",
                        auto_close_seconds=2.0,
                        color="blue",
                    )

            @g.gui_load_root_button.on_click
            def _(event: viser.GuiEvent) -> None:
                filepath = g.gui_root_file_path.value
                self.load_root_constraints(client_id, filepath)
                if event.client:
                    event.client.add_notification(
                        title="Root constraints loaded",
                        body=f"Loaded from {filepath}",
                        auto_close_seconds=2.0,
                        color="green",
                    )

            @g.gui_save_root_button.on_click
            def _(event: viser.GuiEvent) -> None:
                filepath = g.gui_root_file_path.value
                self.save_root_constraints(client_id, filepath)
                if event.client:
                    event.client.add_notification(
                        title="Root constraints saved",
                        body=f"Saved to {filepath}",
                        auto_close_seconds=2.0,
                        color="green",
                    )

            @g.gui_load_mesh_button.on_click
            def _(event: viser.GuiEvent) -> None:
                filepath = g.gui_scene_file_path.value
                transform_type = g.gui_mesh_transform_dropdown.value
                self.load_mesh(client_id, filepath, transform_type)
                if event.client:
                    event.client.add_notification(
                        title="Mesh loading",
                        body=f"Loading {filepath}...",
                        auto_close_seconds=2.0,
                        color="blue",
                    )

            @g.gui_scene_translation_x.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.loaded_scene_mesh_handle is not None:
                    session.loaded_scene_mesh_handle.position = (
                        g.gui_scene_translation_x.value,
                        g.gui_scene_translation_y.value,
                        g.gui_scene_translation_z.value,
                    )

            @g.gui_scene_translation_y.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.loaded_scene_mesh_handle is not None:
                    session.loaded_scene_mesh_handle.position = (
                        g.gui_scene_translation_x.value,
                        g.gui_scene_translation_y.value,
                        g.gui_scene_translation_z.value,
                    )

            @g.gui_scene_translation_z.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                if session.loaded_scene_mesh_handle is not None:
                    session.loaded_scene_mesh_handle.position = (
                        g.gui_scene_translation_x.value,
                        g.gui_scene_translation_y.value,
                        g.gui_scene_translation_z.value,
                    )

            @g.gui_capture_viewport_button.on_click
            def _(event: viser.GuiEvent) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                filepath = g.gui_viewport_capture_path.value
                width = int(g.gui_capture_width.value)
                height = int(g.gui_capture_height.value)

                try:
                    # Ensure directory exists
                    import os

                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    print(f"Capturing viewport from client {client_id}...")
                    print(f"  Resolution: {width}x{height}")

                    # Get render from the client's viewport
                    render = session.client.camera.get_render(height=height, width=width)

                    # Save the image
                    from PIL import Image

                    img = Image.fromarray(render)
                    img.save(filepath)

                    if event.client:
                        event.client.add_notification(
                            title="Viewport Captured",
                            body=f"Saved to {filepath} ({img.width}x{img.height})",
                            auto_close_seconds=3.0,
                            color="green",
                        )
                    print(f"Viewport image saved to: {filepath}")
                    print(f"  Resolution: {img.width}x{img.height}")

                except Exception as e:
                    if event.client:
                        event.client.add_notification(
                            title="Capture Failed",
                            body=f"Failed to capture viewport: {str(e)}",
                            auto_close_seconds=5.0,
                            color="red",
                        )
                    print(f"Error capturing viewport image: {e}")
                    import traceback

                    traceback.print_exc()

        #
        # Keyboard controls
        #
        space_pressed = [False]
        arrow_keys_pressed = set()
        help_modal_holder: list = [None]  # tracks the currently-open help modal handle

        # Single handler that processes both keydown and keyup events
        def handle_keyboard(event: viser.KeyboardEvent) -> None:
            """Handle keyboard events for play/pause, frame navigation, and arrow key
            visualization."""
            # Check if client session still exists
            if client_id not in self.client_sessions:
                return

            session = self.client_sessions[client_id]

            # Handle keyup events first
            if event.event_type == "keyup":
                if event.key == " ":
                    space_pressed[0] = False
                elif event.key in ARROW_KEYS:
                    arrow_keys_pressed.discard(event.key)
                    try:
                        # wait for 100ms before updating the timeline
                        time.sleep(0.1)
                        session.client.timeline.set_highlighted_arrow_keys(sorted(arrow_keys_pressed))
                    except (AttributeError, Exception):
                        pass
                return

            # Handle keydown events
            # Space bar: toggle play/pause (only on first press, not repeat)
            if event.key == " ":
                if not space_pressed[0]:
                    space_pressed[0] = True
                    session.playing = not session.playing
                    g.gui_play_pause_button.label = "Pause" if session.playing else "Play"
                    g.gui_next_frame_button.disabled = session.playing
                    g.gui_prev_frame_button.disabled = session.playing
                return

            # Handle arrow keys for timeline visualization (not for navigation)
            elif event.key in ARROW_KEYS:
                arrow_keys_pressed.add(event.key)
                try:
                    session.client.timeline.set_highlighted_arrow_keys(sorted(arrow_keys_pressed))
                except (AttributeError, Exception):
                    pass
                self.on_arrow_key_press(client_id, arrow_keys_pressed)
                return

            # j/k keys: frame navigation (with fast OS repeat via debounce)
            elif event.key == "j":
                if session.frame_idx > 0:
                    new_frame = session.frame_idx - 1
                    self.set_frame(client_id, new_frame)
                    g.gui_next_frame_button.disabled = False
                    if new_frame == 0:
                        g.gui_prev_frame_button.disabled = True

            elif event.key == "k":
                if session.frame_idx < session.max_frame_idx:
                    new_frame = session.frame_idx + 1
                    self.set_frame(client_id, new_frame)
                    g.gui_prev_frame_button.disabled = False
                    if new_frame == session.max_frame_idx:
                        g.gui_next_frame_button.disabled = True

            elif event.key == "r":
                # Reset/update camera to follow current frame (without smoothing)
                self.update_camera_follow(client_id, session.frame_idx, use_smoothing=False)
                print(f"Camera reset to follow frame {session.frame_idx} (no smoothing)")

            elif event.key == "p":
                # Toggle waypoint control mode. Drives the checkbox so its
                # on_update handler runs (updates session.waypoint_mode and the
                # click-plane visibility), keeping the GUI and state in sync.
                g.gui_waypoint_mode_checkbox.value = not g.gui_waypoint_mode_checkbox.value
                enabled = g.gui_waypoint_mode_checkbox.value
                status = "enabled" if enabled else "disabled"
                print(f"[Keyboard P] Waypoint control mode {status}")
                session.client.add_notification(
                    title="Waypoint Mode",
                    body=("Click in the viewport to place waypoints." if enabled else "Waypoint placement off."),
                    auto_close_seconds=2.0,
                    color="blue" if enabled else "gray",
                )

            elif event.key == "t":
                # Toggle target velocity control along with target heading and arrow key visualization
                current_state = g.gui_use_target_velocity_checkbox.value
                new_state = not current_state

                # Toggle all three controls together
                g.gui_use_target_velocity_checkbox.value = new_state
                g.gui_use_target_heading_checkbox.value = new_state
                g.gui_show_timeline_arrow_keys_checkbox.value = new_state

                # Update target velocity input enabled/disabled state
                g.gui_target_root_velocity.disabled = not new_state

                # Update arrow key visualization
                try:
                    arrow_key_position = g.gui_arrow_key_position_dropdown.value
                    session.client.timeline.configure_arrow_key_overlay(enabled=new_state, position=arrow_key_position)
                except (AttributeError, Exception):
                    pass

                # Update target velocity arrow visibility
                if session.target_velocity_arrow is not None:
                    if not new_state:
                        # Hide arrow when disabled
                        session.target_velocity_arrow.set_visibility(False)
                    else:
                        # Load current velocity when enabled
                        if (
                            session.root_velocities is not None
                            and session.frame_idx >= 0
                            and session.frame_idx < session.root_velocities.shape[1]
                        ):
                            # Get root velocity for first character at current frame
                            root_vel = session.root_velocities[0, session.frame_idx].cpu().numpy()
                            g.gui_target_root_velocity.value = (
                                float(root_vel[0]),
                                float(root_vel[2]),
                            )
                        # Trigger a frame update to show the arrow
                        self.set_frame(client_id, session.frame_idx)

                status = "enabled" if new_state else "disabled"
                print(f"[Keyboard T] Target velocity control {status}")
                print(f"  - Target velocity: {status}")
                print(f"  - Target heading: {status}")
                print(f"  - Arrow key visualization: {status}")

            elif event.key == "z":
                # Trigger sample constraints (same as clicking the "Sample Constraints" button)
                print(f"[Keyboard Z] Triggering sample constraints")

                # Call the same logic as the "Sample Constraints" button
                if not self.client_active(client_id):
                    return

                if session.motion_rep is None:
                    client.add_notification(
                        title="No model loaded",
                        body="Please load a model first.",
                        color="red",
                    )
                    return

                if not skeleton_supports_constraint_sampling(session.motion_rep.skeleton):
                    client.add_notification(
                        title="No dataset for this skeleton",
                        body=(
                            "The Core skeleton has no companion dataset for "
                            "constraint sampling. Load a G1 model to use "
                            "this feature."
                        ),
                        color="orange",
                        auto_close_seconds=4.0,
                    )
                    return

                t_start = time.time()

                file_path = g.gui_motion_file_path.value.strip()
                if not file_path or not os.path.exists(file_path):
                    client.add_notification(
                        title="Invalid file path",
                        body=f"Motion file not found: {file_path}",
                        color="red",
                    )
                    return

                try:
                    seq_data = self.load_motion_from_file(file_path, session, crop_10s=g.gui_crop_motion_checkbox.value)
                except Exception as e:
                    client.add_notification(
                        title="Error loading motion",
                        body=f"Failed to load motion: {str(e)}",
                        color="red",
                    )
                    import traceback

                    traceback.print_exc()
                    return

                # Collect selected constraint types
                constraint_types = []
                if g.gui_constraint_fullbody_checkbox.value:
                    constraint_types.append("Full Body")
                if g.gui_constraint_hands_checkbox.value:
                    constraint_types.append("Hands")
                if g.gui_constraint_feet_checkbox.value:
                    constraint_types.append("Feet")
                if g.gui_constraint_hands_feet_checkbox.value:
                    constraint_types.append("Hands and Feet")
                if g.gui_constraint_2d_waypoints_checkbox.value:
                    constraint_types.append("2D Root Waypoints")
                if g.gui_constraint_2d_trajectory_checkbox.value:
                    constraint_types.append("2D Root Trajectory")

                if len(constraint_types) == 0:
                    client.add_notification(
                        title="No constraint types selected",
                        body="Please select at least one constraint type.",
                        color="red",
                    )
                    return

                continue_from_current = g.gui_continue_from_current_checkbox.value
                self.load_sequence(
                    client_id,
                    seq_data,
                    constraint_types=constraint_types,
                    continue_from_current=continue_from_current,
                    update_text=bool(seq_data.get("text")),
                )

                elapsed = time.time() - t_start
                mode_str = "continued from" if continue_from_current else "loaded"
                constraint_str = ", ".join(constraint_types)
                client.add_notification(
                    title="Constraints Sampled",
                    body=f"{mode_str.title()} from {os.path.basename(file_path)} with {constraint_str} ({elapsed:.2f}s)",
                    auto_close_seconds=3.0,
                    color="green",
                )

            elif event.key == "h":
                # Toggle a modal listing all keyboard shortcuts.
                if help_modal_holder[0] is not None:
                    try:
                        help_modal_holder[0].close()
                    except Exception:
                        pass
                    help_modal_holder[0] = None
                    return
                with session.client.gui.add_modal(
                    "Keyboard Shortcuts",
                    show_close_button=True,
                    size="lg",
                ) as modal:
                    help_modal_holder[0] = modal
                    session.client.gui.add_markdown(KEYBOARD_SHORTCUTS_MD)

        # Register the keyboard handler
        client.scene.on_keyboard_event("keydown", debounce_ms=100)(handle_keyboard)
        client.scene.on_keyboard_event("keyup", debounce_ms=100)(handle_keyboard)

        # Create GUI elements dataclass
