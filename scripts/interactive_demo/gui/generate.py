# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Generate tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiGenerateMixin:
    def _build_generate_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Generate", viser.Icon.WALK):
            g.gui_restart_button = client.gui.add_button("Restart", color="orange")
            g.gui_restart_from_now_button = client.gui.add_button(
                "Restart From Now",
                icon=viser.Icon.PLAYER_SKIP_FORWARD,
                color="green",
            )

            g.gui_clear_all_constraints_button = client.gui.add_button("Clear All Constraints", color="red")

            @g.gui_clear_all_constraints_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.clear_constraints(event.client.client_id)

            # Initial Body Transform folder
            with client.gui.add_folder("Initial Body Transform", expand_by_default=False):
                g.gui_show_transform_gizmo_checkbox = client.gui.add_checkbox(
                    "Show Transform Gizmo",
                    initial_value=False,
                    hint="Show/hide the transform control gizmo for initial body pose",
                )
                g.gui_reset_transform_button = client.gui.add_button("Reset Transform")

                @g.gui_show_transform_gizmo_checkbox.on_update
                def _(_) -> None:
                    if not self.client_active(client_id):
                        return
                    session = self.client_sessions[client_id]
                    if session.transform_gizmo is not None:
                        session.transform_gizmo.visible = g.gui_show_transform_gizmo_checkbox.value

                @g.gui_reset_transform_button.on_click
                def _(_) -> None:
                    if not self.client_active(client_id):
                        return
                    session = self.client_sessions[client_id]

                    # Reset to origin and zero heading. Must stay a numpy array:
                    # the fresh-generation path feeds it to torch.from_numpy.
                    session.init_global_translation = np.zeros(3, dtype=np.float32)
                    session.init_first_heading_angle = 0.0

                    # Update gizmo position
                    if session.transform_gizmo is not None:
                        session.transform_gizmo.position = (0.0, 0.0, 0.0)
                        session.transform_gizmo.wxyz = viser.transforms.SO3.from_y_radians(0.0).wxyz

                    self._update_start_direction_marker(client_id)

                    client.add_notification(
                        title="Transform Reset",
                        body="Initial body transform reset to origin",
                        auto_close_seconds=2.0,
                    )

            # Constraints and Keyframe loading from file
            with client.gui.add_folder("Constraints", expand_by_default=False):
                # Motion file path for constraint loading
                default_motion_path = (
                    "datasets/bones-seed/g1/csv/230306/jog_ff_loop_180_R_001__A244.csv"
                    if "g1" in DEFAULT_MODEL_DIR
                    else "datasets/bones-seed/soma_uniform/bvh/230306/jog_ff_loop_180_R_001__A244.bvh"
                )
                g.gui_motion_file_path = client.gui.add_text(
                    "Motion File Path",
                    initial_value=default_motion_path,
                    hint="Path to BVH (soma) or CSV (g1) motion file for constraint sampling",
                )
                g.gui_random_motion_button = client.gui.add_button(
                    "Random Motion File",
                    hint="Pick a random motion file from datasets/bones-seed/",
                )

                @g.gui_random_motion_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    # Match the skeleton family of the *currently loaded* model.
                    # G1 models use the CSV subset, others use the BVH subset.
                    # Fall back to DEFAULT_MODEL_DIR if no model is loaded yet.
                    is_g1 = "g1" in DEFAULT_MODEL_DIR
                    session = self.client_sessions.get(client_id)
                    if session is not None and session.motion_rep is not None:
                        is_g1 = is_g1_skeleton(session.motion_rep.skeleton)
                    skeleton_key = "g1" if is_g1 else "soma"
                    candidates = self._bones_seed_paths_by_skeleton.get(skeleton_key, [])

                    if not candidates:
                        if event.client:
                            event.client.add_notification(
                                title="No motion files available",
                                body=("Metadata CSV missing or empty — see startup logs."),
                                color="red",
                                auto_close_seconds=3.0,
                            )
                        return

                    picked = random.choice(candidates)
                    g.gui_motion_file_path.value = picked
                    if event.client:
                        event.client.add_notification(
                            title="Random Motion File",
                            body=os.path.basename(picked),
                            color="blue",
                            auto_close_seconds=2.0,
                        )

                g.gui_crop_motion_checkbox = client.gui.add_checkbox(
                    "Crop to 10s",
                    initial_value=True,
                    hint="Randomly crop loaded motion to 10 seconds",
                )

                # Constraint type checkboxes
                g.gui_constraint_fullbody_checkbox = client.gui.add_checkbox("Full Body", initial_value=True)
                g.gui_constraint_hands_checkbox = client.gui.add_checkbox("Hands", initial_value=False)
                g.gui_constraint_feet_checkbox = client.gui.add_checkbox("Feet", initial_value=False)
                g.gui_constraint_hands_feet_checkbox = client.gui.add_checkbox("Hands and Feet", initial_value=False)
                g.gui_constraint_2d_waypoints_checkbox = client.gui.add_checkbox(
                    "2D Root Waypoints", initial_value=False
                )
                g.gui_constraint_2d_trajectory_checkbox = client.gui.add_checkbox(
                    "2D Root Trajectory", initial_value=False
                )

                g.gui_max_keyframe_num = client.gui.add_number(
                    "Max Keyframes",
                    initial_value=1,
                    min=1,
                    max=20,
                    step=1,
                    hint="Maximum number of keyframes to sample from sequence",
                )

                g.gui_continue_from_current_checkbox = client.gui.add_checkbox(
                    "Continue from Current Frame",
                    initial_value=False,
                    hint="If enabled, transform loaded sequence to continue from current position and heading",
                )

                g.gui_load_seq_button = client.gui.add_button("Sample Constraints", color="green")

                @g.gui_load_seq_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    if not self.client_active(client_id):
                        return
                    session = self.client_sessions[client_id]

                    if session.motion_rep is None:
                        if event.client:
                            event.client.add_notification(
                                title="No model loaded",
                                body="Please load a model first.",
                                color="red",
                            )
                        return

                    if not skeleton_supports_constraint_sampling(session.motion_rep.skeleton):
                        if event.client:
                            event.client.add_notification(
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

                    # Load motion from file path
                    file_path = g.gui_motion_file_path.value.strip()
                    if not file_path:
                        if event.client:
                            event.client.add_notification(
                                title="No file path",
                                body="Please enter a motion file path.",
                                color="red",
                            )
                        return

                    if not os.path.exists(file_path):
                        if event.client:
                            event.client.add_notification(
                                title="File not found",
                                body=f"Motion file not found: {file_path}",
                                color="red",
                            )
                        return

                    t_start = time.time()

                    try:
                        seq_data = self.load_motion_from_file(
                            file_path, session, crop_10s=g.gui_crop_motion_checkbox.value
                        )
                    except Exception as e:
                        if event.client:
                            event.client.add_notification(
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
                        if event.client:
                            event.client.add_notification(
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
                    if event.client:
                        mode_str = "continued from" if continue_from_current else "loaded"
                        constraint_str = ", ".join(constraint_types)
                        event.client.add_notification(
                            title="Constraints Sampled",
                            body=f"{mode_str.title()} from {os.path.basename(file_path)} with {constraint_str} ({elapsed:.2f}s)",
                            auto_close_seconds=3.0,
                            color="green",
                        )

            # Waypoint controls
            with client.gui.add_folder("Waypoint", expand_by_default=False):
                g.gui_waypoint_mode_checkbox = client.gui.add_checkbox("Enable Waypoint Mode", initial_value=False)
                g.gui_dense_root_checkbox = client.gui.add_checkbox("Use Dense Root", initial_value=False)
                g.gui_waypoint_interval = client.gui.add_number(
                    "Waypoint Interval", initial_value=60, min=1, max=300, step=1
                )

            with client.gui.add_folder("Target Velocity", expand_by_default=False):
                # Target root velocity control
                g.gui_use_target_velocity_checkbox = client.gui.add_checkbox("Use Target Velocity", initial_value=False)
                g.gui_target_root_velocity = client.gui.add_vector2(
                    "Target Root Velocity (xz)",
                    initial_value=(0.0, 0.0),
                    min=(-5.0, -5.0),
                    max=(5.0, 5.0),
                    step=0.001,
                    hint="Target 2D root velocity in XZ plane (forward/backward, left/right)",
                    disabled=True,  # Initially disabled until checkbox is checked
                )
                g.gui_use_target_heading_checkbox = client.gui.add_checkbox(
                    "Use Target Heading",
                    initial_value=False,
                    hint="If enabled, calculate root heading from target velocity direction",
                )

                # Flag to prevent infinite loop when loading velocity
                _loading_target_velocity = [False]

                @g.gui_target_root_velocity.on_update
                def _(_) -> None:
                    # Update target velocity arrow when value changes
                    # Skip if we're currently loading the velocity to avoid loop
                    if _loading_target_velocity[0]:
                        return
                    if not self.client_active(client.client_id):
                        return
                    session = self.client_sessions[client.client_id]
                    if session.target_velocity_arrow is not None and g.gui_use_target_velocity_checkbox.value:
                        # Trigger a frame update to refresh the arrow
                        self.set_frame(client.client_id, session.frame_idx)

                @g.gui_use_target_velocity_checkbox.on_update
                def _(_) -> None:
                    # Enable/disable target velocity input based on checkbox
                    g.gui_target_root_velocity.disabled = not g.gui_use_target_velocity_checkbox.value
                    # Update target velocity arrow visibility
                    if not self.client_active(client.client_id):
                        return
                    session = self.client_sessions[client.client_id]

                    # Target velocity and waypoint/dense-root are conflicting
                    # modes; turning on target velocity must disable the others
                    # (each set triggers its own on_update to keep state and
                    # visuals in sync — e.g. hides the click plane).
                    if g.gui_use_target_velocity_checkbox.value:
                        if g.gui_waypoint_mode_checkbox.value:
                            g.gui_waypoint_mode_checkbox.value = False
                        if g.gui_dense_root_checkbox.value:
                            g.gui_dense_root_checkbox.value = False

                    # Keep the timeline arrow-key overlay in sync, like the "t"
                    # shortcut does: driving the checkbox runs its on_update,
                    # which calls configure_arrow_key_overlay.
                    if g.gui_show_timeline_arrow_keys_checkbox.value != g.gui_use_target_velocity_checkbox.value:
                        g.gui_show_timeline_arrow_keys_checkbox.value = g.gui_use_target_velocity_checkbox.value

                    if g.gui_use_target_velocity_checkbox.value:
                        # When enabled, load current frame's root velocity
                        if (
                            session.root_velocities is not None
                            and session.frame_idx >= 0
                            and session.frame_idx < session.root_velocities.shape[1]
                        ):
                            # Get root velocity for first character at current frame
                            root_vel = session.root_velocities[0, session.frame_idx].cpu().numpy()
                            # Set flag to prevent the on_update callback from triggering
                            _loading_target_velocity[0] = True
                            g.gui_target_root_velocity.value = (
                                float(root_vel[0]),
                                float(root_vel[2]),
                            )
                            _loading_target_velocity[0] = False

                    if session.target_velocity_arrow is not None:
                        if not g.gui_use_target_velocity_checkbox.value:
                            # Hide arrow when disabled
                            session.target_velocity_arrow.set_visibility(False)
                        else:
                            # Trigger a frame update to show the arrow if enabled
                            self.set_frame(client.client_id, session.frame_idx)

            with client.gui.add_folder("Post Process", expand_by_default=False):
                g.gui_enable_postprocess_checkbox = client.gui.add_checkbox(
                    "Enable Post-Processing",
                    initial_value=False,
                    hint="Apply motion correction to reduce foot skating and improve quality",
                )
                g.gui_postprocess_root_margin = client.gui.add_slider(
                    "Root Margin",
                    min=0.0,
                    max=0.2,
                    step=0.01,
                    initial_value=0.04,
                    hint="Margin for root position correction (default: 0.04)",
                )
                g.gui_postprocess_contact_threshold = client.gui.add_slider(
                    "Contact Threshold",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial_value=0.5,
                    hint="Threshold for foot contact detection (default: 0.5)",
                )

            @g.gui_restart_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.restart(client_id)
                if event.client:
                    event.client.add_notification(
                        title="Restarted",
                        body="Scene has been reset.",
                        auto_close_seconds=2.0,
                        color="blue",
                    )

            @g.gui_restart_from_now_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.restart_from_now(client_id)
                if event.client:
                    event.client.add_notification(
                        title="Restarted From Now",
                        body=f"Cleared motion after frame {self.client_sessions[client_id].frame_idx} and triggered generation.",
                        auto_close_seconds=2.0,
                        color="blue",
                    )

            @g.gui_waypoint_mode_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                session.waypoint_mode = g.gui_waypoint_mode_checkbox.value
                # Toggle click plane visibility (clicks only register while visible)
                if session.click_plane is not None:
                    session.click_plane.visible = session.waypoint_mode

            @g.gui_dense_root_checkbox.on_update
            def _(_) -> None:
                if not self.client_active(client_id):
                    return
                session = self.client_sessions[client_id]
                session.constraints["2D Root"].set_dense_path(g.gui_dense_root_checkbox.value)
                session.constraints["2D Root"].set_smooth_path(g.gui_dense_root_checkbox.value)

        #
        # Visualization tab
        #
