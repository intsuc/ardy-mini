# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class PlaybackMixin:
    def run_client_playback(self, client_id: int):
        """Playback loop for a specific client."""
        print(f"Starting playback loop for client {client_id}")

        elapsed_history = []

        while True:
            # Check if client is still active and should continue
            if not self.client_active(client_id):
                print(f"Client {client_id} no longer active, stopping playback loop")
                break

            session = self.client_sessions[client_id]
            if session.stop_playback:
                print(f"Stop signal received for client {client_id}")
                break

            last_update_time = time.time()

            # Update frame if playing
            if session.playing:
                if session.frame_idx < session.max_frame_idx:
                    session.frame_idx += 1
                    self.set_frame(client_id, session.frame_idx)

            # Sleep to maintain target FPS (using model's native FPS)
            time_remaining = max(0, 1.0 / session.model_fps - (time.time() - last_update_time))
            time.sleep(time_remaining)

            # Track moving average of actual fps
            elapsed = time.time() - last_update_time
            elapsed_history.append(elapsed)
            if len(elapsed_history) > 10:
                elapsed_history.pop(0)

            if self.client_active(client_id):
                session.gui_elements.gui_actual_fps.value = 1.0 / (sum(elapsed_history) / len(elapsed_history))

        print(f"Playback loop ended for client {client_id}")

    def run(self):
        """Main dummy loop to keep server alive."""
        print("Main server loop started")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Server shutting down...")
            # Signal all playback threads to stop
            for session in self.client_sessions.values():
                session.stop_playback = True

    def set_frame(self, client_id: int, frame_idx: int, trigger_by_gui_timeline: bool = False):
        """Set the current frame for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        session.frame_idx = frame_idx
        # Update the Viser timeline GUI if not triggered by it
        if not trigger_by_gui_timeline and hasattr(client, "timeline"):
            try:
                client.timeline.set_current_frame(frame_idx)
                # Set a rolling window: 20 frames before + 200 frames after current frame
                window_start = max(0, frame_idx - TIMELINE_WINDOW_BEFORE)
                window_end = frame_idx + TIMELINE_WINDOW_AFTER
                client.timeline.set_frame_range(start_frame=window_start, end_frame=window_end)
            except (AttributeError, Exception) as e:
                print(f"Could not update timeline frame: {e}")

        session.cur_time = frame_idx / session.model_fps
        session.gui_elements.gui_current_time.value = session.cur_time
        session.gui_elements.gui_frame_idx_input.value = frame_idx

        # Check if approaching end of timeline
        thresh = session.gui_elements.gui_replan_trigger_thresh.value
        enable_auto_replan = session.gui_elements.gui_enable_auto_replan_checkbox.value
        if not trigger_by_gui_timeline and enable_auto_replan and session.max_frame_idx - frame_idx <= thresh:
            # Cheap pre-check to avoid spawning a thread while a replan runs;
            # skip_if_busy makes the trigger drop atomically if another thread
            # won the race between this check and the lock acquisition.
            if not session.replan_lock.locked():
                threading.Thread(
                    target=self.on_replan_trigger,
                    args=(client_id,),
                    kwargs={"skip_if_busy": True},
                    daemon=True,
                ).start()

        # Update constraint visibility - show constraints that have been reached (frame_idx <= current frame)
        show_hand_orientations = session.gui_elements.gui_viz_hand_orientations_checkbox.value
        hide_distant_constraints = session.gui_elements.gui_viz_hide_distant_constraints_checkbox.value

        # Calculate max visible future frame if hiding distant constraints
        max_future_frame = float("inf")
        if hide_distant_constraints:
            future_crop = session.gui_elements.gui_future_crop_length.value
            gen_horizon = session.gen_horizon_len
            max_future_frame = frame_idx + future_crop + gen_horizon

        for track_name, constraint in session.constraints.items():
            for constraint_frame_idx in constraint.scene_elements.keys():
                # Basic visibility: constraint is in the future (not yet reached)
                visibility = constraint_frame_idx >= frame_idx

                # Additionally hide if too far in the future
                if hide_distant_constraints and constraint_frame_idx > max_future_frame:
                    visibility = False

                # Pass show_rotation_axes parameter for End-Effectors constraints
                if track_name == "End-Effectors":
                    constraint.set_keyframe_visibility(
                        constraint_frame_idx,
                        visibility,
                        show_rotation_axes=show_hand_orientations,
                    )
                else:
                    constraint.set_keyframe_visibility(constraint_frame_idx, visibility)
            # Update interval labels visibility once per constraint track
            if track_name == "2D Root":
                constraint.set_interval_labels_visibility(frame_idx)

        # Update character poses and get root position for target velocity arrow
        # Note: Each character has its own actual velocity arrow (blue) shown on their skeleton
        # We also have ONE shared target velocity arrow (orange) for the entire session
        root_pos_for_target = None
        with session.characters_lock:
            if frame_idx >= 0 and frame_idx <= session.max_frame_idx:
                for character_idx, character in enumerate(session.characters.values()):
                    # Get actual root velocity for this frame (pass as tensor)
                    root_velocity = None
                    if session.root_velocities is not None:
                        root_velocity = session.root_velocities[character_idx, frame_idx]

                    foot_contacts = (
                        session.foot_contacts[character_idx, frame_idx] > 0.5
                        if session.foot_contacts is not None
                        else None
                    )
                    character.set_pose(
                        session.joints_pos[character_idx, frame_idx],
                        session.joints_rot[character_idx, frame_idx],
                        foot_contacts=foot_contacts,
                        root_velocity=root_velocity,
                    )

                    # Store root position from first character for target velocity visualization
                    if character_idx == 0 and root_pos_for_target is None:
                        root_pos_for_target = character.skeleton_mesh.cur_joints_pos[character.skeleton.root_idx]

        # Update reference motion character
        if (
            session.ref_character is not None
            and session.ref_joints_pos is not None
            and session.gui_elements.gui_viz_ref_motion_checkbox.value
        ):
            ref_frame = min(frame_idx, session.ref_joints_pos.shape[0] - 1)
            if ref_frame >= 0:
                session.ref_character.set_pose(
                    session.ref_joints_pos[ref_frame],
                    session.ref_joints_rot[ref_frame],
                )

        # Update hand gizmos if enabled
        if session.gui_elements.gui_viz_hand_orientations_checkbox.value:
            if not session.hand_gizmos:
                self.create_hand_gizmos(client_id)
            self.update_hand_gizmos(client_id, frame_idx)
        elif session.hand_gizmos:
            # Hide hand gizmos if checkbox is off
            self.set_hand_gizmos_visibility(client_id, False)

        # Update target velocity arrow visualization (orange, user-specified target)
        if session.target_velocity_arrow is not None:
            gui_elements = session.gui_elements
            # Show target velocity arrow if enabled and we have a valid root position
            if gui_elements.gui_use_target_velocity_checkbox.value and root_pos_for_target is not None:
                # Get target velocity from GUI (x, z)
                target_vel_xz = gui_elements.gui_target_root_velocity.value
                target_velocity = np.array([target_vel_xz[0], 0.0, target_vel_xz[1]])

                # Update target velocity arrow
                session.target_velocity_arrow.update(
                    root_velocity=target_velocity,
                    root_pos=root_pos_for_target,
                    visible=True,
                )

                # Predict future root positions and update 2D root constraints
                if frame_idx % TARGET_VELOCITY_UPDATE_INTERVAL == 0:
                    self._update_root_constraints_from_target_velocity(client_id, frame_idx, target_velocity)
            else:
                # Hide target velocity arrow if disabled or no character available
                session.target_velocity_arrow.set_visibility(False)

        # Update camera
        if session.gui_elements.gui_viz_auto_camera_checkbox.value:
            self.update_camera_follow(client_id, frame_idx)
