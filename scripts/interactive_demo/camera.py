# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class CameraMixin:
    def on_arrow_key_press(self, client_id: int, arrow_keys_pressed: set):
        """Handle arrow key press events to control target velocity."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client
        gui_elements = session.gui_elements

        # Only process if target velocity is enabled
        if not gui_elements.gui_use_target_velocity_checkbox.value:
            return

        # Get current target velocity (x, z)
        current_vel = gui_elements.gui_target_root_velocity.value
        vel_x, vel_z = current_vel[0], current_vel[1]

        # Calculate current magnitude and direction
        magnitude = np.sqrt(vel_x**2 + vel_z**2)

        # Process arrow key inputs
        if "ArrowUp" in arrow_keys_pressed:
            # Increase magnitude by 0.2
            if magnitude < 1e-6:
                # If current velocity is ~0, create a forward velocity
                vel_x, vel_z = 0.0, 0.2
            else:
                # Scale up the velocity
                scale = (magnitude + 0.2) / magnitude
                vel_x *= scale
                vel_z *= scale
            # send notification
            self.client_sessions[client_id].client.add_notification(
                title="Target velocity magnitude increased by 0.2",
                body="",
                auto_close_seconds=3.0,
                color="blue",
            )

        if "ArrowDown" in arrow_keys_pressed:
            # Decrease magnitude by 0.2
            new_magnitude = max(0.0, magnitude - 0.2)
            if magnitude > 1e-6:
                if new_magnitude < 1e-6:
                    # Reduce to zero
                    vel_x, vel_z = 0.0, 0.0
                else:
                    # Scale down the velocity
                    scale = new_magnitude / magnitude
                    vel_x *= scale
                    vel_z *= scale
            # send notification
            self.client_sessions[client_id].client.add_notification(
                title="Target velocity magnitude decreased by 0.2",
                body="",
                auto_close_seconds=3.0,
                color="blue",
            )

        if "ArrowRight" in arrow_keys_pressed:
            # Rotate 30 degrees clockwise around Y axis
            angle_rad = np.radians(30)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            # Rotation matrix around Y: [cos, -sin; sin, cos] for (x, z)
            new_vel_x = cos_a * vel_x - sin_a * vel_z
            new_vel_z = sin_a * vel_x + cos_a * vel_z
            vel_x, vel_z = new_vel_x, new_vel_z
            # send notification
            self.client_sessions[client_id].client.add_notification(
                title="Target velocity direction rotated clockwise by 30 degrees",
                body="",
                auto_close_seconds=3.0,
                color="blue",
            )

        if "ArrowLeft" in arrow_keys_pressed:
            # Rotate 30 degrees counterclockwise around Y axis
            angle_rad = np.radians(-30)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            # Rotation matrix around Y: [cos, -sin; sin, cos] for (x, z)
            new_vel_x = cos_a * vel_x - sin_a * vel_z
            new_vel_z = sin_a * vel_x + cos_a * vel_z
            vel_x, vel_z = new_vel_x, new_vel_z
            # send notification
            self.client_sessions[client_id].client.add_notification(
                title="Target velocity direction rotated counterclockwise by 30 degrees",
                body="",
                auto_close_seconds=3.0,
                color="blue",
            )

        # Clamp to GUI limits
        vel_x = np.clip(vel_x, -5.0, 5.0)
        vel_z = np.clip(vel_z, -5.0, 5.0)

        # Update GUI element
        gui_elements.gui_target_root_velocity.value = (vel_x, vel_z)

        # Trigger visual update (the on_update callback will handle the arrow update)

    def update_camera_follow(self, client_id: int, frame_idx: int, use_smoothing: bool = True):
        """Update camera to follow the first character with selectable camera type.

        Args:
            client_id: Client ID
            frame_idx: Frame index to follow
            use_smoothing: If True, uses exponential smoothing for slow, cinematic camera motion.
                          If False, snaps camera directly to target position.
        """
        if not self.client_active(client_id):
            return

        session = self.client_sessions[client_id]
        client = session.client

        # Get the first character
        with session.characters_lock:
            if len(session.characters) == 0:
                return

            first_character = list(session.characters.values())[0]
            root_idx = first_character.skeleton.root_idx

        # Get character's root position at current frame
        if frame_idx < 0 or frame_idx > session.max_frame_idx:
            return

        if session.joints_pos.shape[0] == 0:
            return

        # Update last update frame
        session.camera_last_update_frame = frame_idx

        # Get root position of the first character (character_idx=0)
        root_pos_3d = session.joints_pos[0, frame_idx, root_idx].cpu().numpy()

        # Get character's facing direction from root velocity
        target_forward_direction = np.array([0.0, 0.0, 1.0])  # Default forward (+Z)
        if session.root_velocities is not None and frame_idx < session.root_velocities.shape[1]:
            root_vel = session.root_velocities[0, frame_idx].cpu().numpy()
            # Use XZ plane velocity for direction
            vel_xz = np.array([root_vel[0], root_vel[2]])
            vel_magnitude = np.linalg.norm(vel_xz)

            # Only use velocity direction if character is moving (threshold to avoid jitter when stationary)
            if vel_magnitude > 0.05:
                target_forward_direction = np.array([vel_xz[0] / vel_magnitude, 0.0, vel_xz[1] / vel_magnitude])

        # Apply smoothing to forward direction
        if session.camera_forward_direction is None or not use_smoothing:
            forward_direction = target_forward_direction.copy()
            session.camera_forward_direction = forward_direction.copy()
        else:
            alpha_direction = 0.02  # Smoothing factor for direction changes
            forward_direction = (
                alpha_direction * target_forward_direction + (1 - alpha_direction) * session.camera_forward_direction
            )
            # Normalize to ensure it remains a unit vector
            forward_direction = forward_direction / np.linalg.norm(forward_direction)
            session.camera_forward_direction = forward_direction.copy()

        # Calculate right direction (perpendicular to forward in XZ plane)
        right_direction = np.array([forward_direction[2], 0.0, -forward_direction[0]])

        # Get camera type from GUI
        camera_type = session.gui_elements.gui_viz_camera_type_dropdown.value

        # === Calculate camera position based on type ===
        if camera_type == "Over-the-shoulder":
            # Camera is positioned behind, to the side, and above the character
            camera_height = 2.8  # Higher to see full body
            back_distance = 10.0  # Farther back to see full character
            side_distance = 1.2  # To the left side

            # Calculate target camera position relative to character
            camera_offset = (
                -forward_direction * back_distance  # Behind
                + -right_direction * side_distance  # To the left
                + np.array([0.0, camera_height, 0.0])  # Above
            )
            target_camera_position = root_pos_3d + camera_offset

            # Look-at point is slightly ahead of the character at chest height
            look_ahead_distance = 1.0  # Look closer to character for better framing
            target_look_at = root_pos_3d + forward_direction * look_ahead_distance + np.array([0.0, 1.2, 0.0])

        elif camera_type == "Front-facing":
            # Camera is positioned in front of and above the character
            camera_height = 2.5  # At head height
            front_distance = 5.5  # In front of the character
            side_distance = 0.8  # Slightly to the side for dynamic view

            # Calculate target camera position (in front)
            camera_offset = (
                forward_direction * front_distance  # In front
                + right_direction * side_distance  # Slightly to the right
                + np.array([0.0, camera_height, 0.0])  # Above
            )
            target_camera_position = root_pos_3d + camera_offset

            # Look at character's chest/face
            target_look_at = root_pos_3d + np.array([0.0, 1.4, 0.0])

        else:
            # Default to over-the-shoulder
            camera_height = 2.8
            back_distance = 4.5
            side_distance = 1.2

            camera_offset = (
                -forward_direction * back_distance
                + -right_direction * side_distance
                + np.array([0.0, camera_height, 0.0])
            )
            target_camera_position = root_pos_3d + camera_offset
            look_ahead_distance = 1.0
            target_look_at = root_pos_3d + forward_direction * look_ahead_distance + np.array([0.0, 1.2, 0.0])

        # === Temporal smoothing for positions ===
        # Store root position for temporal smoothing
        root_pos_2d = np.array([root_pos_3d[0], 0.0, root_pos_3d[2]])

        if use_smoothing:
            buffer_size = 10
            session.camera_position_buffer.append(root_pos_2d.copy())
            if len(session.camera_position_buffer) > buffer_size:
                session.camera_position_buffer.pop(0)

            smoothed_root_pos_2d = np.mean(session.camera_position_buffer, axis=0)
            smoothed_root_pos_3d = np.array([smoothed_root_pos_2d[0], root_pos_3d[1], smoothed_root_pos_2d[2]])

            # Recalculate camera position with smoothed root based on camera type
            if camera_type == "Over-the-shoulder":
                camera_height = 2.8
                back_distance = 4.5
                side_distance = 1.2

                camera_offset_smoothed = (
                    -forward_direction * back_distance
                    + -right_direction * side_distance
                    + np.array([0.0, camera_height, 0.0])
                )
                target_camera_position = smoothed_root_pos_3d + camera_offset_smoothed

                look_ahead_distance = 1.0
                target_look_at = (
                    smoothed_root_pos_3d + forward_direction * look_ahead_distance + np.array([0.0, 1.2, 0.0])
                )

            elif camera_type == "Front-facing":
                camera_height = 2.8
                front_distance = 12.5
                side_distance = 0.8

                camera_offset_smoothed = (
                    forward_direction * front_distance
                    + right_direction * side_distance
                    + np.array([0.0, camera_height, 0.0])
                )
                target_camera_position = smoothed_root_pos_3d + camera_offset_smoothed
                target_look_at = smoothed_root_pos_3d + np.array([0.0, 1.4, 0.0])
        else:
            session.camera_position_buffer.clear()
            session.camera_position_buffer.append(root_pos_2d.copy())

        # === Exponential smoothing for camera motion ===
        if session.camera_position is None or not use_smoothing:
            session.camera_position = target_camera_position.copy()
            session.camera_look_at = target_look_at.copy()
        else:
            alpha_position = 0.1
            alpha_lookat = 0.05

            session.camera_position = (
                alpha_position * target_camera_position + (1 - alpha_position) * session.camera_position
            )
            session.camera_look_at = alpha_lookat * target_look_at + (1 - alpha_lookat) * session.camera_look_at

        # === Set camera to smoothed position ===
        try:
            if hasattr(client, "camera"):
                client.camera.position = tuple(session.camera_position)
                client.camera.look_at = tuple(session.camera_look_at)
        except (AttributeError, Exception) as e:
            pass  # Silent fail
