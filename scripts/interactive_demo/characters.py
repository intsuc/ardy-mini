# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class CharactersMixin:
    def add_character(self, client_id: int, skeleton: SkeletonBase, index: int):
        """Add a character to a client session."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        character_name = f"character{index}"
        # Use session's mesh_mode (set during model loading)
        mesh_mode = getattr(session, "mesh_mode", "core_skin")
        new_character = Character(
            character_name,
            session.client,
            skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=True,
            visible_skeleton=session.gui_elements.gui_viz_skeleton_checkbox.value,
            visible_skinned_mesh=session.gui_elements.gui_viz_skinned_mesh_checkbox.value,
            skinned_mesh_opacity=session.gui_elements.gui_viz_skinned_mesh_opacity_slider.value,
            show_foot_contacts=session.gui_elements.gui_viz_foot_contacts_checkbox.value,
            dark_mode=session.gui_elements.gui_dark_mode_checkbox.value,
            show_root_2d_projection=True,
            mesh_mode=mesh_mode,
        )
        with session.characters_lock:
            session.characters[character_name] = new_character

        # Create target velocity arrow mesh (orange color) only once per session
        # This should be created only for the first character (index == 0)
        if index == 0:
            if session.target_velocity_arrow is None:
                # Clean up any existing scene elements with this name (safety check)
                arrow_name = f"target_velocity_{client_id}"
                try:
                    self.server.scene.remove_by_name(f"{arrow_name}/velocity_line")
                except:
                    pass
                try:
                    self.server.scene.remove_by_name(f"{arrow_name}/velocity_cone")
                except:
                    pass

                session.target_velocity_arrow = VelocityArrowMesh(
                    name=arrow_name,
                    server=self.server,
                    skeleton=skeleton,
                    color=(255, 100, 0),  # Orange color for target velocity
                )
                print(f"[Client {client_id}] Created target velocity arrow")
            else:
                print(f"[Client {client_id}] Target velocity arrow already exists, skipping creation")

    def clear_motions(self, client_id: int):
        """Clear all motions for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        with session.motion_tensor_lock:
            session.motion_tensor = None
            session.joints_pos = None
            session.joints_rot = None
            session.foot_contacts = None
            session.root_velocities = None

        with session.characters_lock:
            for character in session.characters.values():
                character.clear()
            session.characters.clear()

        # Clear target velocity arrow
        if session.target_velocity_arrow is not None:
            print(f"[Client {client_id}] Clearing target velocity arrow")
            session.target_velocity_arrow.clear()
            session.target_velocity_arrow = None

        # Clear hand gizmos
        self.clear_hand_gizmos(client_id)

    def _create_ref_character(self, client_id: int):
        """Create a red reference motion character for the loaded motion."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if session.ref_joints_pos is None or session.motion_rep is None:
            return

        skeleton = session.motion_rep.skeleton
        mesh_mode = getattr(session, "mesh_mode", "core_skin")

        # Clear existing ref character
        if session.ref_character is not None:
            session.ref_character.clear()
            session.ref_character = None

        ref_char = Character(
            "ref_motion",
            session.client,
            skeleton,
            create_skeleton_mesh=False,
            create_skinned_mesh=True,
            visible_skinned_mesh=True,
            skinned_mesh_opacity=0.6,
            show_foot_contacts=False,
            dark_mode=False,
            mesh_mode=mesh_mode,
        )

        # Override mesh color to red
        if ref_char.skinned_mesh is not None:
            ref_char.skinned_mesh.color = (200, 60, 60)
        if ref_char.g1_mesh_rig is not None:
            ref_char.g1_mesh_rig.set_color((200, 60, 60))

        # Set initial pose to the current frame (clamped to the reference range,
        # matching playback's indexing) so enabling the ghost mid-playback shows
        # it aligned with the current frame instead of snapping back to frame 0.
        ref_frame = min(max(session.frame_idx, 0), session.ref_joints_pos.shape[0] - 1)
        ref_char.set_pose(
            session.ref_joints_pos[ref_frame],
            session.ref_joints_rot[ref_frame],
        )
        session.ref_character = ref_char

    def create_hand_gizmos(self, client_id: int):
        """Create transform gizmos for hand/wrist joints."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        # Clear existing gizmos first
        self.clear_hand_gizmos(client_id)

        # Create gizmos for each character
        with session.characters_lock:
            for character_name, character in session.characters.items():
                skeleton = character.skeleton

                # Find hand and foot joint indices using the first joint of each chain
                hand_joints = {}
                for joint_names in [
                    getattr(skeleton, "left_hand_joint_names", []),
                    getattr(skeleton, "right_hand_joint_names", []),
                    getattr(skeleton, "left_foot_joint_names", []),
                    getattr(skeleton, "right_foot_joint_names", []),
                ]:
                    if joint_names and joint_names[0] in skeleton.bone_order_names:
                        joint_name = joint_names[0]
                        joint_idx = skeleton.bone_order_names_index[joint_name]
                        hand_joints[joint_name] = joint_idx

                if not hand_joints:
                    continue

                # Create batched axes for hand orientations (initially invisible)
                # We'll create one batched axes object per character with all hand joints
                if hand_joints:
                    # Get initial positions and rotations (use frame 0 or current frame)
                    frame_idx = max(0, session.frame_idx) if session.frame_idx >= 0 else 0
                    if (
                        session.joints_pos is not None
                        and session.joints_rot is not None
                        and frame_idx < session.joints_pos.shape[1]
                    ):
                        joint_indices = list(hand_joints.values())
                        joint_positions = session.joints_pos[0, frame_idx, joint_indices].cpu().numpy()
                        joint_rotations = session.joints_rot[0, frame_idx, joint_indices].cpu().numpy()

                        # Convert rotation matrices to quaternions (wxyz format for viser)
                        import viser.transforms as tf

                        wxyzs = tf.SO3.from_matrix(joint_rotations).wxyz

                        # Create batched axes for all hand joints
                        hand_axes = session.client.scene.add_batched_axes(
                            f"/hand_axes_{character_name}",
                            batched_wxyzs=wxyzs,
                            batched_positions=joint_positions,
                            axes_length=0.1,
                            axes_radius=0.005,
                        )
                        session.hand_gizmos[character_name] = {
                            "axes": hand_axes,
                            "joint_indices": joint_indices,
                            "joint_names": list(hand_joints.keys()),
                        }
                        print(
                            f"[Client {client_id}] Created hand orientation axes for {character_name} with {len(joint_indices)} joints"
                        )
                    else:
                        print(
                            f"[Client {client_id}] Warning: No motion data available to create hand axes for {character_name}"
                        )

    def update_hand_gizmos(self, client_id: int, frame_idx: int):
        """Update hand gizmo positions and orientations based on current frame."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if not session.hand_gizmos or session.joints_pos is None or session.joints_rot is None:
            return

        with session.characters_lock:
            for character_name, gizmo_data in session.hand_gizmos.items():
                if character_name not in session.characters:
                    continue

                # Get joint positions and rotations for this frame
                # joints_pos: [batch, T, num_joints, 3]
                # joints_rot: [batch, T, num_joints, 3, 3]
                if frame_idx >= session.joints_pos.shape[1]:
                    continue

                # Get the batched axes object and joint indices
                hand_axes = gizmo_data["axes"]
                joint_indices = gizmo_data["joint_indices"]

                # Get positions and rotations for all hand joints
                joint_positions = session.joints_pos[0, frame_idx, joint_indices].cpu().numpy()
                joint_rotations = session.joints_rot[0, frame_idx, joint_indices].cpu().numpy()

                # Convert rotation matrices to quaternions (wxyz format for viser)
                import viser.transforms as tf

                wxyzs = tf.SO3.from_matrix(joint_rotations).wxyz

                # Update batched axes
                hand_axes.batched_positions = joint_positions
                hand_axes.batched_wxyzs = wxyzs
                hand_axes.visible = True

    def set_hand_gizmos_visibility(self, client_id: int, visible: bool):
        """Set visibility of all hand gizmos for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        for character_name, gizmo_data in session.hand_gizmos.items():
            if "axes" in gizmo_data:
                gizmo_data["axes"].visible = visible

    def clear_hand_gizmos(self, client_id: int):
        """Clear all hand gizmos for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        for character_name, gizmo_data in session.hand_gizmos.items():
            if "axes" in gizmo_data:
                session.client.scene.remove_by_name(gizmo_data["axes"].name)
                print(f"[Client {client_id}] Removed hand orientation axes for {character_name}")

        session.hand_gizmos.clear()
