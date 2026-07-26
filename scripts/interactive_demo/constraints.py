# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403

# End-effector joint -> constraint-type key in ardy.constraints.TYPE_TO_CLASS
# (generation regroups End-Effector keyframes by these keys).
EE_JOINT_TO_TYPE = {
    "LeftHand": "left-hand",
    "RightHand": "right-hand",
    "LeftFoot": "left-foot",
    "RightFoot": "right-foot",
}


class ConstraintsMixin:
    def remove_keyframe_from_timeline(
        self,
        client_id: int,
        constraint_type: str,
        frame_idx: int,
        constraint_id: str,
        joint_name: str = None,
    ):
        """Remove a keyframe from the timeline GUI if it exists.

        Args:
            constraint_type: Type of constraint (e.g., "2D Root", "End-Effectors")
            frame_idx: Frame index where the keyframe exists
            constraint_id: ID of the keyframe to remove
            joint_name: For End-Effectors, specify the joint
        """
        if not self.client_active(client_id):
            return False

        session = self.client_sessions[client_id]
        client = session.client

        # Check if timeline is available
        if not hasattr(client, "timeline") or session.timeline_data is None:
            return False

        # Check if timeline tracks are properly initialized
        tracks_ids = session.timeline_data.get("tracks_ids", {})
        if not tracks_ids:
            # Timeline not properly initialized (probably not supported in this viser version)
            return False

        # Get the track ID for the constraint type
        track_name = constraint_type
        if constraint_type == "End-Effectors":
            if joint_name:
                track_name = joint_name.replace("Hand", " Hand").replace("Foot", " Foot")
            else:
                return False

        if track_name not in tracks_ids:
            return False

        track_id = tracks_ids[track_name]

        # Try to remove keyframe from timeline
        try:
            if hasattr(client.timeline, "remove_keyframe"):
                # Check if keyframe is in tracking dict and get its timeline UUID
                if constraint_id not in session.timeline_data["keyframes"]:
                    return False

                # Get the timeline UUID that was returned when we added this keyframe
                keyframe_data = session.timeline_data["keyframes"][constraint_id]
                timeline_uuid = keyframe_data.get("timeline_uuid")

                if not timeline_uuid:
                    return False

                # The viser timeline API takes the UUID returned by add_keyframe
                client.timeline.remove_keyframe(timeline_uuid)

                # Remove from tracking data
                with session.timeline_data["keyframe_update_lock"]:
                    del session.timeline_data["keyframes"][constraint_id]
                return True
            else:
                return False
        except Exception as e:
            return False

    def add_keyframe_to_timeline(
        self,
        client_id: int,
        constraint_type: str,
        frame_idx: int,
        constraint_id: str,
        joint_name: str = None,
    ):
        """Attempt to add a keyframe to the timeline GUI if the API supports it.

        Falls back gracefully if not supported.

        Args:
            joint_name: For End-Effectors, specify the joint (e.g., "LeftHand", "RightFoot")
        """
        if not self.client_active(client_id):
            return False

        session = self.client_sessions[client_id]
        client = session.client

        # Check if timeline is available
        if not hasattr(client, "timeline"):
            print(f"[Timeline] Warning: Client {client_id} has no timeline attribute")
            return False

        if session.timeline_data is None:
            print(f"[Timeline] Warning: Client {client_id} has no timeline_data")
            return False

        # Check if timeline tracks are properly initialized
        tracks_ids = session.timeline_data.get("tracks_ids", {})
        if not tracks_ids:
            # Timeline not properly initialized (probably not supported in this viser version)
            print(f"[Timeline] Warning: No tracks_ids found in timeline_data")
            return False

        # Get the track ID for the constraint type
        track_name = constraint_type
        if constraint_type == "End-Effectors":
            # For end-effectors, map joint name to track name
            if joint_name:
                # Convert joint names like "LeftHand" to "Left Hand"
                track_name = joint_name.replace("Hand", " Hand").replace("Foot", " Foot")
            else:
                # Can't determine which specific track without joint name
                print(f"[Timeline] Warning: End-Effectors constraint without joint_name")
                return False

        if track_name not in tracks_ids:
            print(f"[Timeline] Warning: Track '{track_name}' not found in tracks_ids: {list(tracks_ids.keys())}")
            return False

        track_id = tracks_ids[track_name]

        # Try to add keyframe to timeline (if the API supports it)
        try:
            # Check if the timeline has an add_keyframe method
            if hasattr(client.timeline, "add_keyframe"):
                # Check if this keyframe already exists in timeline_data
                if constraint_id in session.timeline_data["keyframes"]:
                    existing = session.timeline_data["keyframes"][constraint_id]
                    print(
                        f"[Timeline] Keyframe '{constraint_id}' already exists at frame {existing['frame']}, skipping"
                    )
                    return True  # Return true since the keyframe is already there

                # add_keyframe returns a UUID that we need to store for later removal
                print(
                    f"[Timeline] Attempting to add keyframe: track_id={track_id}, frame={frame_idx}, constraint_id={constraint_id}"
                )
                timeline_uuid = client.timeline.add_keyframe(track_id, frame_idx)
                print(f"[Timeline] Successfully added keyframe, received UUID: {timeline_uuid}")

                # Track it in timeline_data - store both the UUID and other metadata
                with session.timeline_data["keyframe_update_lock"]:
                    session.timeline_data["keyframes"][constraint_id] = {
                        "frame": frame_idx,
                        "track_id": track_id,
                        "timeline_uuid": timeline_uuid,  # Store the UUID returned by viser
                    }
                print(f"[Timeline] ✓ Added keyframe: track='{track_name}', frame={frame_idx}, id={constraint_id}")
                return True
            else:
                print(f"[Timeline] Warning: add_keyframe method not available on client.timeline")
                return False
        except (AttributeError, Exception) as e:
            # API not available or failed - that's okay, constraints still work
            print(f"[Timeline] ✗ Failed to add keyframe: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def add_interval_to_timeline(
        self,
        client_id: int,
        constraint_type: str,
        start_frame_idx: int,
        end_frame_idx: int,
        constraint_id: str,
    ):
        """Attempt to add an interval to the timeline GUI if the API supports it.

        Falls back gracefully if not supported.
        """
        if not self.client_active(client_id):
            return False

        session = self.client_sessions[client_id]
        client = session.client

        # Check if timeline is available
        if not hasattr(client, "timeline") or session.timeline_data is None:
            return False

        # Check if timeline tracks are properly initialized
        tracks_ids = session.timeline_data.get("tracks_ids", {})
        if not tracks_ids:
            # Timeline not properly initialized (probably not supported in this viser version)
            return False

        # Get the track ID for the constraint type
        track_name = constraint_type
        if constraint_type == "End-Effectors":
            # For end-effectors, we can't determine which specific track without more info
            # So we skip timeline GUI integration for now
            print(f"Warning: Intervals not supported for End-Effectors in timeline")
            return False

        if track_name not in tracks_ids:
            print(f"Warning: Track '{track_name}' not found in tracks_ids: {list(tracks_ids.keys())}")
            return False

        track_id = tracks_ids[track_name]
        print(
            f"Adding interval to timeline: track='{track_name}', frames={start_frame_idx}-{end_frame_idx}, id={constraint_id}"
        )

        # Try to add interval to timeline (if the API supports it)
        try:
            # Check if the timeline has an add_interval method
            if hasattr(client.timeline, "add_interval"):
                client.timeline.add_interval(track_id, start_frame_idx, end_frame_idx, constraint_id)
                # Track it in timeline_data
                with session.timeline_data["keyframe_update_lock"]:
                    session.timeline_data["intervals"][constraint_id] = {
                        "track_id": track_id,
                        "start_frame_idx": start_frame_idx,
                        "end_frame_idx": end_frame_idx,
                    }
                return True
        except (AttributeError, Exception) as e:
            # API not available or failed - that's okay, constraints still work
            print(f"Timeline GUI interval integration not available: {e}")
            return False

        return False

    def add_waypoint(self, client_id: int, x: float, z: float):
        """Add a waypoint for a client using the 2D Root constraint system."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        interval = session.gui_elements.gui_waypoint_interval.value
        target_frame = max(0, session.frame_idx) + interval
        constraint = session.constraints["2D Root"]

        # If dense mode is enabled, first add a waypoint at the current frame
        if session.gui_elements.gui_dense_root_checkbox.value:
            current_frame = session.frame_idx
            if current_frame >= 0 and current_frame <= session.max_frame_idx:
                # Get current root position from motion data
                current_root_pos = session.joints_pos[0, current_frame, 0, :].clone()
                # Zero out the y component for 2D constraint
                current_root_pos[1] = 0.0

                # Add waypoint at current frame
                # Generate a unique waypoint ID using timestamp to avoid conflicts
                import time

                current_waypoint_id = f"waypoint_{current_frame}_{int(time.time() * 1000000) % 1000000}"
                constraint.add_keyframe(
                    keyframe_id=current_waypoint_id,
                    frame_idx=current_frame,
                    root_pos=current_root_pos,
                    viz_label=True,
                    exists_ok=True,
                    update_path=False,
                )

                # Add to timeline GUI
                success = self.add_keyframe_to_timeline(client_id, "2D Root", current_frame, current_waypoint_id)
                if not success:
                    print(
                        f"[Warning] Failed to add current waypoint to timeline GUI, but constraint was added successfully"
                    )

                print(
                    f"Added current position waypoint at frame {current_frame}: ({current_root_pos[0]:.2f}, {current_root_pos[2]:.2f})"
                )

        # Add the clicked waypoint to the 2D Root constraint track
        # Generate a unique waypoint ID using timestamp to avoid conflicts
        import time

        waypoint_id = f"waypoint_{target_frame}_{int(time.time() * 1000000) % 1000000}"
        root_pos = torch.tensor([x, 0.0, z], dtype=torch.float32)

        constraint.add_keyframe(
            keyframe_id=waypoint_id,
            frame_idx=target_frame,
            root_pos=root_pos,
            viz_label=True,
            exists_ok=True,
            update_path=False,
        )

        # Try to add to timeline GUI
        success = self.add_keyframe_to_timeline(client_id, "2D Root", target_frame, waypoint_id)
        if not success:
            print(f"[Warning] Failed to add waypoint to timeline GUI, but constraint was added successfully")

        # Update the path visualization if dense_path is enabled
        if constraint.dense_path and constraint.line_segments is not None:
            try:
                constraint.update_line_segments()
            except Exception as e:
                print(f"[Warning] Failed to update line segments: {e}")

        print(f"Added target waypoint: ({x:.2f}, {z:.2f}) at frame {target_frame}")

        threading.Thread(target=self.on_replan_trigger, args=(client_id,), daemon=True).start()

    def load_root_constraints(self, client_id: int, filepath: str = "root_constraints.json"):
        """Load root constraints from a JSON file."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Clear existing 2D root constraints
            constraint = session.constraints["2D Root"]
            constraint.clear()

            data_type = data.get("type", "keyframes")  # Default to keyframes for backward compatibility

            if data_type == "dense_trajectory":
                # Load dense trajectory as interval
                trajectory = data.get("trajectory", {})
                if len(trajectory) == 0:
                    print("No trajectory data found")
                    return

                frame_indices = sorted([int(k) for k in trajectory.keys()])
                start_frame = frame_indices[0]
                end_frame = frame_indices[-1]

                # Collect all root positions
                root_positions = []
                for frame_idx in frame_indices:
                    pos_data = trajectory[str(frame_idx)]
                    root_pos = torch.tensor(pos_data, dtype=torch.float32)
                    root_positions.append(root_pos)

                root_pos_tensor = torch.stack(root_positions)

                # Add as interval
                interval_id = f"loaded_trajectory_{start_frame}_{end_frame}"
                constraint.add_interval(
                    interval_id=interval_id,
                    start_frame_idx=start_frame,
                    end_frame_idx=end_frame,
                    root_pos=root_pos_tensor,
                    add_annulus=False,
                )

                # Add to timeline GUI
                self.add_interval_to_timeline(client_id, "2D Root", start_frame, end_frame, interval_id)

                print(
                    f"Loaded dense trajectory with {len(frame_indices)} frames from {filepath} (frames {start_frame}-{end_frame})"
                )
            else:
                # Load keyframes
                keyframes = data.get("keyframes", {})
                for frame_idx, pos_data in keyframes.items():
                    frame_idx = int(frame_idx)
                    root_pos = torch.tensor(pos_data, dtype=torch.float32)
                    keyframe_id = f"loaded_waypoint_{frame_idx}"

                    constraint.add_keyframe(
                        keyframe_id=keyframe_id,
                        frame_idx=frame_idx,
                        root_pos=root_pos,
                        viz_label=True,
                        exists_ok=True,
                    )

                    # Add to timeline GUI
                    self.add_keyframe_to_timeline(client_id, "2D Root", frame_idx, keyframe_id)

                print(f"Loaded {len(keyframes)} root keyframes from {filepath}")

        except FileNotFoundError:
            print(f"File {filepath} not found")
        except Exception as e:
            print(f"Error loading root constraints: {e}")

    def save_root_constraints(self, client_id: int, filepath: str = "root_constraints.json"):
        """Save root constraints to a JSON file."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        constraint = session.constraints["2D Root"]

        # Check if dense path is enabled
        if constraint.dense_path and len(constraint.keyframes) > 0:
            # Export dense interpolated trajectory
            constraint_info = constraint.get_constraint_info()
            frame_indices = constraint_info["frame_idx"]
            root_positions = constraint_info["root_pos"]

            data = {"type": "dense_trajectory", "trajectory": {}}

            for i, frame_idx in enumerate(frame_indices):
                pos = root_positions[i]
                # root positions may be numpy arrays or tensors; normalize to a list.
                data["trajectory"][str(frame_idx)] = (
                    pos.detach().cpu().tolist() if isinstance(pos, torch.Tensor) else np.asarray(pos).tolist()
                )

            print(f"Saved dense trajectory with {len(frame_indices)} frames to {filepath}")
        else:
            # Export keyframes only
            data = {"type": "keyframes", "keyframes": {}}

            for frame_idx, root_pos in constraint.keyframes.items():
                # keyframes may hold numpy arrays (viz track) or tensors; normalize to a list.
                data["keyframes"][str(frame_idx)] = (
                    root_pos.detach().cpu().tolist()
                    if isinstance(root_pos, torch.Tensor)
                    else np.asarray(root_pos).tolist()
                )

            print(f"Saved {len(data['keyframes'])} root keyframes to {filepath}")

        # Save to file
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving root constraints: {e}")

    def add_constraint_callback(
        self,
        client_id: int,
        constraint_id: str,
        constraint_type: str,
        frame_range: tuple[int, int],
        joint_names: list[str] = None,
        verbose: bool = True,
    ):
        """Add a constraint to the session."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        # Need to have at least one motion/character to add constraints
        with session.characters_lock:
            if len(session.characters) == 0:
                print("No characters available to add constraints!")
                return

            # Get motion data from first character
            character = list(session.characters.values())[0]

        end_effector_type = None
        if constraint_type == "End-Effectors":
            if joint_names is None or len(joint_names) == 0:
                print("No EE constraints selected! Couldn't add constraint.")
                return
            # Derive the type keys before appending Hips, which has no type of its own
            end_effector_type = {EE_JOINT_TO_TYPE[name] for name in joint_names if name in EE_JOINT_TO_TYPE}
            # Always include Hips for smoothed root
            joint_names = list(set(joint_names + ["Hips"]))

        is_interval = frame_range[1] != frame_range[0]
        start_frame_idx = int(frame_range[0])
        end_frame_idx = int(frame_range[1])

        # Validate interval
        if start_frame_idx < 0 or end_frame_idx < 0:
            print("Invalid interval! Couldn't add constraint.")
            return
        if end_frame_idx < start_frame_idx:
            print("Invalid interval! Couldn't add constraint.")
            return

        # Collect constraint data
        if is_interval:
            constraint_kwargs = {
                "interval_id": constraint_id,
                "start_frame_idx": start_frame_idx,
                "end_frame_idx": end_frame_idx,
            }
        else:
            constraint_kwargs = {
                "keyframe_id": constraint_id,
                "frame_idx": start_frame_idx,
            }

        # Get joints data from current motion
        if constraint_type in ["Full-Body", "End-Effectors"]:
            if is_interval:
                joints_pos = session.joints_pos[0, start_frame_idx : end_frame_idx + 1]
                joints_rot = session.joints_rot[0, start_frame_idx : end_frame_idx + 1]
            else:
                joints_pos = session.joints_pos[0, start_frame_idx]
                joints_rot = session.joints_rot[0, start_frame_idx]

            constraint_kwargs["joints_pos"] = joints_pos
            constraint_kwargs["joints_rot"] = joints_rot
            if constraint_type == "End-Effectors":
                constraint_kwargs["joint_names"] = joint_names
                constraint_kwargs["end_effector_type"] = end_effector_type

        elif constraint_type == "2D Root":
            # Clone (slices are views into session.joints_pos) and drop the marker
            # to the ground: the 2D root viz expects y = 0, not pelvis height.
            if is_interval:
                root_pos = session.joints_pos[0, start_frame_idx : end_frame_idx + 1, 0, :].clone()
                root_pos[:, 1] = 0.0
            else:
                root_pos = session.joints_pos[0, start_frame_idx, 0, :].clone()
                root_pos[1] = 0.0
            constraint_kwargs["root_pos"] = root_pos

        # Add the constraint
        constraint = session.constraints[constraint_type]
        if is_interval:
            constraint.add_interval(**constraint_kwargs)
        else:
            constraint.add_keyframe(**constraint_kwargs)

        if verbose:
            session.client.add_notification(
                title="Constraint added",
                body="",
                auto_close_seconds=5.0,
                color="blue",
            )

    def remove_constraint_callback(
        self,
        client_id: int,
        constraint_id: str,
        constraint_type: str,
        frame_range: tuple[int, int],
        verbose: bool = True,
    ):
        """Remove a constraint from the session."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        is_interval = frame_range[1] != frame_range[0]
        start_frame_idx = int(frame_range[0])
        end_frame_idx = int(frame_range[1])

        constraint = session.constraints[constraint_type]
        if is_interval:
            constraint.remove_interval(constraint_id, start_frame_idx, end_frame_idx)
        else:
            constraint.remove_keyframe(constraint_id, start_frame_idx)

        if verbose:
            session.client.add_notification(
                title="Constraint removed",
                body="",
                auto_close_seconds=5.0,
                color="blue",
            )

    def clear_constraints(self, client_id: int):
        """Clear all constraints for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client
        with session.timeline_data["keyframe_update_lock"]:
            for constraint in list(session.constraints.values()):
                constraint.clear()
            if hasattr(client, "timeline"):
                client.timeline.clear_keyframes()
                client.timeline.clear_intervals()

        # The reference motion is the ghost of a sampled constraint sequence, so
        # it belongs to the constraints — clear it alongside them.
        if session.ref_character is not None:
            session.ref_character.clear()
            session.ref_character = None
        session.ref_joints_pos = None
        session.ref_joints_rot = None

        client.add_notification(
            title="Constraints cleared",
            body="All constraints have been removed.",
            auto_close_seconds=2.0,
            color="blue",
        )

    def clear_timeline_prompts(self, client_id: int):
        """Clear all text prompts from timeline."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        if session.timeline_data is not None and hasattr(client, "timeline"):
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            for prompt_uuid in prompt_uuid_list:
                try:
                    client.timeline.remove_prompt(prompt_uuid)
                    print(f"Removed prompt '{prompt_uuid}' from timeline")
                except (AttributeError, Exception) as e:
                    print(f"Error removing prompt: {e}")

            # Clear the prompt list
            prompt_uuid_list.clear()
