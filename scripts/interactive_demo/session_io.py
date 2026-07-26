# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class SessionIOMixin:
    def export_session(self, client_id: int, filepath: str):
        """Export generated motion, text prompts, and constraints to a file using pickle."""
        if not self.client_active(client_id):
            return False
        session = self.client_sessions[client_id]

        try:
            # Prepare export data
            export_data = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "model_fps": session.model_fps,
                "max_frame_idx": session.max_frame_idx,
            }

            # Record the skeleton identity so loads can validate compatibility.
            _skel = getattr(getattr(session, "motion_rep", None), "skeleton", None)
            if _skel is not None:
                export_data["skeleton"] = {
                    "name": getattr(_skel, "name", None),
                    "nbjoints": getattr(_skel, "nbjoints", None),
                }

            # Export motion data (keep as numpy arrays)
            if session.joints_pos is not None:
                motion_data = {
                    "joints_pos": session.joints_pos.cpu().numpy(),
                    "joints_rot": session.joints_rot.cpu().numpy() if session.joints_rot is not None else None,
                    "root_velocities": session.root_velocities.cpu().numpy()
                    if session.root_velocities is not None
                    else None,
                    "motion_tensor": session.motion_tensor.cpu().numpy() if session.motion_tensor is not None else None,
                    "foot_contacts": session.foot_contacts.cpu().numpy() if session.foot_contacts is not None else None,
                }

                # Inverse motion_tensor to get local_rot_mats and root_positions
                if session.motion_tensor is not None and session.motion_rep is not None:
                    try:
                        tensor_unnorm = session.motion_rep.unnormalize(session.motion_tensor)
                        inverse_output = session.motion_rep.inverse(tensor_unnorm, is_normalized=False)
                        motion_data["local_rot_mats"] = inverse_output["local_rot_mats"].cpu().numpy()
                        motion_data["root_positions"] = inverse_output["root_positions"].cpu().numpy()
                        print("[Export] Added local_rot_mats and root_positions")
                    except Exception as e:
                        print(f"[Export] Could not compute local_rot_mats/root_positions: {e}")

                export_data["motion"] = motion_data

            # Export text prompts from timeline
            client = self.server.get_clients()[client_id]
            if hasattr(client, "timeline") and client.timeline._prompts:
                prompts_list = []
                for prompt_uuid, prompt in client.timeline._prompts.items():
                    prompts_list.append(
                        {
                            "uuid": prompt.uuid,
                            "text": prompt.text,
                            "start_frame": prompt.start_frame,
                            "end_frame": prompt.end_frame,
                            "color": prompt.color,
                        }
                    )
                export_data["prompts"] = prompts_list
                print(f"[Export] Exporting {len(prompts_list)} text prompts from timeline")
            else:
                # Fallback to single prompt from GUI if timeline is not available
                if session.gui_elements:
                    export_data["prompts"] = [
                        {
                            "text": session.gui_elements.gui_prompt_text.value,
                            "start_frame": 0,
                            "end_frame": session.max_frame_idx,
                            "color": None,
                        }
                    ]

            # Export constraints
            constraints_data = {}

            # Export 2D Root constraints
            root_constraint = session.constraints.get("2D Root")
            if root_constraint and len(root_constraint.keyframes) > 0:
                root_keyframes = {}
                for frame_idx, root_pos in root_constraint.keyframes.items():
                    # Handle both numpy arrays and torch tensors
                    if isinstance(root_pos, torch.Tensor):
                        root_pos_np = root_pos.cpu().numpy()
                    else:
                        root_pos_np = root_pos

                    root_data = {
                        "position": root_pos_np,
                    }
                    # Add heading if it exists
                    if frame_idx in root_constraint.root_headings:
                        root_data["heading"] = root_constraint.root_headings[frame_idx]
                    root_keyframes[frame_idx] = root_data  # Use int key instead of str

                constraints_data["2D Root"] = {
                    "keyframes": root_keyframes,
                    "dense_path": root_constraint.dense_path,
                    "smooth_path": root_constraint.smooth_path,
                }

            # Export Full-Body constraints
            fb_constraint = session.constraints.get("Full-Body")
            if fb_constraint and len(fb_constraint.keyframes) > 0:
                fb_keyframes = {}
                for frame_idx, keyframe_data in fb_constraint.keyframes.items():
                    # Handle both numpy arrays and torch tensors
                    joints_pos = keyframe_data["joints_pos"]
                    joints_rot = keyframe_data["joints_rot"]
                    if isinstance(joints_pos, torch.Tensor):
                        joints_pos = joints_pos.cpu().numpy()
                    if isinstance(joints_rot, torch.Tensor):
                        joints_rot = joints_rot.cpu().numpy()

                    fb_keyframes[frame_idx] = {  # Use int key instead of str
                        "joints_pos": joints_pos,
                        "joints_rot": joints_rot,
                    }

                constraints_data["Full-Body"] = {
                    "keyframes": fb_keyframes,
                }

            # Export End-Effector constraints
            ee_constraint = session.constraints.get("End-Effectors")
            if ee_constraint and len(ee_constraint.keyframes) > 0:
                ee_keyframes = {}
                for frame_idx, keyframe_data in ee_constraint.keyframes.items():
                    # Handle both numpy arrays and torch tensors
                    joints_pos = keyframe_data["joints_pos"]
                    joints_rot = keyframe_data["joints_rot"]
                    if isinstance(joints_pos, torch.Tensor):
                        joints_pos = joints_pos.cpu().numpy()
                    if isinstance(joints_rot, torch.Tensor):
                        joints_rot = joints_rot.cpu().numpy()

                    ee_keyframes[frame_idx] = {  # Use int key instead of str
                        "joints_pos": joints_pos,
                        "joints_rot": joints_rot,
                        "joint_names": keyframe_data["joint_names"],
                        "end_effector_type": keyframe_data["end_effector_type"],
                    }

                constraints_data["End-Effectors"] = {
                    "keyframes": ee_keyframes,
                }

            export_data["constraints"] = constraints_data

            # Save to file using pickle
            os.makedirs(
                os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
                exist_ok=True,
            )
            with open(filepath, "wb") as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[Export] Saved session to {filepath}")
            print(f"[Export] Motion frames: {session.max_frame_idx + 1}")
            print(f"[Export] Root constraints: {len(root_keyframes) if 'root_keyframes' in locals() else 0}")
            print(f"[Export] Full-Body constraints: {len(fb_keyframes) if 'fb_keyframes' in locals() else 0}")
            print(f"[Export] EE constraints: {len(ee_keyframes) if 'ee_keyframes' in locals() else 0}")

            return True
        except Exception as e:
            print(f"[Export] Error exporting session: {e}")
            import traceback

            traceback.print_exc()
            return False

    def load_session(self, client_id: int, filepath: str):
        """Load generated motion, text prompts, and constraints from a pickle file."""
        if not self.client_active(client_id):
            return False
        session = self.client_sessions[client_id]
        client = session.client

        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"[Load] File not found: {filepath}")
                return False

            # Load data from pickle file
            with open(filepath, "rb") as f:
                import_data = pickle.load(f)

            print(f"[Load] Loading session from {filepath}")
            print(f"[Load] Version: {import_data.get('version', 'unknown')}")
            print(f"[Load] Timestamp: {import_data.get('timestamp', 'unknown')}")

            # Load motion data (directly from numpy arrays)
            if "motion" in import_data:
                motion_data = import_data["motion"]

                # Abort (before mutating any session state) if the session's
                # skeleton differs from the currently loaded model's skeleton.
                # Joint count is a reliable discriminator across supported
                # skeletons (27/30/34/77).
                cur_skel = getattr(getattr(session, "motion_rep", None), "skeleton", None)
                if cur_skel is not None:
                    cur_nj = getattr(cur_skel, "nbjoints", None)
                    saved = import_data.get("skeleton") or {}
                    loaded_nj = saved.get("nbjoints")
                    if loaded_nj is None and motion_data.get("joints_pos") is not None:
                        loaded_nj = motion_data["joints_pos"].shape[-2]
                    if cur_nj is not None and loaded_nj is not None and loaded_nj != cur_nj:
                        msg = (
                            f"Session skeleton ({saved.get('name') or '?'}, "
                            f"{loaded_nj} joints) does not match the current model "
                            f"skeleton ({getattr(cur_skel, 'name', '?')}, {cur_nj} "
                            "joints). Load aborted — load the matching model first."
                        )
                        print(f"[Load] {msg}")
                        try:
                            session.client.add_notification(
                                title="Skeleton mismatch",
                                body=msg,
                                color="red",
                                auto_close_seconds=6.0,
                            )
                        except Exception:
                            pass
                        return False

                session.joints_pos = torch.from_numpy(motion_data["joints_pos"]).to(
                    dtype=torch.float32, device=self.device
                )
                if motion_data["joints_rot"] is not None:
                    session.joints_rot = torch.from_numpy(motion_data["joints_rot"]).to(
                        dtype=torch.float32, device=self.device
                    )
                if motion_data["root_velocities"] is not None:
                    session.root_velocities = torch.from_numpy(motion_data["root_velocities"]).to(
                        dtype=torch.float32, device=self.device
                    )
                if motion_data.get("motion_tensor") is not None:
                    session.motion_tensor = torch.from_numpy(motion_data["motion_tensor"]).to(
                        dtype=torch.float32, device=self.device
                    )

                # Restore foot_contacts; reset to None when the session has none
                # so a stale, shorter buffer from a previous generation does not
                # cap playback (set_frame indexes foot_contacts[:, frame_idx]).
                fc = motion_data.get("foot_contacts")
                session.foot_contacts = (
                    torch.from_numpy(fc).to(dtype=torch.float32, device=self.device) if fc is not None else None
                )

                session.max_frame_idx = import_data["max_frame_idx"]
                print(f"[Load] Loaded motion with {session.max_frame_idx + 1} frames")

                # Update frame index input max value
                session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

                # Disable auto-replan when loading a session
                session.gui_elements.gui_enable_auto_replan_checkbox.value = False

            # Clear existing constraints
            for constraint in session.constraints.values():
                constraint.clear()

            # Load constraints
            if "constraints" in import_data:
                constraints_data = import_data["constraints"]

                # Load 2D Root constraints
                if "2D Root" in constraints_data:
                    root_data = constraints_data["2D Root"]
                    root_constraint = session.constraints["2D Root"]

                    # Sort keyframes by frame index to detect consecutive sequences
                    sorted_frames = sorted(root_data["keyframes"].keys())

                    # Group consecutive frames into intervals
                    intervals = []
                    isolated_frames = []

                    i = 0
                    while i < len(sorted_frames):
                        start_idx = sorted_frames[i]
                        end_idx = start_idx

                        # Find consecutive sequence
                        while i + 1 < len(sorted_frames) and sorted_frames[i + 1] == sorted_frames[i] + 1:
                            i += 1
                            end_idx = sorted_frames[i]

                        # If we have at least 2 consecutive frames, treat as interval
                        if end_idx - start_idx >= 1:
                            intervals.append((start_idx, end_idx))
                        else:
                            isolated_frames.append(start_idx)

                        i += 1

                    print(
                        f"[Load] Detected {len(intervals)} intervals and {len(isolated_frames)} isolated root keyframes"
                    )

                    # Add intervals
                    for start_idx, end_idx in intervals:
                        # Collect positions for this interval
                        num_frames = end_idx - start_idx + 1
                        root_positions = []

                        for frame_idx in range(start_idx, end_idx + 1):
                            keyframe_data = root_data["keyframes"][frame_idx]
                            root_pos = torch.from_numpy(keyframe_data["position"]).to(
                                dtype=torch.float32, device=self.device
                            )
                            root_positions.append(root_pos)

                        root_positions_tensor = torch.stack(root_positions)

                        root_constraint.add_interval(
                            interval_id=f"loaded_interval_{start_idx}_{end_idx}",
                            start_frame_idx=start_idx,
                            end_frame_idx=end_idx,
                            root_pos=root_positions_tensor,
                            add_annulus=False,
                        )
                        print(f"[Load] Added interval: frames {start_idx}-{end_idx} ({num_frames} frames)")

                    # Add isolated keyframes
                    for frame_idx in isolated_frames:
                        keyframe_data = root_data["keyframes"][frame_idx]
                        root_pos = torch.from_numpy(keyframe_data["position"]).to(
                            dtype=torch.float32, device=self.device
                        )
                        heading = keyframe_data.get("heading")

                        root_constraint.add_keyframe(
                            keyframe_id=f"loaded_{frame_idx}",
                            frame_idx=frame_idx,
                            root_pos=root_pos,
                            global_root_heading=heading,
                            exists_ok=True,
                            update_path=False,  # Don't update path until all keyframes are loaded
                        )

                    # Set path properties after all keyframes are loaded
                    # Set smooth_path first (doesn't create line segments)
                    smooth_path_value = root_data.get("smooth_path", True)
                    root_constraint.smooth_path = smooth_path_value
                    print(f"[Load] Root constraints: smooth_path={smooth_path_value}")

                    # Set dense_path last, as it creates line_segments and updates visualization
                    dense_path_value = root_data.get("dense_path", False)
                    print(f"[Load] Root constraints: dense_path={dense_path_value}")
                    if dense_path_value:
                        # This will create line_segments and call update_line_segments()
                        root_constraint.set_dense_path(True)
                        if root_constraint.line_segments is not None:
                            print(
                                f"[Load] Created dense path visualization with {len(root_constraint.keyframes)} keyframes, "
                                f"{root_constraint.line_segments.points.shape[0]} line segments"
                            )
                        else:
                            print(f"[Load] Warning: line_segments is None after set_dense_path(True)")

                    print(f"[Load] Loaded {len(root_data['keyframes'])} root constraints")

                # Load Full-Body constraints
                if "Full-Body" in constraints_data:
                    fb_data = constraints_data["Full-Body"]
                    fb_constraint = session.constraints["Full-Body"]

                    for frame_idx, keyframe_data in fb_data["keyframes"].items():
                        # frame_idx is already an int from pickle
                        joints_pos = torch.from_numpy(keyframe_data["joints_pos"]).to(
                            dtype=torch.float32, device=self.device
                        )
                        joints_rot = torch.from_numpy(keyframe_data["joints_rot"]).to(
                            dtype=torch.float32, device=self.device
                        )

                        fb_constraint.add_keyframe(
                            keyframe_id=f"loaded_{frame_idx}",
                            frame_idx=frame_idx,
                            joints_pos=joints_pos,
                            joints_rot=joints_rot,
                            viz_label=True,
                        )

                    print(f"[Load] Loaded {len(fb_data['keyframes'])} full-body constraints")

                # Load End-Effector constraints
                if "End-Effectors" in constraints_data:
                    ee_data = constraints_data["End-Effectors"]
                    ee_constraint = session.constraints["End-Effectors"]

                    for frame_idx, keyframe_data in ee_data["keyframes"].items():
                        # frame_idx is already an int from pickle
                        joints_pos = torch.from_numpy(keyframe_data["joints_pos"]).to(
                            dtype=torch.float32, device=self.device
                        )
                        joints_rot = torch.from_numpy(keyframe_data["joints_rot"]).to(
                            dtype=torch.float32, device=self.device
                        )
                        joint_names = keyframe_data["joint_names"]
                        end_effector_type = keyframe_data["end_effector_type"]

                        ee_constraint.add_keyframe(
                            keyframe_id=f"loaded_{frame_idx}",
                            frame_idx=frame_idx,
                            joints_pos=joints_pos,
                            joints_rot=joints_rot,
                            joint_names=joint_names,
                            end_effector_type=end_effector_type,
                            viz_label=True,
                        )

                    print(f"[Load] Loaded {len(ee_data['keyframes'])} end-effector constraints")

            # Update timeline with loaded prompts and constraints
            if session.timeline_data is not None and hasattr(client, "timeline"):
                # Clear existing timeline prompts
                self.clear_timeline_prompts(client_id)

                # Add loaded prompts to timeline
                if "prompts" in import_data:
                    prompt_uuid_list = []
                    for prompt_data in import_data["prompts"]:
                        try:
                            # check if start_frame is not larger than the end_frame
                            if prompt_data.get("start_frame", 0) > prompt_data.get("end_frame", INFINITE_FRAME_IDX):
                                print(
                                    f"[Load] Warning: Start frame is larger than end frame for prompt: '{prompt_data['text']}'"
                                )
                                continue
                            prompt_uuid = client.timeline.add_prompt(
                                text=prompt_data["text"],
                                start_frame=prompt_data.get("start_frame", 0),
                                end_frame=prompt_data.get("end_frame", INFINITE_FRAME_IDX),
                                color=prompt_data.get("color", None),
                            )
                            prompt_uuid_list.append(prompt_uuid)
                            print(
                                f"[Load] Added prompt to timeline: '{prompt_data['text']}' (frames {prompt_data.get('start_frame', 0)}-{prompt_data.get('end_frame', INFINITE_FRAME_IDX)})"
                            )
                        except Exception as e:
                            print(f"[Load] Warning: Failed to add prompt to timeline: {e}")
                    session.timeline_data["prompt_uuid_list"] = prompt_uuid_list
                    session.timeline_data["prompt_counter"] = len(prompt_uuid_list)
                    print(f"[Load] Loaded {len(prompt_uuid_list)} prompts to timeline")
                # Fallback to old single prompt format
                elif "prompt" in import_data:
                    try:
                        prompt_uuid = client.timeline.add_prompt(
                            text=import_data["prompt"]["text"],
                            start_frame=0,
                            end_frame=INFINITE_FRAME_IDX,
                            color=self.get_prompt_color(0),
                        )
                        session.timeline_data["prompt_uuid_list"].append(prompt_uuid)
                        print(f"[Load] Added prompt to timeline: '{import_data['prompt']['text']}'")
                    except Exception as e:
                        print(f"[Load] Warning: Failed to add prompt to timeline: {e}")

                # Add constraint keyframes to timeline
                if "constraints" in import_data:
                    constraints_data = import_data["constraints"]

                    # Add 2D Root keyframes/intervals to timeline
                    if "2D Root" in constraints_data:
                        root_data = constraints_data["2D Root"]

                        # Re-detect intervals and isolated frames for timeline
                        sorted_frames = sorted(root_data["keyframes"].keys())
                        timeline_intervals = []
                        timeline_isolated = []

                        i = 0
                        while i < len(sorted_frames):
                            start_idx = sorted_frames[i]
                            end_idx = start_idx

                            while i + 1 < len(sorted_frames) and sorted_frames[i + 1] == sorted_frames[i] + 1:
                                i += 1
                                end_idx = sorted_frames[i]

                            if end_idx - start_idx >= 1:
                                timeline_intervals.append((start_idx, end_idx))
                            else:
                                timeline_isolated.append(start_idx)
                            i += 1

                        # Add intervals to timeline
                        for start_idx, end_idx in timeline_intervals:
                            self.add_interval_to_timeline(
                                client_id=client_id,
                                constraint_type="2D Root",
                                start_frame_idx=start_idx,
                                end_frame_idx=end_idx,
                                constraint_id=f"loaded_interval_{start_idx}_{end_idx}",
                            )

                        # Add isolated keyframes to timeline
                        for frame_idx in timeline_isolated:
                            self.add_keyframe_to_timeline(
                                client_id=client_id,
                                constraint_type="2D Root",
                                frame_idx=frame_idx,
                                constraint_id=f"loaded_{frame_idx}",
                            )

                        print(
                            f"[Load] Added {len(timeline_intervals)} root intervals and {len(timeline_isolated)} isolated keyframes to timeline"
                        )

                    # Add Full-Body keyframes to timeline
                    if "Full-Body" in constraints_data:
                        fb_data = constraints_data["Full-Body"]
                        for frame_idx in fb_data["keyframes"].keys():
                            self.add_keyframe_to_timeline(
                                client_id=client_id,
                                constraint_type="Full-Body",
                                frame_idx=frame_idx,
                                constraint_id=f"loaded_{frame_idx}",
                            )
                        print(f"[Load] Added {len(fb_data['keyframes'])} full-body keyframes to timeline")

                    # Add End-Effector keyframes to timeline
                    if "End-Effectors" in constraints_data:
                        ee_data = constraints_data["End-Effectors"]
                        for frame_idx, keyframe_data in ee_data["keyframes"].items():
                            # Add keyframe for each joint (skip Hips as it doesn't have a timeline track)
                            joint_names = keyframe_data["joint_names"]
                            for joint_name in joint_names:
                                if joint_name == "Hips":
                                    continue  # Hips is used for smoothed root but doesn't have a timeline track
                                self.add_keyframe_to_timeline(
                                    client_id=client_id,
                                    constraint_type="End-Effectors",
                                    frame_idx=frame_idx,
                                    constraint_id=f"loaded_{frame_idx}_{joint_name}",
                                    joint_name=joint_name,
                                )
                        print(f"[Load] Added {len(ee_data['keyframes'])} end-effector keyframes to timeline")

            # Update display
            self.set_frame(client_id, 0)

            # Send notification
            client.add_notification(
                title="Session Loaded",
                body=f"Loaded {session.max_frame_idx + 1} frames from {os.path.basename(filepath)}",
                auto_close_seconds=3.0,
            )

            return True
        except Exception as e:
            print(f"[Load] Error loading session: {e}")
            import traceback

            traceback.print_exc()
            client.add_notification(
                title="Load Failed",
                body=f"Error: {str(e)}",
                auto_close_seconds=5.0,
            )
            return False

    def load_mesh(self, client_id: int, filepath: str, transform_type: str):
        """Load a 3D mesh file (.ply or .obj) and apply transformation."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        import trimesh

        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return

            # Load the mesh using trimesh
            mesh = trimesh.load(filepath)

            # Apply transformation if needed
            if transform_type == "Z-up to Y-up":
                # Rotation matrix to convert Z-up to Y-up
                # This rotates -90 degrees around X-axis
                rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                mesh.apply_transform(rotation_matrix)
                print("Applied Z-up to Y-up transformation")
            else:
                print("No transformation applied")

            # Add mesh to the scene
            mesh_name = f"/loaded_mesh_{client_id}"
            mesh_handle = client.scene.add_mesh_trimesh(
                name=mesh_name,
                mesh=mesh,
            )

            # Store the mesh handle and reset translation sliders
            session.loaded_scene_mesh_handle = mesh_handle
            session.gui_elements.gui_scene_translation_x.value = 0.0
            session.gui_elements.gui_scene_translation_y.value = 0.0
            session.gui_elements.gui_scene_translation_z.value = 0.0

            print(f"Loaded mesh from {filepath} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

        except ImportError:
            print("trimesh library not installed. Install with: pip install trimesh")
        except Exception as e:
            print(f"Error loading mesh: {e}")
