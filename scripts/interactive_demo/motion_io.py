# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403


class MotionIOMixin:
    def _get_motion_cache_path(self, file_path: str, skeleton_name: str, fps: float) -> str:
        """Return a cache file path based on motion file identity, skeleton, and fps."""
        file_stat = os.stat(file_path)
        cache_key = f"{os.path.abspath(file_path)}:{file_stat.st_size}:{file_stat.st_mtime_ns}:{skeleton_name}:{fps}"
        cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(self._motion_cache_dir, f"{basename}_{skeleton_name}_{cache_hash}.pt")

    def _get_metadata_df(self):
        """Return cached metadata DataFrame, loading it once on first access."""
        if self._metadata_df is None:
            metadata_path = os.path.join(
                REPO_ROOT,
                "datasets",
                "bones-seed",
                "metadata",
                "seed_metadata_v004.csv",
            )
            if os.path.exists(metadata_path):
                import pandas as pd

                self._metadata_df = pd.read_csv(metadata_path)
            else:
                self._metadata_df = False  # Sentinel: file doesn't exist
        return self._metadata_df if self._metadata_df is not False else None

    def _prime_bones_seed_paths(self) -> None:
        """Build per-skeleton lists of motion-file paths from the metadata CSV.

        Runs once at startup so the "Random Motion File" button does not have to walk the filesystem
        on every click.
        """
        df = self._get_metadata_df()
        if df is None:
            print("[bones-seed] Metadata CSV not found; Random Motion File button will be disabled.")
            return
        # Keep paths relative to the repo root so the populated motion-file
        # field matches the style of the default value (e.g.
        # "datasets/bones-seed/g1/csv/..."), not an absolute path.
        bones_seed_root = "datasets/bones-seed"
        for key, column in (("g1", "move_g1_path"), ("soma", "move_soma_uniform_path")):
            if column not in df.columns:
                continue
            paths = [f"{bones_seed_root}/{rel}" for rel in df[column].dropna().astype(str).tolist() if rel]
            self._bones_seed_paths_by_skeleton[key] = paths
            print(f"[bones-seed] Loaded {len(paths)} {key} motion paths")

    def _parse_motion_file(self, file_path: str, session) -> tuple:
        """Parse a BVH/CSV file into (local_rot_mats, root_trans) CPU tensors.

        Results are cached to .cache/motion/ so repeated loads are instant.
        """
        motion_rep_infer = session.motion_rep
        skeleton = motion_rep_infer.skeleton
        fps = motion_rep_infer.fps
        skeleton_name = type(skeleton).__name__

        cache_path = self._get_motion_cache_path(file_path, skeleton_name, fps)
        if os.path.exists(cache_path):
            print(f"Loading cached motion from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            return cached["local_rot_mats"], cached["root_trans"]

        print(f"Parsing motion file {file_path} (will cache result)")
        device = "cpu"  # Parse on CPU for caching; caller moves to GPU
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".bvh":
            from ardy.skeleton.bvh import parse_bvh_motion

            bvh_fps = 120
            local_rot_mats, root_trans, parsed_fps = parse_bvh_motion(file_path)
            # The demo assumes 120 fps BVH input; sanity-check the file matches
            # (frame_time is a float, so compare rounded).
            assert round(parsed_fps) == bvh_fps, f"Expected {bvh_fps} fps BVH, got {parsed_fps:.3f} fps: {file_path}"
            step = round(bvh_fps / fps)
            root_trans = root_trans[::step]
            local_rot_mats = local_rot_mats[::step]

            import ardy as _ardy_pkg

            global_offsets_path = os.path.join(
                os.path.dirname(_ardy_pkg.__file__),
                "assets",
                "skeletons",
                "somaskel77",
                "standard_t_pose_global_offsets_rots.p",
            )

            local_rot_mats = local_rot_mats.to(device=device, dtype=torch.float32)
            root_trans = root_trans.to(device=device, dtype=torch.float32)

            full_skel_parents = SOMASkeleton77.bone_order_names_with_parents
            full_bone_names = [x for x, _ in full_skel_parents]
            full_parent_idx = torch.tensor(
                [-1 if y is None else full_bone_names.index(y) for x, y in full_skel_parents],
                device=device,
            )
            full_root_idx = 0
            neutral_joints_full = torch.ones(
                (len(local_rot_mats), 77, 3),
                device=device,
                dtype=torch.float32,
            )
            _, global_rot_mats = batch_rigid_transform(
                local_rot_mats,
                neutral_joints_full,
                full_parent_idx,
                full_root_idx,
            )

            if os.path.exists(global_offsets_path):
                global_rot_offsets = torch.load(global_offsets_path, weights_only=True).squeeze().to(device)
                global_rot_mats = torch.einsum(
                    "T N m n, N o n -> T N m o",
                    global_rot_mats,
                    global_rot_offsets,
                )
                parent_rots = global_rot_mats[:, full_parent_idx]
                parent_rots[:, full_root_idx] = torch.eye(3, device=device)
                local_rot_mats = torch.einsum(
                    "T N m n, T N n o -> T N m o",
                    parent_rots.transpose(-2, -1),
                    global_rot_mats,
                )
            else:
                print(
                    f"Warning: standard t-pose offsets not found at {global_offsets_path}, skipping t-pose conversion"
                )

            model_nbjoints = skeleton.nbjoints
            if local_rot_mats.shape[1] > model_nbjoints:
                full_bone_names = [x for x, _ in SOMASkeleton77.bone_order_names_with_parents]
                model_bone_names = skeleton.bone_order_names
                skel_indices = [full_bone_names.index(name) for name in model_bone_names]
                global_rot_mats_sub = global_rot_mats[:, skel_indices]
                local_rot_mats = global_rots_to_local_rots(global_rot_mats_sub, skeleton)

        elif ext == ".csv":
            import xml.etree.ElementTree as ET

            from scipy.spatial.transform import Rotation as ScipyRotation

            converter = session.mujoco_converter
            if converter is None:
                raise ValueError("MujocoQposConverter not available. Load a G1 model first.")

            with open(file_path) as f:
                csv_lines = f.readlines()
            csv_data = np.array([np.array([float(x) for x in row.split(",")[1:]]) for row in csv_lines[1:]])
            raw_fps = 120
            step = round(raw_fps / fps)
            csv_data = csv_data[::step]
            num_frames = csv_data.shape[0]

            root_trans_raw = csv_data[:, :3]
            root_rot_euler = csv_data[:, 3:6]
            root_rot_scipy = ScipyRotation.from_euler("xyz", root_rot_euler, degrees=True)

            R_zup_to_yup = ScipyRotation.from_euler("x", -90, degrees=True)
            x_forward_to_y_forward = ScipyRotation.from_euler("z", -90, degrees=True)
            mujoco_to_ardy_scipy = R_zup_to_yup * x_forward_to_y_forward

            root_trans_np = mujoco_to_ardy_scipy.apply(root_trans_raw)
            root_rot_ardy = mujoco_to_ardy_scipy * root_rot_scipy * mujoco_to_ardy_scipy.inv()

            nb_joints = skeleton.nbjoints
            local_rot_mats_np = np.tile(np.eye(3), (num_frames, nb_joints, 1, 1))
            local_rot_mats_np[:, 0] = root_rot_ardy.as_matrix()

            tree = ET.parse(converter.xml_path)
            root_xml = tree.getroot()
            xml_classes = [x for x in tree.findall(".//default") if "class" in x.attrib]
            joint_axes_xml = {}
            for xml_class in xml_classes:
                j = xml_class.findall("joint")
                if j:
                    joint_axes_xml[xml_class.get("class")] = j[0].get("axis")
            parent_map = {child: parent for parent in root_xml.iter() for child in parent}

            for joint_id_in_csv, joint in enumerate(root_xml.find("worldbody").findall(".//joint")):
                joint_name_in_skeleton = joint.get("name").replace("_joint", "_skel")
                if joint_name_in_skeleton not in skeleton.bone_order_names:
                    continue
                axis_values = [float(x) for x in (joint.get("axis") or joint_axes_xml[joint.get("class")]).split(" ")]
                axis_in_mujoco = ["x", "y", "z"][np.argmax(axis_values)]
                joint_dof_rad = csv_data[:, joint_id_in_csv + 6] * np.pi / 180
                # SciPy's array-API from_euler reads the last axis of `angles` as the axis
                # count, so a single-axis sequence needs shape (N, 1), not a bare (N,).
                joint_rot = ScipyRotation.from_euler(axis_in_mujoco, joint_dof_rad[:, None], degrees=False)
                body = parent_map[joint]
                if "quat" in body.attrib:
                    extra_rot = ScipyRotation.from_quat(
                        [float(x) for x in body.get("quat").strip().split(" ")],
                        scalar_first=True,
                    )
                    joint_rot = extra_rot * joint_rot
                skel_idx = skeleton.bone_order_names.index(joint_name_in_skeleton)
                local_rot_mats_np[:, skel_idx] = (
                    mujoco_to_ardy_scipy * joint_rot * mujoco_to_ardy_scipy.inv()
                ).as_matrix()

            local_rot_mats = torch.tensor(local_rot_mats_np, dtype=torch.float32, device=device)
            root_trans = torch.tensor(root_trans_np * 0.01, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .bvh or .csv")

        # Save to disk cache (CPU tensors)
        torch.save({"local_rot_mats": local_rot_mats, "root_trans": root_trans}, cache_path)
        print(f"Cached motion to {cache_path}")
        return local_rot_mats, root_trans

    def load_motion_from_file(self, file_path: str, session, crop_10s: bool = False) -> dict:
        """Load a motion sequence from a BVH or CSV file and convert to motion_rep features.

        Args:
            file_path: Path to BVH (soma skeleton) or CSV (g1 skeleton) file.
            session: Client session with loaded model and motion_rep_infer.
            crop_10s: If True, randomly crop the motion to 10 seconds.

        Returns:
            dict with "motion" (normalized feature tensor [T, D]) and "text" (empty string).
        """
        motion_rep_infer = session.motion_rep
        fps = motion_rep_infer.fps
        device = (
            next(iter(motion_rep_infer.skeleton.buffers())).device
            if list(motion_rep_infer.skeleton.buffers())
            else "cpu"
        )

        # Load parsed tensors (cached on disk after first parse)
        local_rot_mats, root_trans = self._parse_motion_file(file_path, session)
        local_rot_mats = local_rot_mats.to(device=device, dtype=torch.float32)
        root_trans = root_trans.to(device=device, dtype=torch.float32)

        # Optionally crop to 10 seconds
        if crop_10s:
            max_frames = int(10.0 * fps)
            total_frames = local_rot_mats.shape[0]
            if total_frames > max_frames:
                start = np.random.randint(0, total_frames - max_frames + 1)
                local_rot_mats = local_rot_mats[start : start + max_frames]
                root_trans = root_trans[start : start + max_frames]

        # Convert to motion_rep features
        local_rot_mats = local_rot_mats.unsqueeze(0)  # [1, T, J, 3, 3]
        root_trans = root_trans.unsqueeze(0)  # [1, T, 3]

        feats = motion_rep_infer(local_rot_mats, root_trans, to_normalize=False)  # [1, T, D]
        # Canonicalize: rotate first frame heading to 0 and translate root to origin
        rotated = motion_rep_infer.rotate_to(feats, torch.tensor(0.0, device=device))
        # Translate 2D root to origin manually (avoid ensure_batched conflict)
        root_pos = rotated[:, :, motion_rep_infer.slice_dict["root_pos"]]
        first_2d = root_pos[:, 0, [0, 2]].clone()
        can_feats = motion_rep_infer.translate_2d(rotated, -first_2d)
        motion = motion_rep_infer.normalize(can_feats).squeeze(0)  # [T, D]

        # Look up text description from cached metadata
        text = ""
        meta = self._get_metadata_df()
        if meta is not None:
            import pandas as pd

            file_stem = os.path.splitext(os.path.basename(file_path))[0]
            row = meta[meta["filename"] == file_stem]
            if not row.empty:
                desc_cols = [
                    "content_natural_desc_1",
                    "content_natural_desc_2",
                    "content_natural_desc_3",
                    "content_natural_desc_4",
                ]
                descs = [str(row.iloc[0][c]) for c in desc_cols if c in row.columns and pd.notna(row.iloc[0][c])]
                if descs:
                    text = descs[np.random.randint(len(descs))]

        return {"motion": motion, "text": text}

    def load_sequence(
        self,
        client_id: int,
        seq_data,
        constraint_types: list = None,
        continue_from_current: bool = False,
        update_text: bool = True,
    ):
        """Load a sequence from the dataset and sample constraints of specified types.

        Args:
            client_id: Client ID
            seq_data: Sequence data from dataset
            constraint_types: List of constraint types to sample (default: ["Full Body"])
            continue_from_current: If True, transform sequence to continue from current position and heading
            update_text: If True, update the text prompt with the sequence text
        """
        if constraint_types is None:
            constraint_types = ["Full Body"]
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        # Require model to be loaded for visualization
        if session.motion_rep is None:
            session.client.add_notification(
                title="No model loaded",
                body="Please load a model first to visualize sequences.",
                color="red",
            )
            return

        motion = seq_data["motion"]
        motion = motion.to(self.device)
        motion_len = motion.shape[0]
        print(f"Motion length: {motion_len}")

        # Get frame offset and transform if continuing from current frame
        frame_offset = 0
        current_frame_idx = session.frame_idx
        if continue_from_current and session.motion_tensor is not None and current_frame_idx >= 0:
            frame_offset = current_frame_idx + 1  # Start from next frame

            # Get current root position and heading from motion tensor
            if current_frame_idx <= session.max_frame_idx:
                # Unnormalize the loaded motion to work in real space
                motion_unnorm = session.motion_rep.unnormalize(motion.unsqueeze(0))  # [1, T, D]

                # Get current motion state (unnormalized)
                current_motion = session.motion_tensor[0:1, current_frame_idx : current_frame_idx + 1]  # [1, 1, D]
                current_motion_unnorm = session.motion_rep.unnormalize(current_motion)

                # Extract current heading angle
                current_heading = session.motion_rep.get_root_heading_angle(current_motion_unnorm)[:, 0]  # [1]

                # Extract current 2D root position
                root_pos = current_motion_unnorm[:, :, session.motion_rep.slice_dict["root_pos"]]  # [1, 1, 3]
                current_root_2d = root_pos[:, 0, [0, 2]]  # [1, 2]

                # Transform loaded motion to match current heading and position
                # 1. Rotate to match current heading
                motion_rotated = session.motion_rep.rotate_to(motion_unnorm, current_heading)

                # 2. Translate to match current 2D position
                motion_transformed = session.motion_rep.translate_2d_to(motion_rotated, current_root_2d)

                # Normalize back
                motion = session.motion_rep.normalize(motion_transformed).squeeze(0)  # [T, D]

                print(
                    f"Continuing from frame {session.frame_idx}, offset={frame_offset}, heading={current_heading.item():.2f} rad"
                )

        # Get joints positions and rotations
        inverse_output = session.motion_rep.inverse(
            motion,
            is_normalized=True,
        )
        joints_pos = inverse_output["posed_joints"]  # [T, J, 3]
        joints_rot = inverse_output["global_rot_mats"]  # [T, J, 3, 3]

        # Clear old constraints (and their reference ghost) BEFORE storing the
        # new reference below — clear_constraints() resets ref_joints_pos, so
        # running it afterwards would wipe the reference we just set and leave a
        # frozen/absent ghost. Continuation mode keeps its constraints and reads
        # the prior reference for concatenation, so it is skipped here.
        if not continue_from_current:
            self.clear_constraints(client_id)

        # Store reference motion for visualization.
        # NOTE: keep joints_pos/joints_rot as the continuation sequence — the
        # constraint-sampling loop below indexes them by seq_idx. Build the
        # stored reference in separate tensors so concatenation doesn't shift
        # those indices.
        #
        # In continue-from-current mode the continuation starts at frame_offset,
        # while playback indexes the reference by the global timeline frame
        # (playback.py: ref_joints_pos[frame_idx]). Concatenate the continuation
        # directly after the current session's reference motion (up to
        # frame_offset) so the frames the character already followed stay intact
        # and the continuation lines up with the constraints, which are placed at
        # seq_idx + frame_offset.
        ref_joints_pos = joints_pos  # [T, J, 3]
        ref_joints_rot = joints_rot  # [T, J, 3, 3]
        if continue_from_current and frame_offset > 0 and session.joints_pos is not None:
            # Prefix with the motion the character actually followed, then append
            # the continuation. session.joints_pos is the same posed-joint
            # representation (generation.py) and is guaranteed to be at least
            # frame_offset frames long (current_frame_idx < T), so the continuation
            # lands at exactly index frame_offset — where the sampled constraints
            # are placed (seq_idx + frame_offset) — giving pixel-accurate overlap.
            # (The previous-reference ghost can be shorter than frame_offset once
            # playback advances past it, which shifted the continuation and left a
            # too-short, static reference.)
            prev_pos = session.joints_pos[0, :frame_offset].to(joints_pos)
            prev_rot = session.joints_rot[0, :frame_offset].to(joints_rot)
            ref_joints_pos = torch.cat([prev_pos, joints_pos], dim=0)
            ref_joints_rot = torch.cat([prev_rot, joints_rot], dim=0)
        session.ref_joints_pos = ref_joints_pos
        session.ref_joints_rot = ref_joints_rot

        # Create/update reference character if visualization is enabled
        if session.gui_elements.gui_viz_ref_motion_checkbox.value:
            self._create_ref_character(client_id)

        # Update text prompt
        if update_text:
            text = seq_data["text"]
            session.gui_elements.gui_prompt_text.value = text

            # Handle timeline prompts based on continuation mode
            if continue_from_current:
                # Use on_text_prompt_update to handle continuation (updates last prompt and adds new one)
                self.on_text_prompt_update(client_id, trigger_replan=False)
            else:
                # Clear all prompts and add new one for loaded sequence
                self.clear_timeline_prompts(client_id)
                self.on_text_prompt_update(client_id, trigger_replan=False, initial_prompt=True)

        # Get max keyframes from GUI
        max_keyframe_num = session.gui_elements.gui_max_keyframe_num.value

        # Determine sampling range based on continuation mode.
        # In continuation mode, don't sample constraints within the first 2 seconds
        # (2 x fps frames) so the character has room to reach them.
        min_offset = round(2 * session.motion_rep.fps) if continue_from_current else 0

        # Check if sequence is long enough for multiple keyframes in continuous mode
        if continue_from_current and motion_len < min_offset + max_keyframe_num:
            # Sequence too short - only sample last frame
            available_indices = [motion_len - 1]
            print(f"Continuous mode: sequence too short ({motion_len} frames), sampling only last frame")
        else:
            # Normal sampling or sufficient length
            if continue_from_current:
                # Sample from frames at least 40 away from start
                available_indices = list(range(min_offset, motion_len))
                print(f"Continuous mode: sampling from frames {min_offset} to {motion_len - 1}")
            else:
                # Sample from all frames
                available_indices = list(range(motion_len))

        # Sample keyframe indices (common for all constraint types except trajectory)
        def sample_keyframe_indices():
            if len(available_indices) <= max_keyframe_num:
                return available_indices
            else:
                print(f"Sampling {max_keyframe_num} keyframes from {len(available_indices)} available indices")
                num_keyframes = np.random.randint(1, max_keyframe_num + 1)
                sampled = np.random.choice(available_indices, size=num_keyframes, replace=False)
                return sorted([int(idx) for idx in sampled])

        # Sample constraints for each selected type
        total_constraints = 0
        for constraint_type in constraint_types:
            if constraint_type == "Full Body":
                keyframe_indices = sample_keyframe_indices()
                constraint = session.constraints["Full-Body"]
                for seq_idx in keyframe_indices:
                    timeline_frame_idx = seq_idx + frame_offset
                    constraint_id = f"fullbody_sampled_{timeline_frame_idx}"
                    constraint.add_keyframe(
                        keyframe_id=constraint_id,
                        frame_idx=timeline_frame_idx,
                        joints_pos=joints_pos[seq_idx],
                        joints_rot=joints_rot[seq_idx],
                    )
                    self.add_keyframe_to_timeline(client_id, "Full-Body", timeline_frame_idx, constraint_id)
                total_constraints += len(keyframe_indices)

            elif constraint_type in ["Hands", "Feet", "Hands and Feet"]:
                # Map constraint type to joint list
                joint_map = {
                    "Hands": [("LeftHand", "left-hand"), ("RightHand", "right-hand")],
                    "Feet": [("LeftFoot", "left-foot"), ("RightFoot", "right-foot")],
                    "Hands and Feet": [
                        ("LeftHand", "left-hand"),
                        ("RightHand", "right-hand"),
                        ("LeftFoot", "left-foot"),
                        ("RightFoot", "right-foot"),
                    ],
                }
                ee_joints = joint_map[constraint_type]

                keyframe_indices = sample_keyframe_indices()
                constraint = session.constraints["End-Effectors"]

                for seq_idx in keyframe_indices:
                    timeline_frame_idx = seq_idx + frame_offset
                    for joint, ee_type in ee_joints:
                        constraint_id = f"ee_{joint}_sampled_{timeline_frame_idx}"
                        joint_names = [
                            joint,
                            "Hips",
                        ]  # Always include Hips for smoothed root
                        constraint.add_keyframe(
                            keyframe_id=constraint_id,
                            frame_idx=timeline_frame_idx,
                            joints_pos=joints_pos[seq_idx],
                            joints_rot=joints_rot[seq_idx],
                            joint_names=joint_names,
                            end_effector_type=ee_type,
                        )
                        self.add_keyframe_to_timeline(
                            client_id,
                            "End-Effectors",
                            timeline_frame_idx,
                            constraint_id,
                            joint_name=joint,
                        )
                total_constraints += len(keyframe_indices) * len(ee_joints)

            elif constraint_type == "2D Root Waypoints":
                keyframe_indices = sample_keyframe_indices()
                root_constraint = session.constraints["2D Root"]
                timeline_added = 0
                for seq_idx in keyframe_indices:
                    timeline_frame_idx = seq_idx + frame_offset
                    constraint_id = f"root2d_waypoint_sampled_{timeline_frame_idx}"
                    root_pos = joints_pos[seq_idx, 0, :].clone()
                    root_pos[1] = 0.0  # Set y to 0
                    root_constraint.add_keyframe(
                        keyframe_id=constraint_id,
                        frame_idx=timeline_frame_idx,
                        root_pos=root_pos,
                        viz_label=True,
                        update_path=False,
                    )
                    if self.add_keyframe_to_timeline(client_id, "2D Root", timeline_frame_idx, constraint_id):
                        timeline_added += 1
                total_constraints += len(keyframe_indices)
                print(
                    f"[Load Sequence] Added {len(keyframe_indices)} 2D root waypoints, {timeline_added} added to timeline"
                )

            elif constraint_type == "2D Root Trajectory":
                # Sample trajectory range
                if continue_from_current and motion_len < min_offset + 1:
                    start_idx = end_idx = motion_len - 1  # Degenerate trajectory
                    print(f"Continuous mode: trajectory too short, using last frame only")
                else:
                    available_start = min_offset if continue_from_current else 0
                    max_traj_len = motion_len - available_start
                    traj_len = np.random.randint(1, max_traj_len + 1)
                    start_idx = np.random.randint(available_start, motion_len - traj_len + 1)
                    end_idx = start_idx + traj_len - 1

                timeline_start_idx, timeline_end_idx = (
                    start_idx + frame_offset,
                    end_idx + frame_offset,
                )
                constraint_id = f"root2d_trajectory_{timeline_start_idx}_{timeline_end_idx}"

                # Extract and add trajectory
                root_pos = joints_pos[start_idx : end_idx + 1, 0, :].clone()
                root_pos[:, 1] = 0.0  # Set y to 0
                session.constraints["2D Root"].add_interval(
                    interval_id=constraint_id,
                    start_frame_idx=timeline_start_idx,
                    end_frame_idx=timeline_end_idx,
                    root_pos=root_pos,
                    add_annulus=False,
                )
                self.add_interval_to_timeline(
                    client_id,
                    "2D Root",
                    timeline_start_idx,
                    timeline_end_idx,
                    constraint_id,
                )
                total_constraints += 1

        # Notify user about added constraints
        constraint_str = ", ".join(constraint_types)
        session.client.add_notification(
            title="Constraints Sampled",
            body=f"Added {total_constraints} constraints ({constraint_str}). Check 3D view for visualization.",
            auto_close_seconds=5.0,
            color="green",
        )

        # Only restart if not continuing from current frame
        if not continue_from_current:
            self.restart(client_id)
        else:
            # trigger replan
            self.on_replan_trigger(client_id)
