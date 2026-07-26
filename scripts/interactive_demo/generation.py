# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

from .common import *  # noqa: F401,F403
from .window_budget import compute_window_num_frames


class GenerationMixin:
    def restart(self, client_id: int):
        """Restart the demo for a client."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        playing = session.playing
        session.playing = False
        self.clear_motions(client_id)
        session.max_frame_idx = -1
        session.frame_idx = 0

        # Reset camera state for smooth transitions
        session.camera_position = None
        session.camera_look_at = None
        session.camera_forward_direction = None
        session.camera_position_buffer.clear()
        session.camera_last_update_frame = -1

        # Clear all timeline prompts and add current active prompt from frame 0 to infinity
        self.clear_timeline_prompts(client_id)

        client = session.client
        if session.timeline_data is not None and hasattr(client, "timeline"):
            # Add current active prompt from frame 0 to infinity
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            current_prompt = session.gui_elements.gui_prompt_text.value
            try:
                new_uuid = client.timeline.add_prompt(
                    text=current_prompt,
                    start_frame=0,
                    end_frame=INFINITE_FRAME_IDX,
                    color=self.get_prompt_color(0),
                )
                prompt_uuid_list.append(new_uuid)
                session.timeline_data["prompt_counter"] = 1  # Reset counter
                print(f"Added prompt for restart: '{current_prompt}' (frames 0-∞)")
            except (AttributeError, Exception) as e:
                print(f"Error adding prompt: {e}")

        self._generate_step(client_id)

        session.playing = playing
        self.set_frame(client_id, 0)

    def restart_from_now(self, client_id: int):
        """Restart generation from the current frame, clearing all motions after it."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if session.model is None:
            print(f"Model not loaded for client {client_id}!")
            return

        current_frame = session.frame_idx
        if current_frame < 0:
            # No motion yet, fall back to normal restart
            self.restart(client_id)
            return

        playing = session.playing
        session.playing = False

        # Crop motion data to current frame (keep frames 0..current_frame)
        keep_end = current_frame + 1
        with session.motion_tensor_lock:
            if session.motion_tensor is not None:
                session.motion_tensor = session.motion_tensor[:, :keep_end]
            if session.joints_pos is not None:
                session.joints_pos = session.joints_pos[:, :keep_end]
            if session.joints_rot is not None:
                session.joints_rot = session.joints_rot[:, :keep_end]
            if session.foot_contacts is not None:
                session.foot_contacts = session.foot_contacts[:, :keep_end]
            if session.root_velocities is not None:
                session.root_velocities = session.root_velocities[:, :keep_end]

        session.max_frame_idx = current_frame
        session.gui_elements.gui_frame_idx_input.max = current_frame

        # Clear timeline text prompts that start after the current frame,
        # and extend the active prompt (the one covering current_frame) to infinity
        client = session.client
        if session.timeline_data is not None and hasattr(client, "timeline"):
            prompt_uuid_list = session.timeline_data.get("prompt_uuid_list", [])
            kept_uuids = []
            active_uuid = None
            for prompt_uuid in prompt_uuid_list:
                try:
                    prompt = client.timeline._prompts.get(prompt_uuid)
                    if prompt is None:
                        continue
                    if prompt.start_frame > current_frame:
                        # Prompt starts after current frame — remove it
                        client.timeline.remove_prompt(prompt_uuid)
                        print(f"[Restart From Now] Removed future prompt '{prompt.text}' (start={prompt.start_frame})")
                    else:
                        kept_uuids.append(prompt_uuid)
                        # Track the last prompt that covers current_frame
                        if prompt.start_frame <= current_frame:
                            active_uuid = prompt_uuid
                except Exception as e:
                    print(f"[Restart From Now] Error processing prompt: {e}")

            # Extend the active prompt to infinity so generation continues with it
            if active_uuid is not None:
                try:
                    client.timeline.update_prompt(active_uuid, end_frame=INFINITE_FRAME_IDX)
                    active_prompt = client.timeline._prompts.get(active_uuid)
                    if active_prompt:
                        print(f"[Restart From Now] Extended prompt '{active_prompt.text}' to infinity")
                except Exception as e:
                    print(f"[Restart From Now] Error extending prompt: {e}")

            session.timeline_data["prompt_uuid_list"] = kept_uuids
            session.timeline_data["prompt_counter"] = len(kept_uuids)

        print(f"[Restart From Now] Cleared motion after frame {current_frame}, triggering generation")

        self._generate_step(client_id)

        session.playing = playing

    def on_replan_trigger(self, client_id: int, skip_if_busy: bool = False):
        """Called when approaching end of timeline or when prompt changes.

        With skip_if_busy=True the trigger is dropped when a replan is already running, instead of
        queuing behind it. Used by the per-frame auto-replan check, which would otherwise pile up
        redundant generations (it re-fires on the next frame anyway).
        """
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if skip_if_busy:
            if not session.replan_lock.acquire(blocking=False):
                return
            try:
                self._generate_step(client_id)
            finally:
                session.replan_lock.release()
        else:
            with session.replan_lock:
                self._generate_step(client_id)

    def _get_history_motion(self, session: ClientSession):
        """Get history motion for autoregressive generation."""
        frame_idx = session.frame_idx
        replan_buffer_size = session.gui_elements.gui_replan_buffer_size.value
        history_crop_length = session.gui_elements.gui_history_crop_length.value
        motion_tensor = session.motion_tensor

        cur_motion_len = motion_tensor.shape[1] if motion_tensor is not None else 0
        history_end_idx = min(cur_motion_len - 1, frame_idx + replan_buffer_size)
        if (
            cur_motion_len >= session.num_frames_per_token
        ):  # if there are history frames, ensure history end idx is at least num_frames_per_token - 1
            history_end_idx = max(history_end_idx, session.num_frames_per_token - 1)
        history_length = min(history_end_idx + 1, history_crop_length)
        history_length = history_length // session.num_frames_per_token * session.num_frames_per_token
        history_start_idx = max(0, history_end_idx - history_length + 1)

        history_motion_tensor = None
        if motion_tensor is not None and history_start_idx <= history_end_idx:
            history_motion_tensor = motion_tensor[:, history_start_idx : history_end_idx + 1]

        return history_motion_tensor, history_start_idx, history_end_idx, history_length

    def _generate_step(self, client_id: int):
        """One autoregressive generation step."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        if session.model is None:
            print(f"Model not loaded for client {client_id}!")
            return

        start_time = time.time()

        history_motion_tensor, history_start_idx, history_end_idx, history_length = self._get_history_motion(session)
        print(
            f"Generate with frame idx {session.frame_idx}, history start: {history_start_idx}, end: {history_end_idx}, length: {history_length}"
        )

        num_samples = session.gui_elements.gui_num_samples.value
        text_feat = session.text_embedding.repeat(num_samples, 1, 1)
        text_pad_mask = torch.ones(text_feat.shape[0], text_feat.shape[1], device=self.device, dtype=torch.bool)

        motion_mask = None
        observed_motion = None

        # Check if we have timeline constraints (including waypoints)
        constraint_idx_list = [c.get_constraint_info()["frame_idx"] for c in session.constraints.values()]
        # merge all constraint indices into a single list
        all_constraint_indices = [idx for sublist in constraint_idx_list for idx in sublist]
        has_valid_timeline_constraints = (
            len(all_constraint_indices) > 0 and max(all_constraint_indices) > history_end_idx
        )

        # number of frames of the visible sequence to the model
        num_frames = compute_window_num_frames(
            history_length=history_length,
            gen_horizon_len=session.gen_horizon_len,
            num_frames_per_token=session.num_frames_per_token,
            max_window_len=session.max_window_len,
            history_start_idx=history_start_idx,
            max_constraint_idx=(max(all_constraint_indices) if has_valid_timeline_constraints else None),
            future_crop_length=session.gui_elements.gui_future_crop_length.value,
        )

        # Process timeline constraints
        if has_valid_timeline_constraints:
            motion_mask, observed_motion = self.compute_constraint_mask(
                session,
                num_samples,
                num_frames=num_frames + history_start_idx,
                history_end_idx=history_end_idx,
            )

            if motion_mask is not None and observed_motion is not None:
                motion_mask = motion_mask[:, history_start_idx:]
                observed_motion = observed_motion[:, history_start_idx:]
                motion_mask[:, :history_length] = 0.0  # disable history frames constraints
                observed_motion[:, :history_length] = 0.0

        # if motion_mask is not None and observed_motion is not None:
        #     print(f"motion mask non zero: {(motion_mask != 0.0).sum()}, observed motion non zero: {(observed_motion != 0.0).sum()}")
        #     # motion_mask_by_dim = motion_mask.any(dim=(0, 1))
        #     # print(f"Nonzero dimensions: {motion_mask_by_dim.nonzero()}")
        #     observed_motion_by_dim = observed_motion.any(dim=(0, 1))
        #     print(f"Observed motion nonzero dimensions: {observed_motion_by_dim.nonzero().squeeze()}")

        print(f"Num frames: {num_frames}")

        if history_motion_tensor is None:
            num_samples = session.gui_elements.gui_num_samples.value
            init_global_translation = (
                torch.from_numpy(session.init_global_translation)
                .to(dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(num_samples, 1)
            )
            init_first_heading_angle = (
                torch.ones(num_samples, dtype=torch.float32, device=self.device) * session.init_first_heading_angle
            )
        else:
            init_global_translation = None
            init_first_heading_angle = None

        # Generate motion
        samples = session.model.autoregressive_step(
            num_frames=num_frames,
            num_denoising_steps=session.gui_elements.gui_diffusion_steps_slider.value,
            motion_mask=motion_mask,
            observed_motion=observed_motion,
            cfg_weight=(
                session.gui_elements.gui_cfg_text_weight.value,
                session.gui_elements.gui_cfg_constraint_weight.value,
            ),
            texts=None,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            init_history_sequence=history_motion_tensor,
            init_global_translation=init_global_translation,
            init_first_heading_angle=init_first_heading_angle,
        )

        # Convert to joints
        samples_unnormalized = session.motion_rep.unnormalize(samples)
        pred_joints_output = session.motion_rep.inverse(
            samples_unnormalized,
            is_normalized=False,
        )

        joints_pos = pred_joints_output["posed_joints"]
        joints_rot = pred_joints_output["global_rot_mats"]
        foot_contacts = pred_joints_output.get("foot_contacts")

        # Apply post-processing if enabled
        if session.gui_elements.gui_enable_postprocess_checkbox.value:
            postprocess_start_time = time.time()

            # Get constraints in the generation horizon
            model_constraints = self.compute_model_constraints_lst(
                session,
                num_frames=session.gen_horizon_len + history_length + history_start_idx,
                history_end_idx=history_end_idx,
            )

            #  check if the model_constraints is not empty
            if len(model_constraints) > 0:
                # Get local rotations and root positions from pred_joints_output
                local_rot_mats = pred_joints_output["local_rot_mats"]  # [B, T, J, 3, 3]
                root_positions = pred_joints_output["root_positions"]  # [B, T, 3]

                # subtract history_end_idx from the model_constraints frame indices
                for constraint in model_constraints:
                    constraint.frame_indices = constraint.frame_indices - history_start_idx - history_length

                # Apply post-processing to generation horizon frames
                corrected_output = post_process_motion(
                    local_rot_mats[:, history_length:],
                    root_positions[:, history_length:],
                    foot_contacts[:, history_length:],
                    session.motion_rep.skeleton,
                    constraint_lst=model_constraints if model_constraints else None,
                    contact_threshold=session.gui_elements.gui_postprocess_contact_threshold.value,
                    root_margin=session.gui_elements.gui_postprocess_root_margin.value,
                )

                # calculate corrected motion_tensor, joints_pos, joints_rot, foot_contacts
                joints_pos[:, history_length:] = corrected_output["posed_joints"]
                joints_rot[:, history_length:] = corrected_output["global_rot_mats"]
                corrected_tensor_unnormalized = session.motion_rep(
                    local_joint_rots=corrected_output["local_rot_mats"],
                    root_positions=corrected_output["root_positions"],
                    to_normalize=False,
                )
                corrected_tensor_normalized = session.motion_rep.normalize(corrected_tensor_unnormalized)
                samples_unnormalized[:, history_length:] = corrected_tensor_unnormalized
                samples[:, history_length:] = corrected_tensor_normalized
                foot_contacts[:, history_length:] = corrected_tensor_unnormalized[
                    :, :, session.motion_rep.slice_dict["foot_contacts"]
                ]

            postprocess_end_time = time.time()
            print(
                f"[PostProcess] Motion correction applied in {postprocess_end_time - postprocess_start_time:.4f} seconds"
            )

        # Extract root velocities from motion representation
        joint_velocities = samples_unnormalized[:, :, session.motion_rep.slice_dict["velocities"]]
        joint_velocities = joint_velocities.reshape(
            num_samples,
            history_length + session.gen_horizon_len,
            session.motion_rep.skeleton.nbjoints,
            3,
        )
        root_velocities = joint_velocities[:, :, session.motion_rep.skeleton.root_idx, :]

        # Update motion data
        with session.motion_tensor_lock:
            if session.motion_tensor is None:
                session.motion_tensor = samples.clone()
                session.joints_pos = joints_pos.clone()
                session.joints_rot = joints_rot.clone()
                session.foot_contacts = foot_contacts.clone()
                session.root_velocities = root_velocities.clone()
                for i in range(num_samples):
                    self.add_character(client_id, session.motion_rep.skeleton, i)
            else:
                session.motion_tensor = torch.cat(
                    [
                        session.motion_tensor[:, : history_end_idx + 1],
                        samples[:, history_length:],
                    ],
                    dim=1,
                )
                session.joints_pos = torch.cat(
                    [
                        session.joints_pos[:, : history_end_idx + 1],
                        joints_pos[:, history_length:],
                    ],
                    dim=1,
                )
                session.joints_rot = torch.cat(
                    [
                        session.joints_rot[:, : history_end_idx + 1],
                        joints_rot[:, history_length:],
                    ],
                    dim=1,
                )
                session.foot_contacts = torch.cat(
                    [
                        session.foot_contacts[:, : history_end_idx + 1],
                        foot_contacts[:, history_length:],
                    ],
                    dim=1,
                )
                session.root_velocities = torch.cat(
                    [
                        session.root_velocities[:, : history_end_idx + 1],
                        root_velocities[:, history_length:],
                    ],
                    dim=1,
                )

            # Update timeline
            session.max_frame_idx = session.motion_tensor.shape[1] - 1

        # Update frame index input max value
        session.gui_elements.gui_frame_idx_input.max = session.max_frame_idx

        end_time = time.time()
        print(f"Generate step time: {end_time - start_time} seconds")
