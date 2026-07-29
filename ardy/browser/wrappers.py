# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Small, explicit ONNX entry points used by the browser model pack.

The regular ARDY Python runtime owns classifier-free guidance, diffusion,
autoregressive state, root recentering, decoding, and skeleton conversion.
These wrappers deliberately move the expensive neural-network portions and the
decoder-side motion conversion into stable ONNX contracts.  The small two-pass
denoiser remains available for text-only generation, while a separate
constraint graph exposes ARDY's full separated-CFG contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ardy.geometry import cont6d_to_matrix
from ardy.model.minilm_encoder import pool_token_embeddings


class BrowserMiniLMEncoder(nn.Module):
    """Exportable MiniLM student returning one direct root/body condition token.

    Transformers 5.x builds a 4-D attention mask through Python control flow
    that the legacy ONNX tracer cannot reliably capture.  Supplying the already
    expanded additive mask bypasses that control flow while preserving the
    exact BERT attention semantics.
    """

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask_2d = attention_mask
        embedding_dtype = self.student.backbone.embeddings.word_embeddings.weight.dtype
        additive_mask = (1.0 - mask_2d.to(embedding_dtype))[:, None, None, :]
        additive_mask = additive_mask * torch.finfo(embedding_dtype).min

        output = self.student.backbone(
            input_ids=input_ids,
            attention_mask=additive_mask,
            token_type_ids=token_type_ids,
        )
        pooled = pool_token_embeddings(
            output.last_hidden_state,
            mask_2d,
            self.student.pooling_mode,
        )
        if self.student.normalize_embedding:
            pooled = F.normalize(pooled, p=2, dim=-1)
        shared = self.student.adapter(pooled)
        conditions = torch.cat(
            (
                self.student.root_head(shared),
                self.student.body_head(shared),
            ),
            dim=-1,
        )
        return conditions[:, None, :]


class BrowserTextCFGDenoiser(nn.Module):
    """Text-only, two-pass classifier-free-guidance ARDY denoiser.

    Constraints and future tokens are intentionally omitted from the browser
    contract.  The conditional and unconditional passes are concatenated into a
    single B=2 invocation, then combined with ``cfg_weight``.
    """

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        cfg_weight: torch.Tensor,
        x: torch.Tensor,
        history_len: torch.Tensor,
        generation_len: torch.Tensor,
        history_mask: torch.Tensor,
        generation_mask: torch.Tensor,
        history_token_mask: torch.Tensor,
        generation_token_mask: torch.Tensor,
        text_conditions: torch.Tensor,
        timestep: torch.Tensor,
        first_heading_angle: torch.Tensor,
    ) -> torch.Tensor:
        x_2 = torch.cat((x, x), dim=0)
        history_len_2 = torch.cat((history_len, history_len), dim=0)
        generation_len_2 = torch.cat((generation_len, generation_len), dim=0)
        zero_len_2 = torch.zeros_like(history_len_2)

        history_mask_2 = torch.cat((history_mask, history_mask), dim=0)
        generation_mask_2 = torch.cat((generation_mask, generation_mask), dim=0)
        zero_frame_mask_2 = torch.zeros_like(history_mask_2)
        history_token_mask_2 = torch.cat(
            (history_token_mask, history_token_mask),
            dim=0,
        )
        generation_token_mask_2 = torch.cat(
            (generation_token_mask, generation_token_mask),
            dim=0,
        )
        zero_token_mask_2 = torch.zeros_like(history_token_mask_2)

        text_2 = torch.cat(
            (text_conditions, torch.zeros_like(text_conditions)),
            dim=0,
        )
        # Released Core40 has use_text_mask=False, but retain a correctly shaped
        # mask so the wrapper remains semantically complete.
        text_mask_2 = torch.cat(
            (
                torch.ones_like(text_conditions[..., 0]),
                torch.zeros_like(text_conditions[..., 0]),
            ),
            dim=0,
        )
        timestep_2 = torch.cat((timestep, timestep), dim=0)
        heading_2 = torch.cat(
            (first_heading_angle, first_heading_angle),
            dim=0,
        )

        prediction_2 = self.denoiser(
            x=x_2,
            history_len=history_len_2,
            generation_len=generation_len_2,
            future_len=zero_len_2,
            history_mask=history_mask_2 > 0.5,
            generation_mask=generation_mask_2 > 0.5,
            future_mask=zero_frame_mask_2 > 0.5,
            history_token_mask=history_token_mask_2 > 0.5,
            generation_token_mask=generation_token_mask_2 > 0.5,
            future_token_mask=zero_token_mask_2 > 0.5,
            text_feat=text_2,
            text_feat_pad_mask=text_mask_2 > 0.5,
            timesteps=timestep_2,
            first_heading_angle=heading_2,
            motion_mask=None,
            observed_motion=None,
        )
        conditional, unconditional = torch.chunk(prediction_2, 2, dim=0)
        return unconditional + cfg_weight * (conditional - unconditional)


class BrowserConstraintCFGDenoiser(nn.Module):
    """Constraint-capable, three-pass separated-CFG ARDY denoiser.

    This mirrors :class:`AutoLatentClassifierFreeGuidedModelSeparated` without
    optional tensor inputs, which are awkward to represent consistently across
    ONNX Runtime Web backends:

    * pass 0 receives text and no kinematic constraints;
    * pass 1 receives constraints and no text;
    * pass 2 is unconditional.

    Future tokens only participate in the constraint pass.  Supplying zero
    ``motion_mask`` and ``observed_motion`` tensors is therefore a valid way to
    invoke the graph without active constraints, although the smaller
    :class:`BrowserTextCFGDenoiser` is preferred for that case.
    """

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        text_cfg_weight: torch.Tensor,
        constraint_cfg_weight: torch.Tensor,
        x: torch.Tensor,
        history_len: torch.Tensor,
        generation_len: torch.Tensor,
        future_len: torch.Tensor,
        history_mask: torch.Tensor,
        generation_mask: torch.Tensor,
        future_mask: torch.Tensor,
        history_token_mask: torch.Tensor,
        generation_token_mask: torch.Tensor,
        future_token_mask: torch.Tensor,
        text_conditions: torch.Tensor,
        text_condition_mask: torch.Tensor,
        timestep: torch.Tensor,
        first_heading_angle: torch.Tensor,
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
    ) -> torch.Tensor:
        x_3 = torch.cat((x, x, x), dim=0)
        history_len_3 = torch.cat((history_len, history_len, history_len), dim=0)
        generation_len_3 = torch.cat(
            (generation_len, generation_len, generation_len),
            dim=0,
        )
        future_len_3 = torch.cat((future_len, future_len, future_len), dim=0)

        history_mask_3 = torch.cat(
            (history_mask, history_mask, history_mask),
            dim=0,
        )
        generation_mask_3 = torch.cat(
            (generation_mask, generation_mask, generation_mask),
            dim=0,
        )
        future_mask_3 = torch.cat(
            (future_mask, future_mask, future_mask),
            dim=0,
        )
        history_token_mask_3 = torch.cat(
            (history_token_mask, history_token_mask, history_token_mask),
            dim=0,
        )
        generation_token_mask_3 = torch.cat(
            (
                generation_token_mask,
                generation_token_mask,
                generation_token_mask,
            ),
            dim=0,
        )
        future_token_mask_3 = torch.cat(
            (
                torch.zeros_like(future_token_mask),
                future_token_mask,
                torch.zeros_like(future_token_mask),
            ),
            dim=0,
        )

        text_3 = torch.cat(
            (
                text_conditions,
                torch.zeros_like(text_conditions),
                torch.zeros_like(text_conditions),
            ),
            dim=0,
        )
        text_mask_3 = torch.cat(
            (
                text_condition_mask,
                torch.zeros_like(text_condition_mask),
                torch.zeros_like(text_condition_mask),
            ),
            dim=0,
        )
        timestep_3 = torch.cat((timestep, timestep, timestep), dim=0)
        heading_3 = torch.cat(
            (
                first_heading_angle,
                first_heading_angle,
                first_heading_angle,
            ),
            dim=0,
        )
        motion_mask_3 = torch.cat(
            (
                torch.zeros_like(motion_mask),
                motion_mask,
                torch.zeros_like(motion_mask),
            ),
            dim=0,
        )
        observed_motion_3 = torch.cat(
            (
                torch.zeros_like(observed_motion),
                observed_motion,
                torch.zeros_like(observed_motion),
            ),
            dim=0,
        )

        prediction_3 = self.denoiser(
            x=x_3,
            history_len=history_len_3,
            generation_len=generation_len_3,
            future_len=future_len_3,
            history_mask=history_mask_3 > 0.5,
            generation_mask=generation_mask_3 > 0.5,
            future_mask=future_mask_3 > 0.5,
            history_token_mask=history_token_mask_3 > 0.5,
            generation_token_mask=generation_token_mask_3 > 0.5,
            future_token_mask=future_token_mask_3 > 0.5,
            text_feat=text_3,
            text_feat_pad_mask=text_mask_3 > 0.5,
            timesteps=timestep_3,
            first_heading_angle=heading_3,
            motion_mask=motion_mask_3,
            observed_motion=observed_motion_3,
        )
        text_prediction, constraint_prediction, unconditional = torch.chunk(
            prediction_3,
            3,
            dim=0,
        )
        guided = (
            unconditional
            + text_cfg_weight * (text_prediction - unconditional)
            + constraint_cfg_weight * (constraint_prediction - unconditional)
        )
        # Core40 does not consume frame-level future masks/lengths directly
        # (the token mask drives future conditioning), and it was trained with
        # ``use_text_mask=False``.  Keep these tensors in the stable ONNX
        # contract anyway: later compatible checkpoints may use them and the
        # browser runtime should not need a graph-specific feed signature.
        contract_anchor = (
            future_len.to(dtype=guided.dtype).sum()
            + future_mask.to(dtype=guided.dtype).sum()
            + text_condition_mask.to(dtype=guided.dtype).sum()
        ) * guided.new_tensor(0.0)
        return guided + contract_anchor


class BrowserMotionDecoder(nn.Module):
    """Decode hybrid tokens to ARDY features and complete Core27 motion data."""

    def __init__(self, autoencoder: nn.Module, motion_rep) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.motion_rep = motion_rep
        self.num_frames_per_token = int(autoencoder.num_frames_per_token)
        self.nframe_root_dim = int(motion_rep.motion_root_dim) * self.num_frames_per_token
        self.latent_dim = int(autoencoder._latent_embedding_dim)
        self.num_joints = int(motion_rep.skeleton.nbjoints)

        parents = motion_rep.skeleton.joint_parents.detach().to(torch.long)
        neutral = motion_rep.skeleton.neutral_joints.detach().to(torch.float32)
        self.parent_indices = tuple(int(value) for value in parents.tolist())
        self.register_buffer("joint_parents", parents, persistent=False)
        self.register_buffer("neutral_joints", neutral, persistent=False)

    def _posed_joints_from_global_rotations(
        self,
        global_rotations: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> torch.Tensor:
        """ONNX-friendly FK using the checkpoint's global joint rotations."""
        positions = [root_positions]
        for joint_index in range(1, self.num_joints):
            parent_index = self.parent_indices[joint_index]
            offset = (self.neutral_joints[joint_index] - self.neutral_joints[parent_index]).to(
                device=global_rotations.device,
                dtype=global_rotations.dtype,
            )
            rotated_offset = torch.matmul(
                global_rotations[:, :, parent_index],
                offset[:, None],
            ).squeeze(-1)
            positions.append(positions[parent_index] + rotated_offset)
        return torch.stack(positions, dim=2)

    def _local_rotations_from_global_rotations(
        self,
        global_rotations: torch.Tensor,
    ) -> torch.Tensor:
        """ONNX-friendly global-to-local conversion for a fixed skeleton."""
        local_rotations = [global_rotations[:, :, 0]]
        for joint_index in range(1, self.num_joints):
            parent_index = self.parent_indices[joint_index]
            parent_inverse = global_rotations[:, :, parent_index].transpose(-2, -1)
            local_rotations.append(
                torch.matmul(
                    parent_inverse,
                    global_rotations[:, :, joint_index],
                )
            )
        return torch.stack(local_rotations, dim=2)

    def forward(
        self,
        hybrid_tokens: torch.Tensor,
        motion_pad_mask: torch.Tensor,
        global_translation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, num_tokens = hybrid_tokens.shape[:2]
        num_frames = num_tokens * self.num_frames_per_token

        root_motion = hybrid_tokens[:, :, : self.nframe_root_dim].reshape(
            batch_size,
            num_frames,
            self.motion_rep.motion_root_dim,
        )
        latent_body = hybrid_tokens[:, :, self.nframe_root_dim :]

        root_world = self.motion_rep.global_root_stats.unnormalize(root_motion)
        root_world = torch.cat(
            (
                root_world[..., :3] + global_translation[:, None, :],
                root_world[..., 3:],
            ),
            dim=-1,
        )
        root_world = self.motion_rep.global_root_stats.normalize(root_world)

        lengths = (motion_pad_mask > 0.5).sum(dim=-1)
        local_root = self.motion_rep.global_root_to_local_root(
            root_world,
            normalized=True,
            lengths=lengths,
        )
        decoded = self.autoencoder.detokenize(
            latent_body,
            external_cond=local_root,
            motion_pad_mask=motion_pad_mask > 0.5,
        )
        normalized_motion = self.motion_rep.concat_root_body(
            root_world,
            decoded["body"],
        )

        explicit_motion = self.motion_rep.unnormalize(normalized_motion)
        root_positions = explicit_motion[
            ...,
            self.motion_rep.slice_dict["root_pos"],
        ]
        rotation_features = explicit_motion[
            ...,
            self.motion_rep.slice_dict["global_rot_data"],
        ].reshape(
            batch_size,
            num_frames,
            self.num_joints,
            6,
        )
        global_rotations = cont6d_to_matrix(rotation_features)
        local_rotations = self._local_rotations_from_global_rotations(
            global_rotations,
        )
        posed_joints = self._posed_joints_from_global_rotations(
            global_rotations,
            root_positions,
        )
        foot_contacts = (
            explicit_motion[
                ...,
                self.motion_rep.slice_dict["foot_contacts"],
            ]
            > 0.5
        )
        global_root_heading = explicit_motion[
            ...,
            self.motion_rep.slice_dict["global_root_heading"],
        ]
        return (
            normalized_motion,
            posed_joints,
            local_rotations,
            global_rotations,
            root_positions,
            foot_contacts,
            global_root_heading,
        )


__all__ = [
    "BrowserConstraintCFGDenoiser",
    "BrowserMiniLMEncoder",
    "BrowserMotionDecoder",
    "BrowserTextCFGDenoiser",
]
