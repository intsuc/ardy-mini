# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Small, explicit ONNX entry points used by the browser model pack.

The regular ARDY Python runtime owns classifier-free guidance, diffusion,
autoregressive state, root recentering, decoding, and skeleton conversion.
These wrappers deliberately move the expensive neural-network portions and the
decoder-side motion conversion into three stable ONNX contracts.
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


class BrowserMotionDecoder(nn.Module):
    """Decode fixed-window hybrid tokens to ARDY features and Core27 joints."""

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

    def forward(
        self,
        hybrid_tokens: torch.Tensor,
        motion_pad_mask: torch.Tensor,
        global_translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        posed_joints = self._posed_joints_from_global_rotations(
            global_rotations,
            root_positions,
        )
        return normalized_motion, posed_joints


__all__ = [
    "BrowserMiniLMEncoder",
    "BrowserMotionDecoder",
    "BrowserTextCFGDenoiser",
]
