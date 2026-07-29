# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the browser ONNX export wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch
from torch import nn

from ardy.browser.export import _graph_contracts
from ardy.browser.wrappers import (
    BrowserConstraintCFGDenoiser,
    BrowserMiniLMEncoder,
    BrowserMotionDecoder,
    BrowserTextCFGDenoiser,
)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = SimpleNamespace(
            word_embeddings=nn.Embedding(32, 4),
        )
        self.last_attention_mask = None

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.last_attention_mask = attention_mask
        hidden = self.embeddings.word_embeddings(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.pooling_mode = "mean"
        self.normalize_embedding = False
        self.adapter = nn.Identity()
        self.root_head = nn.Linear(4, 3, bias=False)
        self.body_head = nn.Linear(4, 3, bias=False)


def test_browser_minilm_expands_mask_and_returns_one_condition_token():
    student = _FakeStudent().eval()
    wrapper = BrowserMiniLMEncoder(student).eval()
    input_ids = torch.tensor([[1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    token_type_ids = torch.zeros_like(input_ids)

    output = wrapper(input_ids, attention_mask, token_type_ids)

    assert output.shape == (1, 1, 6)
    expanded = student.backbone.last_attention_mask
    assert expanded.shape == (1, 1, 1, 3)
    assert expanded[0, 0, 0, :2].eq(0).all()
    assert expanded[0, 0, 0, 2] < -1e20


class _CaptureDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        text_scalar = kwargs["text_feat"].mean(dim=(1, 2))[:, None, None]
        if kwargs["motion_mask"] is None:
            constraint_scalar = torch.zeros_like(text_scalar)
        else:
            constraint_scalar = kwargs["observed_motion"].mul(kwargs["motion_mask"]).mean(dim=(1, 2))[:, None, None]
        return kwargs["x"] + text_scalar + constraint_scalar


def test_browser_denoiser_runs_two_pass_text_only_cfg():
    inner = _CaptureDenoiser()
    wrapper = BrowserTextCFGDenoiser(inner).eval()
    x = torch.zeros(1, 2, 5)
    one_frame = torch.ones(1, 8)
    one_token = torch.ones(1, 2)

    output = wrapper(
        torch.tensor([2.5]),
        x,
        torch.tensor([0]),
        torch.tensor([8]),
        torch.zeros_like(one_frame),
        one_frame,
        torch.zeros_like(one_token),
        one_token,
        torch.full((1, 1, 6), 2.0),
        torch.tensor([9]),
        torch.zeros(1),
    )

    assert torch.allclose(output, torch.full_like(output, 5.0))
    captured = inner.last_kwargs
    assert captured["x"].shape[0] == 2
    assert captured["motion_mask"] is None
    assert captured["observed_motion"] is None
    assert not captured["future_mask"].any()
    assert not captured["future_token_mask"].any()


def test_browser_constraint_denoiser_matches_separated_cfg_semantics():
    inner = _CaptureDenoiser()
    wrapper = BrowserConstraintCFGDenoiser(inner).eval()
    x = torch.zeros(1, 3, 5)
    frame_count = 12
    token_count = 3
    history_mask = torch.zeros(1, frame_count)
    history_mask[:, :4] = 1
    generation_mask = torch.zeros(1, frame_count)
    generation_mask[:, 4:8] = 1
    future_mask = torch.zeros(1, frame_count)
    future_mask[:, 8:] = 1
    history_token_mask = torch.tensor([[1.0, 0.0, 0.0]])
    generation_token_mask = torch.tensor([[0.0, 1.0, 0.0]])
    future_token_mask = torch.tensor([[0.0, 0.0, 1.0]])
    motion_mask = torch.ones(1, frame_count, 2)
    observed_motion = torch.full_like(motion_mask, 3.0)

    output = wrapper(
        torch.tensor([2.0]),
        torch.tensor([1.5]),
        x,
        torch.tensor([4]),
        torch.tensor([4]),
        torch.tensor([4]),
        history_mask,
        generation_mask,
        future_mask,
        history_token_mask,
        generation_token_mask,
        future_token_mask,
        torch.full((1, 1, 6), 2.0),
        torch.ones(1, 1),
        torch.tensor([9]),
        torch.zeros(1),
        motion_mask,
        observed_motion,
    )

    assert output.shape == (1, token_count, 5)
    assert torch.allclose(output, torch.full_like(output, 8.5))
    captured = inner.last_kwargs
    assert captured["x"].shape[0] == 3
    assert torch.equal(
        captured["text_feat"].mean(dim=(1, 2)),
        torch.tensor([2.0, 0.0, 0.0]),
    )
    assert not captured["future_token_mask"][0].any()
    assert captured["future_token_mask"][1, 2]
    assert not captured["future_token_mask"][2].any()
    assert not captured["motion_mask"][0].any()
    assert captured["motion_mask"][1].all()
    assert not captured["motion_mask"][2].any()


class _IdentityStats:
    def __init__(self, size: int) -> None:
        self.mean = torch.zeros(size)
        self.std = torch.ones(size)
        self.std_eps = torch.ones(size)

    def normalize(self, value):
        return value

    def unnormalize(self, value):
        return value


class _FakeAutoencoder(nn.Module):
    num_frames_per_token = 4
    _latent_embedding_dim = 128

    def detokenize(self, latent_body, external_cond, motion_pad_mask):
        batch, tokens = latent_body.shape[:2]
        frames = tokens * self.num_frames_per_token
        identity_6d = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            device=latent_body.device,
            dtype=latent_body.dtype,
        )
        rotations = identity_6d.repeat(batch, frames, 3)
        contacts = torch.zeros(
            batch,
            frames,
            2,
            device=latent_body.device,
            dtype=latent_body.dtype,
        )
        body = torch.cat((rotations, contacts), dim=-1)
        return {"body": body}


class _FakeMotionRep:
    motion_root_dim = 5
    motion_rep_dim = 25
    body_dim = 20
    slice_dict: ClassVar = {
        "root_pos": slice(0, 3),
        "global_rot_data": slice(5, 23),
        "foot_contacts": slice(23, 25),
        "global_root_heading": slice(3, 5),
    }

    def __init__(self) -> None:
        self.global_root_stats = _IdentityStats(5)
        self.stats = _IdentityStats(25)
        self.skeleton = SimpleNamespace(
            nbjoints=3,
            joint_parents=torch.tensor([-1, 0, 1]),
            neutral_joints=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
        )

    def global_root_to_local_root(self, root, normalized, lengths):
        return torch.zeros(
            root.shape[0],
            root.shape[1],
            4,
            device=root.device,
            dtype=root.dtype,
        )

    def concat_root_body(self, root, body):
        return torch.cat((root, body), dim=-1)

    def unnormalize(self, value):
        return value


def test_browser_decoder_returns_world_motion_and_posed_joints():
    decoder = BrowserMotionDecoder(
        _FakeAutoencoder(),
        _FakeMotionRep(),
    ).eval()
    hybrid = torch.zeros(1, 2, 148)
    mask = torch.ones(1, 8)
    translation = torch.tensor([[1.0, 2.0, 3.0]])

    (
        motion,
        joints,
        local_rotations,
        global_rotations,
        root_positions,
        foot_contacts,
        global_root_heading,
    ) = decoder(hybrid, mask, translation)

    assert motion.shape == (1, 8, 25)
    assert joints.shape == (1, 8, 3, 3)
    assert local_rotations.shape == (1, 8, 3, 3, 3)
    assert global_rotations.shape == (1, 8, 3, 3, 3)
    assert root_positions.shape == (1, 8, 3)
    assert foot_contacts.shape == (1, 8, 2)
    assert foot_contacts.dtype == torch.bool
    assert global_root_heading.shape == (1, 8, 2)
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
        ]
    )
    assert torch.allclose(joints[0, 0], expected)
    assert torch.allclose(root_positions[0, 0], translation[0])
    identity = torch.eye(3).expand(3, 3, 3)
    assert torch.allclose(local_rotations[0, 0], identity)
    assert torch.allclose(global_rotations[0, 0], identity)


def test_manifest_graph_semantics_match_browser_runtime_contract():
    graphs = _graph_contracts()

    assert graphs["text_encoder"]["outputs"] == {"textConditions": "text_conditions"}
    assert graphs["denoiser"]["inputs"]["historyLength"] == "history_len"
    assert graphs["denoiser"]["inputs"]["generationLength"] == "generation_len"
    assert graphs["constraint_denoiser"]["inputs"]["textCfgWeight"] == "text_cfg_weight"
    assert graphs["constraint_denoiser"]["inputs"]["constraintCfgWeight"] == "constraint_cfg_weight"
    assert graphs["constraint_denoiser"]["inputs"]["futureLength"] == "future_len"
    assert graphs["constraint_denoiser"]["inputs"]["motionMask"] == "motion_mask"
    assert graphs["constraint_denoiser"]["inputs"]["observedMotion"] == "observed_motion"
    assert graphs["decoder"]["outputs"] == {
        "normalizedMotion": "normalized_motion",
        "posedJoints": "posed_joints",
        "localRotations": "local_rotations",
        "globalRotations": "global_rotations",
        "rootPositions": "root_positions",
        "footContacts": "foot_contacts",
        "globalRootHeading": "global_root_heading",
    }
