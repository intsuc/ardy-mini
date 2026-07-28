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
        return kwargs["x"] + text_scalar


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
        body = identity_6d.repeat(batch, frames, 3)
        return {"body": body}


class _FakeMotionRep:
    motion_root_dim = 5
    motion_rep_dim = 23
    body_dim = 18
    slice_dict: ClassVar = {
        "root_pos": slice(0, 3),
        "global_rot_data": slice(5, 23),
    }

    def __init__(self) -> None:
        self.global_root_stats = _IdentityStats(5)
        self.stats = _IdentityStats(23)
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

    motion, joints = decoder(hybrid, mask, translation)

    assert motion.shape == (1, 8, 23)
    assert joints.shape == (1, 8, 3, 3)
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
        ]
    )
    assert torch.allclose(joints[0, 0], expected)


def test_manifest_graph_semantics_match_browser_runtime_contract():
    graphs = _graph_contracts()

    assert graphs["text_encoder"]["outputs"] == {"textConditions": "text_conditions"}
    assert graphs["denoiser"]["inputs"]["historyLength"] == "history_len"
    assert graphs["denoiser"]["inputs"]["generationLength"] == "generation_len"
    assert graphs["decoder"]["outputs"] == {
        "normalizedMotion": "normalized_motion",
        "posedJoints": "posed_joints",
    }
