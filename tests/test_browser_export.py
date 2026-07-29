# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the browser ONNX export wrappers."""

from __future__ import annotations

import tarfile
from types import SimpleNamespace
from typing import ClassVar

import torch
from torch import nn

from ardy.browser.export import (
    _graph_contracts,
    _specialize_denoiser_position_tables,
    _write_deterministic_tar_gz,
)
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

    assert set(graphs) == {"text_encoder", "denoiser", "decoder"}
    assert graphs["text_encoder"]["outputs"] == {"textConditions": "text_conditions"}
    assert graphs["denoiser"]["inputs"]["historyLength"] == "history_len"
    assert graphs["denoiser"]["inputs"]["generationLength"] == "generation_len"
    assert graphs["decoder"]["outputs"] == {
        "normalizedMotion": "normalized_motion",
        "posedJoints": "posed_joints",
        "localRotations": "local_rotations",
        "globalRotations": "global_rotations",
        "rootPositions": "root_positions",
        "footContacts": "foot_contacts",
        "globalRootHeading": "global_root_heading",
    }


def _fake_position_block() -> SimpleNamespace:
    timestep_encoder = SimpleNamespace(pe=torch.arange(24, dtype=torch.float32).reshape(1, 12, 2))
    motion_encoder = SimpleNamespace(
        max_len=6,
        pe=torch.arange(22, dtype=torch.float32).reshape(11, 2),
    )
    return SimpleNamespace(
        positional_encoding_mode="learned_prefix_zero_at_first_generation",
        sequence_pos_encoder=timestep_encoder,
        embed_timestep=SimpleNamespace(sequence_pos_encoder=timestep_encoder),
        motion_token_embedding=motion_encoder,
    )


def test_browser_position_tables_are_trimmed_without_changing_reachable_values():
    root = _fake_position_block()
    body = _fake_position_block()
    denoiser = SimpleNamespace(root_model=root, body_model=body)
    original_root_timestep = root.sequence_pos_encoder.pe.clone()
    original_root_motion = root.motion_token_embedding.pe.clone()

    _specialize_denoiser_position_tables(
        denoiser,
        num_timesteps=4,
        max_motion_tokens=3,
    )

    assert root.sequence_pos_encoder.pe.shape == (1, 4, 2)
    assert torch.equal(root.sequence_pos_encoder.pe, original_root_timestep[:, :4])
    assert root.embed_timestep.sequence_pos_encoder is root.sequence_pos_encoder
    assert root.motion_token_embedding.max_len == 3
    assert root.motion_token_embedding.pe.shape == (5, 2)
    assert torch.equal(
        root.motion_token_embedding.pe,
        torch.cat((original_root_motion[:3], original_root_motion[-2:]), dim=0),
    )
    assert body.sequence_pos_encoder.pe.shape == (1, 4, 2)
    assert body.motion_token_embedding.pe.shape == (5, 2)


def test_browser_archive_is_reproducible_ustar_with_root_level_members(tmp_path):
    source = tmp_path / "source"
    tokenizer = source / "tokenizer"
    tokenizer.mkdir(parents=True)
    manifest = source / "manifest.json"
    model = source / "denoiser.onnx"
    tokenizer_json = tokenizer / "tokenizer.json"
    manifest.write_text('{"schema_version":2}\n', encoding="utf-8")
    model.write_bytes(b"onnx payload")
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    members = [tokenizer_json, model, manifest]
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    _write_deterministic_tar_gz(source, members, first)
    _write_deterministic_tar_gz(source, members, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_mode & 0o777 == 0o644
    with tarfile.open(first, mode="r:gz") as archive:
        infos = archive.getmembers()
        assert [info.name for info in infos] == [
            "manifest.json",
            "denoiser.onnx",
            "tokenizer/tokenizer.json",
        ]
        assert all(info.isfile() for info in infos)
        assert all(info.mtime == 0 for info in infos)
        assert all(info.uid == 0 and info.gid == 0 for info in infos)
        assert all(info.mode == 0o644 for info in infos)
        assert archive.extractfile("denoiser.onnx").read() == b"onnx payload"
