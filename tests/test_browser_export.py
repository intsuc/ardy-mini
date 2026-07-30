# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the browser ONNX export wrappers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from ardy.browser.export import (
    BrowserExportConfig,
    _build_fp32_reference_payload,
    _graph_contracts,
    _local_checkpoint_identity,
    _publish_archive_set,
    _specialize_denoiser_position_tables,
    _validate_config,
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


def test_browser_export_validates_optional_fp32_reference_output(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "mixed.tar.gz"

    with pytest.raises(ValueError, match="different paths"):
        _validate_config(
            BrowserExportConfig(
                output_path=output,
                fp32_reference_output_path=tmp_path / "." / output.name,
                minilm_artifact=artifact,
            )
        )

    with pytest.raises(ValueError, match=r"must end in \.tar\.gz"):
        _validate_config(
            BrowserExportConfig(
                output_path=output,
                fp32_reference_output_path=tmp_path / "reference.tgz",
                minilm_artifact=artifact,
            )
        )

    reference_directory = tmp_path / "reference.tar.gz"
    reference_directory.mkdir()
    with pytest.raises(IsADirectoryError, match="FP32 reference output"):
        _validate_config(
            BrowserExportConfig(
                output_path=output,
                fp32_reference_output_path=reference_directory,
                minilm_artifact=artifact,
            )
        )


def _without_pack_specific_metadata(manifest: dict) -> dict:
    contract = copy.deepcopy(manifest)
    for key in ("files", "precision", "verification"):
        contract.pop(key, None)
    return contract


def test_fp32_reference_payload_matches_candidate_contract_and_is_reproducible(
    tmp_path: Path,
):
    candidate_dir = tmp_path / "candidate"
    tokenizer_dir = candidate_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    tokenizer_paths = [
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
    ]
    tokenizer_paths[0].write_bytes(b'{"tokenizer":"identical"}\n')
    tokenizer_paths[1].write_bytes(b'{"max_length":128}\n')

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    graph_payloads = {
        "text_encoder": b"original fp32 text graph",
        "denoiser": b"original fp32 denoiser graph",
        "decoder": b"original fp32 decoder graph",
    }
    graph_paths = {}
    for graph_name, payload in graph_payloads.items():
        graph_path = reference_dir / f"{graph_name}.onnx"
        graph_path.write_bytes(payload)
        graph_paths[graph_name] = graph_path

    candidate_manifest = {
        "format": "ardy-browser-model-pack",
        "schema_version": 2,
        "model": {"id": "test-model", "variant": "test"},
        "files": {"mixed-only.onnx": {"sha256": "candidate", "size_bytes": 1}},
        "tokenizer": {"directory": "tokenizer", "max_length": 128},
        "graphs": {graph_name: {"model": f"{graph_name}.onnx"} for graph_name in graph_payloads},
        "dimensions": {"text_condition_dim": 2048},
        "runtime": {
            "contract_revision": 3,
            "required_webgpu_features": ["shader-f16"],
        },
        "precision": {
            "format": "mixed-fp16",
            "toolchain": {
                "onnx": "test",
                "onnxruntime": "test",
                "torch": "test",
            },
        },
        "verification": {"mixed_precision": {"status": "passed"}},
    }
    fp32_verification = {"backend": "onnxruntime-cpu", "status": "passed"}

    members = _build_fp32_reference_payload(
        candidate_manifest=candidate_manifest,
        output_dir=reference_dir,
        graph_paths=graph_paths,
        tokenizer_paths=tokenizer_paths,
        verification=fp32_verification,
    )
    reference_manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))

    assert _without_pack_specific_metadata(reference_manifest) == (_without_pack_specific_metadata(candidate_manifest))
    assert reference_manifest["precision"] == {
        "format": "fp32",
        "public_io_dtype": "float32",
        "required_webgpu_features": [],
        "toolchain": candidate_manifest["precision"]["toolchain"],
        "onnx_bytes": sum(map(len, graph_payloads.values())),
        "graphs": {
            graph_name: {
                "model": f"{graph_name}.onnx",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for graph_name, payload in sorted(graph_payloads.items())
        },
    }
    assert reference_manifest["verification"] == {"fp32_export": fp32_verification}
    for source in tokenizer_paths:
        copied = reference_dir / "tokenizer" / source.name
        assert copied.read_bytes() == source.read_bytes()
        assert (
            reference_manifest["files"][f"tokenizer/{source.name}"]["sha256"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )

    first = tmp_path / "reference-first.tar.gz"
    second = tmp_path / "reference-second.tar.gz"
    _write_deterministic_tar_gz(reference_dir, members, first)
    _write_deterministic_tar_gz(reference_dir, members, second)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "manifest.json",
            "decoder.onnx",
            "denoiser.onnx",
            "text_encoder.onnx",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ]


def test_local_checkpoint_identity_binds_all_source_files_without_local_path(
    tmp_path: Path,
):
    model_name = "ARDY-Core-RP-20FPS-Horizon40"
    checkpoint_dir = tmp_path / model_name
    checkpoint_dir.mkdir()
    payloads = {
        "config.yaml": b"model: test\n",
        "denoiser.safetensors": b"denoiser",
        "tokenizer.safetensors": b"tokenizer",
    }
    for filename, payload in payloads.items():
        (checkpoint_dir / filename).write_bytes(payload)

    identity = _local_checkpoint_identity(tmp_path, model_name)

    assert identity is not None
    assert identity["format"] == "ardy-local-checkpoint-files"
    assert set(identity["files"]) == set(payloads)
    assert len(identity["fingerprint"]) == 64
    encoded = json.dumps(identity)
    assert str(tmp_path) not in encoded
    for filename, payload in payloads.items():
        assert identity["files"][filename] == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    (checkpoint_dir / "denoiser.safetensors").write_bytes(b"changed")
    changed = _local_checkpoint_identity(tmp_path, model_name)
    assert changed is not None
    assert changed["fingerprint"] != identity["fingerprint"]


def test_archive_set_publication_rolls_back_both_destinations_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    candidate_stage = tmp_path / "candidate.stage"
    reference_stage = tmp_path / "reference.stage"
    candidate_output = tmp_path / "candidate.tar.gz"
    reference_output = tmp_path / "reference.tar.gz"
    candidate_stage.write_bytes(b"new candidate")
    reference_stage.write_bytes(b"new reference")
    candidate_output.write_bytes(b"old candidate")
    reference_output.write_bytes(b"old reference")

    real_replace = os.replace
    failed = False

    def fail_second_publish(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(source) == reference_stage
            and Path(destination) == reference_output
        ):
            failed = True
            raise OSError("simulated reference publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr("ardy.browser.export.os.replace", fail_second_publish)

    with pytest.raises(OSError, match="simulated"):
        _publish_archive_set(
            [
                (candidate_stage, candidate_output),
                (reference_stage, reference_output),
            ]
        )

    assert candidate_output.read_bytes() == b"old candidate"
    assert reference_output.read_bytes() == b"old reference"
    assert candidate_stage.read_bytes() == b"new candidate"
    assert reference_stage.read_bytes() == b"new reference"


def test_archive_set_publication_replaces_both_destinations(tmp_path: Path):
    candidate_stage = tmp_path / "candidate.stage"
    reference_stage = tmp_path / "reference.stage"
    candidate_output = tmp_path / "candidate.tar.gz"
    reference_output = tmp_path / "reference.tar.gz"
    candidate_stage.write_bytes(b"new candidate")
    reference_stage.write_bytes(b"new reference")
    candidate_output.write_bytes(b"old candidate")
    reference_output.write_bytes(b"old reference")

    _publish_archive_set(
        [
            (candidate_stage, candidate_output),
            (reference_stage, reference_output),
        ]
    )

    assert candidate_output.read_bytes() == b"new candidate"
    assert reference_output.read_bytes() == b"new reference"
    assert not candidate_stage.exists()
    assert not reference_stage.exists()
