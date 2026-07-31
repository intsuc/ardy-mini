# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the browser ONNX export wrappers."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from ardy.browser.export import (
    BROWSER_MODEL_FILES_FORMAT,
    BROWSER_MODEL_FILES_SCHEMA_VERSION,
    BrowserExportConfig,
    _build_fp32_payload,
    _graph_contracts,
    _local_checkpoint_identity,
    _model_revision,
    _public_minilm_lineage,
    _publish_directory_set,
    _source_code_identity,
    _specialize_denoiser_position_tables,
    _validate_config,
    _write_model_files_directory,
    export_browser_model_files,
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


def test_browser_model_files_are_reproducible_and_only_keep_compressed_assets(
    tmp_path: Path,
):
    source = tmp_path / "source"
    tokenizer = source / "tokenizer"
    tokenizer.mkdir(parents=True)
    model = source / "denoiser.onnx"
    tokenizer_json = tokenizer / "tokenizer.json"
    model.write_bytes(b"onnx payload")
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    payloads = [tokenizer_json, model]
    manifest = {
        "format": BROWSER_MODEL_FILES_FORMAT,
        "schema_version": BROWSER_MODEL_FILES_SCHEMA_VERSION,
        "model": {"revision": "1" * 64},
        "files": {
            path.relative_to(source).as_posix(): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in payloads
        },
    }
    first = tmp_path / "first"
    second = tmp_path / "second"

    _write_model_files_directory(
        source_directory=source,
        payload_paths=payloads,
        manifest=manifest,
        output_directory=first,
    )
    _write_model_files_directory(
        source_directory=source,
        payload_paths=list(reversed(payloads)),
        manifest=manifest,
        output_directory=second,
    )

    first_paths = sorted(
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file()
    )
    assert first_paths == [
        "denoiser.onnx.gz",
        "model.json.gz",
        "tokenizer/tokenizer.json.gz",
    ]
    for relative_path in first_paths:
        assert (first / relative_path).read_bytes() == (
            second / relative_path
        ).read_bytes()
        assert (first / relative_path).stat().st_mode & 0o777 == 0o644

    finalized = json.loads(
        gzip.decompress((first / "model.json.gz").read_bytes()).decode("utf-8")
    )
    assert finalized["format"] == "ardy-browser-model-files"
    assert finalized["schema_version"] == 1
    for relative_path, source_path in (
        ("denoiser.onnx", model),
        ("tokenizer/tokenizer.json", tokenizer_json),
    ):
        compressed = first / f"{relative_path}.gz"
        assert compressed.read_bytes()[:2] == b"\x1f\x8b"
        assert compressed.read_bytes()[4:8] == b"\0\0\0\0"
        assert gzip.decompress(compressed.read_bytes()) == source_path.read_bytes()
        record = finalized["files"][relative_path]
        assert record["sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        assert record["size_bytes"] == source_path.stat().st_size
        assert record["transport"] == {
            "path": f"{relative_path}.gz",
            "compression": "gzip",
            "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
            "size_bytes": compressed.stat().st_size,
        }
        assert not (first / relative_path).exists()


def test_browser_export_validates_model_family_output(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    output_file = tmp_path / "model-family"
    output_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="model output"):
        _validate_config(
            BrowserExportConfig(
                output_directory=output_file,
                minilm_artifact=artifact,
                checkpoints_dir=checkpoints,
            )
        )

    with pytest.raises(ValueError, match="must not overlap"):
        _validate_config(
            BrowserExportConfig(
                output_directory=artifact / "browser",
                minilm_artifact=artifact,
                checkpoints_dir=checkpoints,
            )
        )

    with pytest.raises(ValueError, match="model ID is fixed"):
        _validate_config(
            BrowserExportConfig(
                output_directory=tmp_path / "output",
                minilm_artifact=artifact,
                checkpoints_dir=checkpoints,
                model_id="ardy-without-required-llama-prefix",
            )
        )


def _without_file_specific_metadata(manifest: dict) -> dict:
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
        "format": "ardy-browser-model-files",
        "schema_version": 1,
        "model": {
            "id": "test-model",
            "revision": "1" * 64,
            "variant": "test",
        },
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

    reference_manifest, reference_payloads = _build_fp32_payload(
        candidate_manifest=candidate_manifest,
        output_dir=reference_dir,
        graph_paths=graph_paths,
        tokenizer_paths=tokenizer_paths,
        verification=fp32_verification,
    )

    candidate_contract = _without_file_specific_metadata(candidate_manifest)
    candidate_contract["runtime"]["required_webgpu_features"] = []
    assert _without_file_specific_metadata(reference_manifest) == candidate_contract
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
    assert reference_manifest["runtime"]["required_webgpu_features"] == []
    for source in tokenizer_paths:
        copied = reference_dir / "tokenizer" / source.name
        assert copied.read_bytes() == source.read_bytes()
        assert (
            reference_manifest["files"][f"tokenizer/{source.name}"]["sha256"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )

    first = tmp_path / "reference-first"
    second = tmp_path / "reference-second"
    _write_model_files_directory(
        source_directory=reference_dir,
        payload_paths=reference_payloads,
        manifest=reference_manifest,
        output_directory=first,
    )
    _write_model_files_directory(
        source_directory=reference_dir,
        payload_paths=reference_payloads,
        manifest=reference_manifest,
        output_directory=second,
    )

    assert (first / "model.json.gz").read_bytes() == (
        second / "model.json.gz"
    ).read_bytes()
    assert sorted(
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file()
    ) == [
        "decoder.onnx.gz",
        "denoiser.onnx.gz",
        "model.json.gz",
        "text_encoder.onnx.gz",
        "tokenizer/tokenizer.json.gz",
        "tokenizer/tokenizer_config.json.gz",
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
        "stats/motion/mean.npy": b"motion-mean",
        "stats/motion/std.npy": b"motion-std",
        "stats/post_quantization/mean.npy": b"post-mean",
        "stats/post_quantization/std.npy": b"post-std",
        "stats/pre_quantization/mean.npy": b"pre-mean",
        "stats/pre_quantization/std.npy": b"pre-std",
    }
    for filename, payload in payloads.items():
        path = checkpoint_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    identity = _local_checkpoint_identity(tmp_path, model_name)

    assert identity is not None
    assert identity["format"] == "ardy-local-checkpoint-files"
    assert identity["format_version"] == 2
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

    (checkpoint_dir / "stats/post_quantization/std.npy").write_bytes(b"changed-stats")
    changed_stats = _local_checkpoint_identity(tmp_path, model_name)
    assert changed_stats is not None
    assert changed_stats["fingerprint"] != changed["fingerprint"]


def test_local_checkpoint_identity_records_one_common_hub_revision(tmp_path: Path):
    model_name = "ARDY-Core-RP-20FPS-Horizon40"
    checkpoint_dir = tmp_path / model_name
    revision = "a" * 40
    filenames = (
        "config.yaml",
        "denoiser.safetensors",
        "tokenizer.safetensors",
        "stats/motion/mean.npy",
        "stats/motion/std.npy",
        "stats/post_quantization/mean.npy",
        "stats/post_quantization/std.npy",
        "stats/pre_quantization/mean.npy",
        "stats/pre_quantization/std.npy",
    )
    for filename in filenames:
        payload = checkpoint_dir / filename
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(filename.encode())
        metadata = (
            checkpoint_dir
            / ".cache"
            / "huggingface"
            / "download"
            / f"{filename}.metadata"
        )
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            f"{revision}\nopaque-etag\n1234.5\n",
            encoding="utf-8",
        )

    identity = _local_checkpoint_identity(tmp_path, model_name)

    assert identity is not None
    assert identity["source"] == {
        "provider": "huggingface",
        "repo_id": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
        "revision": revision,
    }
    assert str(checkpoint_dir / ".cache") not in json.dumps(identity)

    mismatched = (
        checkpoint_dir
        / ".cache"
        / "huggingface"
        / "download"
        / "stats/motion/std.npy.metadata"
    )
    mismatched.write_text(f"{'b' * 40}\netag\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different Hugging Face revisions"):
        _local_checkpoint_identity(tmp_path, model_name)


def _public_lineage_fixture() -> tuple[dict, dict]:
    denoiser_hash = "d" * 64
    artifact_config = {
        "format_version": 2,
        "artifact_fingerprint": "a" * 64,
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "compatible_ardy_models": ["ARDY-Core-RP-20FPS-Horizon40"],
        "metadata": {
            "target_definition": "[W_root @ teacher, W_body @ teacher]",
            "teacher_cache_fingerprint": "b" * 64,
            "training": {
                "ardy_model": "ARDY-Core-RP-20FPS-Horizon40",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "base_model_revision": "1" * 40,
            },
            "teacher_cache_lineage": {
                "checkpoint_sha256": denoiser_hash,
                "foundation_model_name_or_path": (
                    "meta-llama/Meta-Llama-3-8B-Instruct"
                ),
                "foundation_model_revision": "2" * 40,
                "base_model_name_or_path": (
                    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
                ),
                "base_model_revision": "3" * 40,
                "peft_model_name_or_path": (
                    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
                ),
                "peft_model_revision": "4" * 40,
                "bias_applied": False,
                "device": {"name": "private-machine-detail"},
                "shard_sha256": {"teacher-00000.pt": "e" * 64},
                "corpus_provenance": {
                    "dataset": {
                        "repo_id": "nvidia/SEED-Timeline-Annotations",
                        "revision": "5" * 40,
                        "filename": "timelines.jsonl",
                        "sha256": "6" * 64,
                        "size_bytes": 123,
                        "license": "CC BY 4.0",
                        "owner": "NVIDIA",
                        "url": (
                            "https://huggingface.co/datasets/"
                            "nvidia/SEED-Timeline-Annotations"
                        ),
                        "local_path": "/private/timelines.jsonl",
                    },
                    "manifest": {
                        "filename": "prompts.jsonl",
                        "sha256": "7" * 64,
                    },
                    "preparation": {"normalization": "NFKC"},
                    "counts": {"written": 10},
                },
            },
        },
    }
    checkpoint_identity = {
        "fingerprint": "c" * 64,
        "files": {
            "denoiser.safetensors": {
                "sha256": denoiser_hash,
                "size_bytes": 10,
            }
        },
    }
    return artifact_config, checkpoint_identity


def test_public_minilm_lineage_is_pinned_and_excludes_private_cache_details():
    artifact_config, checkpoint_identity = _public_lineage_fixture()

    lineage = _public_minilm_lineage(
        artifact_config,
        resolved_model="ARDY-Core-RP-20FPS-Horizon40",
        checkpoint_identity=checkpoint_identity,
    )

    assert lineage["student"]["base_model"] == {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1" * 40,
    }
    assert lineage["teacher"]["foundation_model"]["repo_id"] == (
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    assert lineage["dataset"]["revision"] == "5" * 40
    encoded = json.dumps(lineage)
    assert "private-machine-detail" not in encoded
    assert "/private/timelines.jsonl" not in encoded
    assert "teacher-00000.pt" not in encoded

    artifact_config["metadata"]["teacher_cache_lineage"][
        "checkpoint_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="different ARDY denoiser"):
        _public_minilm_lineage(
            artifact_config,
            resolved_model="ARDY-Core-RP-20FPS-Horizon40",
            checkpoint_identity=checkpoint_identity,
        )


def test_source_code_identity_accepts_pinned_archive_commit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ARDY_SOURCE_GIT_COMMIT", "8" * 40)

    assert _source_code_identity() == {
        "repository": "https://github.com/intsuc/ardy-mini",
        "commit": "8" * 40,
    }


def test_model_revision_is_canonical_and_changes_with_either_weight_identity():
    minilm_fingerprint = "1" * 64
    checkpoint_fingerprint = "2" * 64
    source_commit = "5" * 40
    checkpoint_identity = {"fingerprint": checkpoint_fingerprint}
    identity = {
        "format": "ardy-browser-model-identity",
        "schema_version": 2,
        "ardy_model": "ARDY-Core-RP-20FPS-Horizon40",
        "ardy_checkpoint_fingerprint": checkpoint_fingerprint,
        "minilm_artifact_fingerprint": minilm_fingerprint,
        "source_code_commit": source_commit,
    }
    expected = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    actual = _model_revision(
        resolved_model=identity["ardy_model"],
        minilm_artifact_fingerprint=minilm_fingerprint,
        checkpoint_identity=checkpoint_identity,
        source_code_identity={"commit": source_commit},
    )

    assert actual == expected
    assert (
        _model_revision(
            resolved_model=identity["ardy_model"],
            minilm_artifact_fingerprint="3" * 64,
            checkpoint_identity=checkpoint_identity,
            source_code_identity={"commit": source_commit},
        )
        != actual
    )
    assert (
        _model_revision(
            resolved_model=identity["ardy_model"],
            minilm_artifact_fingerprint=minilm_fingerprint,
            checkpoint_identity={"fingerprint": "4" * 64},
            source_code_identity={"commit": source_commit},
        )
        != actual
    )
    assert (
        _model_revision(
            resolved_model=identity["ardy_model"],
            minilm_artifact_fingerprint=minilm_fingerprint,
            checkpoint_identity=checkpoint_identity,
            source_code_identity={"commit": "6" * 40},
        )
        != actual
    )


def test_export_browser_model_files_publishes_fp16_and_fp32_model_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "model-family"
    output.mkdir()
    (output / "stale").write_text("old", encoding="utf-8")

    def fake_working_export(_config, working_directory):
        candidate = working_directory / "denoiser.onnx"
        candidate.write_bytes(b"mixed")
        reference_directory = working_directory / ".fp32"
        reference_directory.mkdir()
        reference = reference_directory / "denoiser.onnx"
        reference.write_bytes(b"fp32")

        def manifest_for(path: Path) -> dict:
            return {
                "format": BROWSER_MODEL_FILES_FORMAT,
                "schema_version": BROWSER_MODEL_FILES_SCHEMA_VERSION,
                "model": {"revision": "a" * 64},
                "files": {
                    "denoiser.onnx": {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
                },
            }

        return (
            manifest_for(candidate),
            [candidate],
            manifest_for(reference),
            [reference],
        )

    monkeypatch.setattr("ardy.browser.export._validate_config", lambda _config: None)
    monkeypatch.setattr(
        "ardy.browser.export._export_browser_model_files_working_directory",
        fake_working_export,
    )

    result = export_browser_model_files(
        BrowserExportConfig(
            output_directory=output,
            minilm_artifact=tmp_path / "unused",
        )
    )

    assert result == output
    assert not (output / "stale").exists()
    assert gzip.decompress(
        (output / "fp16" / "denoiser.onnx.gz").read_bytes()
    ) == b"mixed"
    assert gzip.decompress(
        (output / "fp32" / "denoiser.onnx.gz").read_bytes()
    ) == b"fp32"
    assert (output / "fp16" / "model.json.gz").is_file()
    assert (output / "fp32" / "model.json.gz").is_file()
    assert not (output / "fp16" / "model.json").exists()
    assert not (output / "fp32" / "model.json").exists()


def test_directory_publication_rolls_back_both_destinations_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    candidate_stage = tmp_path / "candidate.stage"
    reference_stage = tmp_path / "reference.stage"
    candidate_output = tmp_path / "candidate"
    reference_output = tmp_path / "reference"
    for directory, payload in (
        (candidate_stage, b"new candidate"),
        (reference_stage, b"new reference"),
        (candidate_output, b"old candidate"),
        (reference_output, b"old reference"),
    ):
        directory.mkdir()
        (directory / "model.json.gz").write_bytes(payload)

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
        _publish_directory_set(
            [
                (candidate_stage, candidate_output),
                (reference_stage, reference_output),
            ]
        )

    assert (candidate_output / "model.json.gz").read_bytes() == b"old candidate"
    assert (reference_output / "model.json.gz").read_bytes() == b"old reference"
    assert (candidate_stage / "model.json.gz").read_bytes() == b"new candidate"
    assert (reference_stage / "model.json.gz").read_bytes() == b"new reference"


def test_directory_publication_replaces_both_destinations(tmp_path: Path):
    candidate_stage = tmp_path / "candidate.stage"
    reference_stage = tmp_path / "reference.stage"
    candidate_output = tmp_path / "candidate"
    reference_output = tmp_path / "reference"
    for directory, payload in (
        (candidate_stage, b"new candidate"),
        (reference_stage, b"new reference"),
        (candidate_output, b"old candidate"),
        (reference_output, b"old reference"),
    ):
        directory.mkdir()
        (directory / "model.json.gz").write_bytes(payload)

    _publish_directory_set(
        [
            (candidate_stage, candidate_output),
            (reference_stage, reference_output),
        ]
    )

    assert (candidate_output / "model.json.gz").read_bytes() == b"new candidate"
    assert (reference_output / "model.json.gz").read_bytes() == b"new reference"
    assert not candidate_stage.exists()
    assert not reference_stage.exists()
