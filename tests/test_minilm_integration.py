# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""Offline integration tests for the distilled MiniLM conditioning path."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from ardy.model import minilm_encoder
from ardy.model.backbone import DualConditionTextProjection
from ardy.model.minilm_encoder import (
    ARDY_CONDITION_DIM,
    ARTIFACT_CONFIG,
    ARTIFACT_FORMAT_VERSION,
    BACKBONE_DIR,
    HEADS_FILE,
    LEGACY_ARTIFACT_FORMAT_VERSION,
    MiniLMArdyEncoder,
    MotionConditionStudent,
    pool_token_embeddings,
)


class _TinyBackbone(nn.Module):
    """Small saveable backbone used in place of a downloaded MiniLM model."""

    CONFIG_FILE = "tiny_backbone_config.json"
    WEIGHTS_FILE = "tiny_backbone.pt"

    def __init__(self, hidden_size: int = 8, vocab_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, output_dir: str | Path, safe_serialization: bool = True) -> None:
        self.last_safe_serialization = safe_serialization
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / self.CONFIG_FILE).write_text(
            json.dumps(
                {
                    "hidden_size": self.config.hidden_size,
                    "vocab_size": self.vocab_size,
                }
            ),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output_path / self.WEIGHTS_FILE)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> _TinyBackbone:
        model_path = Path(model_dir)
        config = json.loads((model_path / cls.CONFIG_FILE).read_text(encoding="utf-8"))
        model = cls(**config)
        model.load_state_dict(
            torch.load(
                model_path / cls.WEIGHTS_FILE,
                map_location="cpu",
                weights_only=True,
            )
        )
        return model


class _TinyTokenizer:
    """Deterministic tokenizer with the subset of the HF tokenizer contract used here."""

    CONFIG_FILE = "tiny_tokenizer_config.json"

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / self.CONFIG_FILE).write_text(
            json.dumps({"vocab_size": self.vocab_size}),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> _TinyTokenizer:
        config = json.loads((Path(model_dir) / cls.CONFIG_FILE).read_text(encoding="utf-8"))
        return cls(**config)

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        if not padding or not truncation or return_tensors != "pt":
            raise AssertionError("The inference wrapper must request padded PyTorch tensors")

        token_rows = []
        for text in texts:
            words = text.split() or [""]
            token_rows.append([1 + (sum(word.encode("utf-8")) % (self.vocab_size - 1)) for word in words[:max_length]])

        width = max(len(row) for row in token_rows)
        input_ids = torch.zeros((len(token_rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(token_rows):
            input_ids[row_index, : len(row)] = torch.tensor(row)
            attention_mask[row_index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DualConditionTextProjectionTest(unittest.TestCase):
    def test_direct_root_and_body_paths_match_legacy_linear_outputs(self) -> None:
        torch.manual_seed(7)
        input_dim = 7
        condition_dim = 4
        embeddings = torch.randn(3, 2, input_dim)

        legacy_root = nn.Linear(input_dim, condition_dim)
        legacy_body = nn.Linear(input_dim, condition_dim)
        root_projection = DualConditionTextProjection(input_dim, condition_dim, 0)
        body_projection = DualConditionTextProjection(input_dim, condition_dim, 1)
        root_projection.load_state_dict(legacy_root.state_dict())
        body_projection.load_state_dict(legacy_body.state_dict())

        direct_conditions = torch.cat(
            (
                F.linear(embeddings, legacy_root.weight, bias=None),
                F.linear(embeddings, legacy_body.weight, bias=None),
            ),
            dim=-1,
        )

        torch.testing.assert_close(root_projection(embeddings), legacy_root(embeddings))
        torch.testing.assert_close(body_projection(embeddings), legacy_body(embeddings))
        torch.testing.assert_close(root_projection(direct_conditions), legacy_root(embeddings))
        torch.testing.assert_close(body_projection(direct_conditions), legacy_body(embeddings))
        self.assertEqual(set(root_projection.state_dict()), {"weight", "bias"})

    def test_zero_direct_condition_preserves_checkpoint_bias(self) -> None:
        for projected_text_index in (0, 1):
            with self.subTest(projected_text_index=projected_text_index):
                projection = DualConditionTextProjection(7, 4, projected_text_index)
                with torch.no_grad():
                    projection.bias.copy_(torch.tensor([0.2, -0.3, 0.5, 1.1]))

                output = projection(torch.zeros(2, 3, 8))
                expected = projection.bias.expand_as(output)
                torch.testing.assert_close(output, expected)

    def test_invalid_dimensions_and_projection_index_are_rejected(self) -> None:
        projection = DualConditionTextProjection(7, 4, 0)
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported text condition dimension 6.*legacy 7.*dual-condition 8",
        ):
            projection(torch.zeros(2, 6))

        legacy_only = DualConditionTextProjection(7, 4, None)
        with self.assertRaisesRegex(ValueError, "Unsupported text condition dimension 8"):
            legacy_only(torch.zeros(2, 8))

        with self.assertRaisesRegex(ValueError, "must be None, 0, or 1"):
            DualConditionTextProjection(7, 4, 2)


class MotionConditionStudentArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.tokenizer = _TinyTokenizer()
        self.student = MotionConditionStudent(
            backbone=_TinyBackbone(),
            adapter_dim=6,
            condition_dim=ARDY_CONDITION_DIM,
            normalize_embedding=True,
        ).eval()
        self.prompts = ["walk forward", "turn left quickly"]
        self.base_model = "local/tiny-minilm"
        self.compatible_models = ["test-motion-model"]

    @staticmethod
    def _loader_patches():
        return (
            patch.object(
                minilm_encoder.AutoModel,
                "from_pretrained",
                side_effect=_TinyBackbone.from_pretrained,
            ),
            patch.object(
                minilm_encoder.AutoTokenizer,
                "from_pretrained",
                side_effect=_TinyTokenizer.from_pretrained,
            ),
        )

    def _save_artifact(
        self,
        output_dir: str | Path,
        *,
        student: MotionConditionStudent | None = None,
        metadata: dict | None = None,
    ) -> Path:
        return (student or self.student).save_artifact(
            output_dir,
            self.tokenizer,
            metadata={} if metadata is None else metadata,
            base_model_name_or_path=self.base_model,
            compatible_ardy_models=self.compatible_models,
            max_length=16,
        )

    @staticmethod
    def _rewrite_as_legacy_v1(artifact_path: Path, *, omit_pooling: bool = False) -> dict:
        config_path = artifact_path / ARTIFACT_CONFIG
        current = json.loads(config_path.read_text(encoding="utf-8"))
        legacy = {
            key: value
            for key, value in current.items()
            if key not in {"artifact_files", "artifact_fingerprint", "metadata"}
        }
        legacy["format_version"] = LEGACY_ARTIFACT_FORMAT_VERSION
        legacy.update(current["metadata"])
        if omit_pooling:
            legacy.pop("pooling_mode")
            legacy.pop("pooled_dim")
        config_bytes = json.dumps(legacy, indent=2, sort_keys=True).encode("utf-8")
        heads_hash = hashlib.sha256((artifact_path / HEADS_FILE).read_bytes()).hexdigest()
        legacy["artifact_fingerprint"] = hashlib.sha256(config_bytes + heads_hash.encode("ascii")).hexdigest()
        config_path.write_text(
            json.dumps(legacy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return legacy

    def test_artifact_round_trip_and_model_compatibility_check(self) -> None:
        encoded = self.tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        with torch.inference_mode():
            expected = self.student(**encoded)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(
                temp_dir,
                metadata={
                    "training_prompts": len(self.prompts),
                    # Canonical-looking metadata remains namespaced and cannot
                    # override inference-critical fields.
                    "format_version": 999,
                    "base_model": "wrong/model",
                },
            )

            self.assertTrue((artifact_path / BACKBONE_DIR).is_dir())
            self.assertTrue((artifact_path / HEADS_FILE).is_file())
            config = json.loads((artifact_path / ARTIFACT_CONFIG).read_text(encoding="utf-8"))
            self.assertEqual(config["format_version"], ARTIFACT_FORMAT_VERSION)
            self.assertEqual(config["hidden_dim"], 8)
            self.assertEqual(config["pooled_dim"], 8)
            self.assertEqual(config["pooling_mode"], "mean")
            self.assertEqual(config["condition_dim"], ARDY_CONDITION_DIM)
            self.assertEqual(config["output_dim"], 2 * ARDY_CONDITION_DIM)
            self.assertEqual(config["base_model"], self.base_model)
            self.assertEqual(config["compatible_ardy_models"], self.compatible_models)
            self.assertEqual(config["metadata"]["format_version"], 999)
            self.assertEqual(config["metadata"]["base_model"], "wrong/model")
            self.assertIn(HEADS_FILE, config["artifact_files"])
            self.assertIn(
                f"{BACKBONE_DIR}/{_TinyBackbone.WEIGHTS_FILE}",
                config["artifact_files"],
            )
            self.assertIn(
                f"{BACKBONE_DIR}/{_TinyTokenizer.CONFIG_FILE}",
                config["artifact_files"],
            )
            self.assertEqual(len(config["artifact_fingerprint"]), 64)

            model_patch, tokenizer_patch = self._loader_patches()
            with model_patch, tokenizer_patch:
                restored, restored_tokenizer, restored_config = MotionConditionStudent.from_artifact(artifact_path)
                restored.eval()
                restored_encoded = restored_tokenizer(
                    self.prompts,
                    padding=True,
                    truncation=True,
                    max_length=16,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    actual = restored(**restored_encoded)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(restored_config, config)

                encoder = MiniLMArdyEncoder(
                    str(artifact_path),
                    dtype="float32",
                    device="cpu",
                    expected_ardy_model="test-motion-model",
                )
                conditions, lengths = encoder(self.prompts)
                torch.testing.assert_close(conditions[:, 0, :], expected)
                self.assertFalse(torch.is_inference(conditions))
                self.assertEqual(
                    conditions.shape,
                    (2, 1, 2 * ARDY_CONDITION_DIM),
                )
                self.assertEqual(lengths, [1, 1])
                self.assertIn(config["artifact_fingerprint"], encoder.cache_namespace)

                with self.assertRaisesRegex(
                    ValueError,
                    r"trained for .*test-motion-model.*not 'another-model'",
                ):
                    encoder.assert_compatible("another-model")

    def test_richer_pooling_is_mask_aware_and_round_trips(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[5.0, 6.0], [200.0, 200.0], [300.0, 300.0]],
            ]
        ).requires_grad_()
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        pooled = pool_token_embeddings(hidden, attention_mask, "mean_cls_max_std")
        expected = torch.tensor(
            [
                [2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0],
                [5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 0.0, 0.0],
            ]
        )
        torch.testing.assert_close(pooled, expected)
        pooled.square().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

        student = MotionConditionStudent(
            backbone=_TinyBackbone(),
            adapter_dim=6,
            condition_dim=ARDY_CONDITION_DIM,
            normalize_embedding=True,
            pooling_mode="mean_cls_max_std",
        ).eval()
        self.assertEqual(student.pooled_dim, 32)
        encoded = self.tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        with torch.inference_mode():
            expected_conditions = student(**encoded)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(temp_dir, student=student)
            config = json.loads((artifact_path / ARTIFACT_CONFIG).read_text(encoding="utf-8"))
            self.assertEqual(config["pooling_mode"], "mean_cls_max_std")
            self.assertEqual(config["pooled_dim"], 32)

            model_patch, tokenizer_patch = self._loader_patches()
            with model_patch, tokenizer_patch:
                restored, _, _ = MotionConditionStudent.from_artifact(artifact_path)
            restored.eval()
            with torch.inference_mode():
                actual_conditions = restored(**encoded)
            torch.testing.assert_close(actual_conditions, expected_conditions)

    def test_legacy_artifact_without_pooling_fields_defaults_to_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(temp_dir)
            self._rewrite_as_legacy_v1(artifact_path, omit_pooling=True)

            model_patch, tokenizer_patch = self._loader_patches()
            with model_patch, tokenizer_patch:
                restored, _, restored_config = MotionConditionStudent.from_artifact(artifact_path)
            self.assertEqual(restored.pooling_mode, "mean")
            self.assertEqual(restored.pooled_dim, 8)
            self.assertNotIn("pooling_mode", restored_config)
            self.assertEqual(
                restored_config["format_version"],
                LEGACY_ARTIFACT_FORMAT_VERSION,
            )

    def test_v2_manifest_detects_backbone_tokenizer_and_head_tampering(self) -> None:
        payloads = (
            f"{BACKBONE_DIR}/{_TinyBackbone.WEIGHTS_FILE}",
            f"{BACKBONE_DIR}/{_TinyTokenizer.CONFIG_FILE}",
            HEADS_FILE,
        )
        for relative_path in payloads:
            with self.subTest(relative_path=relative_path), tempfile.TemporaryDirectory() as temp_dir:
                artifact_path = self._save_artifact(temp_dir)
                with (artifact_path / relative_path).open("ab") as output_file:
                    output_file.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    MotionConditionStudent.from_artifact(artifact_path)

    def test_v2_fingerprint_detects_config_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(
                temp_dir,
                metadata={"training_prompts": 2},
            )
            config_path = artifact_path / ARTIFACT_CONFIG
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["metadata"]["training_prompts"] = 3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                MotionConditionStudent.from_artifact(artifact_path)

    def test_invalid_compatibility_and_dimensions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(temp_dir)
            config_path = artifact_path / ARTIFACT_CONFIG
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["compatible_ardy_models"] = []
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty list"):
                MotionConditionStudent.from_artifact(artifact_path)

        invalid_student = MotionConditionStudent(
            backbone=_TinyBackbone(),
            adapter_dim=6,
            condition_dim=5,
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(ValueError, "condition_dim=1024"),
        ):
            self._save_artifact(temp_dir, student=invalid_student)

    def test_inconsistent_artifact_config_fields_are_rejected(self) -> None:
        mutations = (
            ("condition_dim", 1000, "condition_dim"),
            ("output_dim", 2047, "output_dim"),
            ("max_length", 0, "max_length"),
            ("hidden_dim", 7, "pooled_dim"),
            ("pooled_dim", 7, "pooled_dim"),
            ("pooling_mode", "invalid", "Unsupported pooling mode"),
        )
        for field, value, message in mutations:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                artifact_path = self._save_artifact(temp_dir)
                config_path = artifact_path / ARTIFACT_CONFIG
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config[field] = value
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MotionConditionStudent.from_artifact(artifact_path)

    def test_configured_hidden_dim_must_match_saved_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(temp_dir)
            config_path = artifact_path / ARTIFACT_CONFIG
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["hidden_dim"] = 9
            config["pooled_dim"] = 9
            unsigned = dict(config)
            unsigned.pop("artifact_fingerprint")
            encoded = json.dumps(
                unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            config["artifact_fingerprint"] = hashlib.sha256(encoded).hexdigest()
            config_path.write_text(json.dumps(config), encoding="utf-8")

            model_patch, tokenizer_patch = self._loader_patches()
            with (
                model_patch,
                tokenizer_patch,
                self.assertRaisesRegex(ValueError, "does not match.*hidden size"),
            ):
                MotionConditionStudent.from_artifact(artifact_path)

    def test_nonpositive_runtime_max_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = self._save_artifact(temp_dir)
            model_patch, tokenizer_patch = self._loader_patches()
            with model_patch, tokenizer_patch, self.assertRaisesRegex(ValueError, "positive integer"):
                MiniLMArdyEncoder(
                    str(artifact_path),
                    dtype="float32",
                    device="cpu",
                    max_length=0,
                )

    def test_invalid_pooling_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported pooling mode"):
            MotionConditionStudent(
                backbone=_TinyBackbone(),
                pooling_mode="not-a-pooling-mode",
            )
        with self.assertRaisesRegex(ValueError, "Unsupported pooling mode"):
            pool_token_embeddings(
                torch.zeros(1, 1, 2),
                torch.ones(1, 1, dtype=torch.long),
                "not-a-pooling-mode",
            )

    def test_unsupported_artifact_format_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ARTIFACT_CONFIG
            config_path.write_text(
                json.dumps({"format_version": ARTIFACT_FORMAT_VERSION + 1}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                rf"Unsupported MiniLM artifact format {ARTIFACT_FORMAT_VERSION + 1}",
            ):
                MotionConditionStudent.from_artifact(temp_dir)


if __name__ == "__main__":
    unittest.main()
