# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for text-condition runtime, server, and export integration."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from ardy.model.ardy_model import Ardy
from ardy.model.text_encoder_api import (
    DIRECT_CONDITION_SCHEMA,
    DIRECT_ROOT_BODY_DIM,
    LEGACY_LLM2VEC_DIM,
    TextEncoderAPI,
    build_text_encoder_metadata,
    embedding_to_numpy,
)
from ardy.model.trt import TRTCFGDenoiser
from scripts.export_onnx import (
    engine_path,
    make_denoiser_dummy_inputs,
    verify_denoiser,
)


class ExportTextDimensionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = SimpleNamespace(
            num_frames_per_token=4,
            nframe_root_dim=20,
            latent_embedding_dim=128,
            motion_rep=SimpleNamespace(motion_rep_dim=256),
            llm_shape=[1, LEGACY_LLM2VEC_DIM],
        )

    def test_dummy_inputs_support_legacy_and_direct_widths(self) -> None:
        legacy = make_denoiser_dummy_inputs(
            self.wrapper,
            num_tokens=3,
            num_text_tokens=1,
            device="cpu",
        )
        direct = make_denoiser_dummy_inputs(
            self.wrapper,
            num_tokens=3,
            num_text_tokens=1,
            device="cpu",
            text_feature_dim=DIRECT_ROOT_BODY_DIM,
        )
        self.assertEqual(legacy["text_feat"].shape, (1, 1, LEGACY_LLM2VEC_DIM))
        self.assertEqual(direct["text_feat"].shape, (1, 1, DIRECT_ROOT_BODY_DIM))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            make_denoiser_dummy_inputs(
                self.wrapper,
                device="cpu",
                text_feature_dim=0,
            )

    def test_direct_engine_name_never_collides_with_legacy_name(self) -> None:
        legacy = engine_path(
            "/tmp/engines",
            "denoiser",
            64,
            True,
            text_feature_dim=LEGACY_LLM2VEC_DIM,
        )
        direct = engine_path(
            "/tmp/engines",
            "denoiser",
            64,
            True,
            text_feature_dim=DIRECT_ROOT_BODY_DIM,
        )
        self.assertEqual(
            legacy,
            os.path.join("/tmp/engines", "denoiser_tok64_fp16.trt"),
        )
        self.assertEqual(
            direct,
            os.path.join(
                "/tmp/engines",
                "denoiser_text2048_direct_tok64_fp16.trt",
            ),
        )
        self.assertNotEqual(legacy, direct)
        self.assertEqual(
            engine_path(
                "/tmp/engines",
                "decoder",
                64,
                True,
                text_feature_dim=DIRECT_ROOT_BODY_DIM,
            ),
            os.path.join("/tmp/engines", "decoder_tok64_fp16.trt"),
        )

    def test_verifier_uses_the_requested_direct_width(self) -> None:
        class FakeDenoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.num_frames_per_token = 4
                self.nframe_root_dim = 20
                self.latent_embedding_dim = 128
                self.motion_rep = SimpleNamespace(motion_rep_dim=256)
                self.llm_shape = [1, LEGACY_LLM2VEC_DIM]
                self.observed_text_width = None

            def forward(self, *args):
                self.observed_text_width = int(args[12].shape[-1])
                return args[2]

        class FakeTRTWrapper:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __call__(self, *args):
                return args[2]

        denoiser = FakeDenoiser()
        with patch("ardy.model.trt.TRTCFGDenoiser", FakeTRTWrapper):
            difference = verify_denoiser(
                denoiser,
                "unused.trt",
                num_frames_per_token=4,
                num_tokens=3,
                num_text_tokens=1,
                text_feature_dim=DIRECT_ROOT_BODY_DIM,
            )
        self.assertEqual(difference, 0.0)
        self.assertEqual(denoiser.observed_text_width, DIRECT_ROOT_BODY_DIM)


class _CaptureDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        root_projection = SimpleNamespace(
            in_features=LEGACY_LLM2VEC_DIM,
            out_features=1024,
            projected_text_index=0,
        )
        body_projection = SimpleNamespace(
            in_features=LEGACY_LLM2VEC_DIM,
            out_features=1024,
            projected_text_index=1,
        )
        self.model = SimpleNamespace(
            num_frames_per_token=4,
            nframe_root_dim=20,
            latent_embedding_dim=128,
            motion_rep=SimpleNamespace(motion_rep_dim=256),
            llm_shape=[1, LEGACY_LLM2VEC_DIM],
            root_model=SimpleNamespace(embed_text=root_projection),
            body_model=SimpleNamespace(embed_text=body_projection),
        )
        self.last_text_shape: tuple[int, ...] | None = None

    def forward(self, **kwargs):
        self.last_text_shape = tuple(kwargs["text_feat"].shape)
        return torch.zeros_like(kwargs["x"])


def _tiny_ardy(output_dim: int | None) -> tuple[Ardy, _CaptureDenoiser]:
    model = object.__new__(Ardy)
    nn.Module.__init__(model)
    model.device = "cpu"
    denoiser = _CaptureDenoiser()
    model.denoiser = denoiser
    if output_dim is None:
        model.text_encoder = object()
    else:
        model.text_encoder = SimpleNamespace(output_dim=output_dim)
    return model, denoiser


class CompileWarmupDimensionTest(unittest.TestCase):
    def test_attached_direct_encoder_controls_warmup_shape(self) -> None:
        model, denoiser = _tiny_ardy(DIRECT_ROOT_BODY_DIM)
        self.assertEqual(model.resolve_text_feature_dim(), DIRECT_ROOT_BODY_DIM)
        model.warmup(num_tokens=2, num_text_tokens=1, num_iterations=1)
        self.assertEqual(denoiser.last_text_shape, (1, 1, DIRECT_ROOT_BODY_DIM))

    def test_legacy_fallback_and_invalid_width(self) -> None:
        legacy_model, _ = _tiny_ardy(None)
        self.assertEqual(
            legacy_model.resolve_text_feature_dim(),
            LEGACY_LLM2VEC_DIM,
        )
        invalid_model, _ = _tiny_ardy(123)
        with self.assertRaisesRegex(ValueError, "expected one of: 2048, 4096"):
            invalid_model.resolve_text_feature_dim()


class TensorRTTextShapeValidationTest(unittest.TestCase):
    def test_engine_width_mismatch_fails_before_inference(self) -> None:
        fake_engine = SimpleNamespace(input_shapes={"text_feat": (1, 1, DIRECT_ROOT_BODY_DIM)})
        denoiser = SimpleNamespace(
            nframe_root_dim=20,
            latent_embedding_dim=128,
            motion_rep=SimpleNamespace(motion_rep_dim=256),
        )
        with patch("ardy.model.trt.TRTEngine", return_value=fake_engine):
            wrapper = TRTCFGDenoiser(
                "unused.trt",
                denoiser,
                num_frames_per_token=4,
                num_tokens=3,
                num_text_tokens=1,
            )

        one = torch.ones(1, 1)
        with self.assertRaisesRegex(
            ValueError,
            "does not match this TensorRT engine's width 2048",
        ):
            wrapper(
                cfg_weight_text=torch.ones(1),
                cfg_weight_cstr=torch.ones(1),
                token_seq_t=torch.zeros(1, 1, 148),
                history_len=torch.ones(1, dtype=torch.long),
                generation_len=torch.ones(1, dtype=torch.long),
                future_len=torch.zeros(1, dtype=torch.long),
                history_mask=one.bool(),
                generation_mask=one.bool(),
                future_mask=one.bool(),
                history_token_mask=one.bool(),
                generation_token_mask=one.bool(),
                future_token_mask=one.bool(),
                text_feat=torch.zeros(1, 1, LEGACY_LLM2VEC_DIM),
                text_feat_pad_mask=one.bool(),
                timesteps=torch.zeros(1, dtype=torch.long),
            )


class ServerMetadataTest(unittest.TestCase):
    def test_metadata_and_bfloat16_wire_conversion(self) -> None:
        encoder = SimpleNamespace(
            output_dim=DIRECT_ROOT_BODY_DIM,
            compatible_ardy_models=["ARDY-Core-RP-20FPS-Horizon40"],
            cache_namespace="student-fingerprint",
        )
        metadata = build_text_encoder_metadata(encoder)
        self.assertEqual(metadata["output_dim"], DIRECT_ROOT_BODY_DIM)
        self.assertEqual(metadata["condition_schema"], DIRECT_CONDITION_SCHEMA)
        converted = embedding_to_numpy(torch.ones(1, DIRECT_ROOT_BODY_DIM, dtype=torch.bfloat16))
        self.assertEqual(converted.dtype, np.float32)

    def test_direct_metadata_requires_compatibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "must disclose"):
            build_text_encoder_metadata(
                SimpleNamespace(
                    output_dim=DIRECT_ROOT_BODY_DIM,
                    compatible_ardy_models=[],
                    cache_namespace="unsafe",
                )
            )


class TextEncoderAPICompatibilityTest(unittest.TestCase):
    @staticmethod
    def _client_with_response(response):
        client = Mock()
        client.predict.return_value = response
        return client

    def test_metadata_direct_response_sets_namespace_and_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_path = Path(temp_dir) / "direct.npy"
            np.save(
                embedding_path,
                np.ones((1, DIRECT_ROOT_BODY_DIM), dtype=np.float32),
            )
            metadata = {
                "schema_version": 1,
                "output_dim": DIRECT_ROOT_BODY_DIM,
                "condition_schema": DIRECT_CONDITION_SCHEMA,
                "compatible_ardy_models": ["ARDY-Core-RP-20FPS-Horizon40"],
                "cache_namespace": "student-fingerprint",
            }
            client = self._client_with_response(
                (
                    {"value": str(embedding_path)},
                    None,
                    None,
                    metadata,
                )
            )
            with patch("ardy.model.text_encoder_api.Client", return_value=client):
                encoder = TextEncoderAPI("http://encoder.example/")
            tensor, lengths = encoder(["walk forward"])
            self.assertEqual(tensor.shape, (1, 1, DIRECT_ROOT_BODY_DIM))
            self.assertEqual(lengths, [1])
            self.assertIn("student-fingerprint", encoder.cache_namespace)
            encoder.assert_compatible("ARDY-Core-RP-20FPS-Horizon40")
            with self.assertRaisesRegex(ValueError, "not 'ARDY-G1"):
                encoder.assert_compatible("ARDY-G1-RP-25FPS-Horizon52")

    def test_metadata_free_legacy_4096_server_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_path = Path(temp_dir) / "legacy.npy"
            np.save(
                embedding_path,
                np.ones((1, LEGACY_LLM2VEC_DIM), dtype=np.float32),
            )
            client = self._client_with_response(({"value": str(embedding_path)}, None, None))
            with patch("ardy.model.text_encoder_api.Client", return_value=client):
                encoder = TextEncoderAPI("http://legacy.example/")
            encoder.ensure_metadata()
            self.assertEqual(encoder.output_dim, LEGACY_LLM2VEC_DIM)
            self.assertIn("legacy-server", encoder.cache_namespace)
            encoder.assert_compatible("ARDY-G1-RP-25FPS-Horizon52")

    def test_metadata_free_2048_server_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_path = Path(temp_dir) / "anonymous-direct.npy"
            np.save(
                embedding_path,
                np.ones((1, DIRECT_ROOT_BODY_DIM), dtype=np.float32),
            )
            client = self._client_with_response(({"value": str(embedding_path)}, None, None))
            with patch("ardy.model.text_encoder_api.Client", return_value=client):
                encoder = TextEncoderAPI("http://unsafe.example/")
            with self.assertRaisesRegex(
                RuntimeError,
                "without schema metadata",
            ):
                encoder.ensure_metadata()


class _FakeLocalEncoder:
    def __init__(self) -> None:
        self.to_calls: list[tuple[str, torch.dtype]] = []

    def to(self, device=None, dtype=None):
        self.to_calls.append((str(device), dtype))
        return self


class LoadTextEncoderDeviceTest(unittest.TestCase):
    def test_explicit_device_is_passed_to_constructor(self) -> None:
        load_module = importlib.import_module("ardy.model.load_model")
        fake = _FakeLocalEncoder()
        captured = {}

        def instantiate(conf):
            captured.update(conf)
            return fake

        conf = {
            "_target_": "tests.fake.Encoder",
            "device": "auto",
            "dtype": "bfloat16",
        }
        with (
            patch.object(
                load_module,
                "_select_text_encoder_conf",
                return_value=(conf, None),
            ),
            patch.object(
                load_module,
                "instantiate_from_dict",
                side_effect=instantiate,
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            result = load_module.load_text_encoder(mode="local", device="cpu")

        self.assertIs(result, fake)
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(fake.to_calls, [("cpu", torch.bfloat16)])

    def test_environment_device_is_resolved_before_constructor(self) -> None:
        load_module = importlib.import_module("ardy.model.load_model")
        fake = _FakeLocalEncoder()
        captured = {}

        def instantiate(conf):
            captured.update(conf)
            return fake

        conf = {
            "_target_": "tests.fake.Encoder",
            "device": "auto",
            "dtype": "bfloat16",
        }
        with (
            patch.dict(os.environ, {"TEXT_ENCODER_DEVICE": "cpu"}),
            patch.object(
                load_module,
                "_select_text_encoder_conf",
                return_value=(conf, None),
            ),
            patch.object(
                load_module,
                "instantiate_from_dict",
                side_effect=instantiate,
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            load_module.load_text_encoder(mode="local", device=None)

        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(fake.to_calls, [("cpu", torch.bfloat16)])


if __name__ == "__main__":
    unittest.main()
