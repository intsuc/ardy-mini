# SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from ardy.model.llm2vec.llm2vec import LLM2Vec
from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder


def test_encode_serial_mode_honors_explicit_device_with_multiple_gpus() -> None:
    encoder = object.__new__(LLM2Vec)
    nn.Module.__init__(encoder)
    encoder._convert_to_str = MethodType(
        lambda _self, _instruction, text: text,
        encoder,
    )
    encoder.to = Mock(return_value=encoder)
    encoder._encode = Mock(
        side_effect=lambda batch, **_kwargs: torch.tensor(
            [[float(len(text))] for text in batch],
        )
    )

    with (
        patch("ardy.model.llm2vec.llm2vec.torch.cuda.device_count", return_value=2),
        patch(
            "ardy.model.llm2vec.llm2vec.mp.get_context",
            side_effect=AssertionError("multiprocessing must not be used"),
        ),
    ):
        embeddings = encoder.encode(
            ["short", "a longer prompt"],
            batch_size=1,
            show_progress_bar=False,
            device="cuda:1",
            use_multiprocessing=False,
        )

    encoder.to.assert_called_once_with("cuda:1")
    assert encoder._encode.call_count == 2
    assert all(call.kwargs["device"] == "cuda:1" for call in encoder._encode.call_args_list)
    torch.testing.assert_close(
        embeddings,
        torch.tensor([[5.0], [15.0]], dtype=torch.float32),
    )


def test_wrapper_requests_serial_batch_size_one() -> None:
    encoder = object.__new__(LLM2VecEncoder)
    encoder.llm_dim = 4
    encoder._device = "cpu"
    encoder.model = Mock()
    encoder.model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

    encoded, lengths = encoder(["walk forward"])

    encoder.model.encode.assert_called_once_with(
        ["walk forward"],
        batch_size=1,
        show_progress_bar=False,
        device="cpu",
        use_multiprocessing=False,
    )
    assert encoded.shape == (1, 1, 4)
    assert lengths == [1]


def test_explicit_foundation_loads_mntp_then_supervised_adapter() -> None:
    logical_foundation_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    events: list[str] = []

    class FakeModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config = config

    class FakeLlamaConfig:
        pass

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        foundation_path = root / "foundation"
        base_path = root / "mntp"
        supervised_path = root / "supervised"
        foundation_path.mkdir()
        base_path.mkdir()
        supervised_path.mkdir()
        (base_path / "config.json").write_text(
            json.dumps({"_name_or_path": logical_foundation_name}),
            encoding="utf-8",
        )

        foundation_config = FakeLlamaConfig()
        foundation_config._name_or_path = str(foundation_path)
        foundation_model = FakeModel(foundation_config)
        merged_mntp_model = FakeModel(foundation_config)
        supervised_model = FakeModel(foundation_config)

        model_class = SimpleNamespace()

        def load_foundation(path: str, **kwargs: object) -> FakeModel:
            events.append("foundation")
            assert path == str(foundation_path)
            assert kwargs == {
                "torch_dtype": torch.bfloat16,
                "cache_dir": "/cache",
            }
            return foundation_model

        model_class.from_pretrained = Mock(side_effect=load_foundation)

        mntp_adapter = Mock()

        def merge_mntp() -> FakeModel:
            events.append("merge-mntp")
            return merged_mntp_model

        mntp_adapter.merge_and_unload.side_effect = merge_mntp

        def load_adapter(
            model: nn.Module,
            path: str,
            **kwargs: object,
        ) -> object:
            assert kwargs == {"cache_dir": "/cache"}
            if path == str(base_path):
                events.append("mntp")
                assert model is foundation_model
                return mntp_adapter
            if path == str(supervised_path):
                events.append("supervised")
                assert model is merged_mntp_model
                return supervised_model
            raise AssertionError(f"unexpected adapter path: {path}")

        tokenizer = SimpleNamespace(
            eos_token="<eos>",
            pad_token=None,
            padding_side=None,
        )
        with (
            patch(
                "ardy.model.llm2vec.llm2vec.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
            patch(
                "ardy.model.llm2vec.llm2vec.AutoConfig.from_pretrained",
                return_value=foundation_config,
            ) as load_config,
            patch.object(
                LLM2Vec,
                "_get_model_class",
                return_value=model_class,
            ),
            patch(
                "ardy.model.llm2vec.llm2vec.PeftModel.from_pretrained",
                side_effect=load_adapter,
            ) as load_peft,
        ):
            encoder = LLM2Vec.from_pretrained(
                base_model_name_or_path=str(base_path),
                peft_model_name_or_path=str(supervised_path),
                foundation_model_name_or_path=str(foundation_path),
                torch_dtype=torch.bfloat16,
                cache_dir="/cache",
            )

    assert events == ["foundation", "mntp", "merge-mntp", "supervised"]
    assert encoder.model is supervised_model
    assert encoder.model.config._name_or_path == logical_foundation_name
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
    load_tokenizer.assert_called_once_with(str(base_path), cache_dir="/cache")
    load_config.assert_called_once_with(str(foundation_path), cache_dir="/cache")
    assert load_peft.call_count == 2
