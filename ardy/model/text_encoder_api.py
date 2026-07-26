# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by intsuc in 2026: added distilled MiniLM text-encoder support.
"""Remote text encoder API client (Gradio) for motion generation."""

import hashlib
import json
import logging
from typing import Any

import numpy as np
import torch
from gradio_client import Client

# Suppress the [httpx] logs (GET requests)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress internal gradio_client logs
logging.getLogger("gradio_client").setLevel(logging.WARNING)

TEXT_ENCODER_METADATA_VERSION = 1
LEGACY_LLM2VEC_DIM = 4096
DIRECT_ROOT_BODY_DIM = 2048
LEGACY_CONDITION_SCHEMA = "llm2vec-embedding-v1"
DIRECT_CONDITION_SCHEMA = "ardy-root-body-direct-v1"


def _url_fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def embedding_to_numpy(embedding: torch.Tensor) -> np.ndarray:
    """Convert encoder output to the server's portable float32 wire format."""

    return embedding.detach().to(device="cpu", dtype=torch.float32).numpy()


def build_text_encoder_metadata(text_encoder) -> dict[str, Any]:
    """Build the schema returned alongside a server-side embedding.

    A 2048-dimensional response is checkpoint-specific and is therefore only
    safe when the server discloses both its direct-condition schema and a
    non-empty compatibility list.
    """

    output_dim = getattr(text_encoder, "output_dim", None)
    if output_dim is None:
        output_dim = getattr(text_encoder, "llm_dim", None)
    try:
        output_dim = int(output_dim)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Text encoder must expose an integer output_dim, got {output_dim!r}") from error

    compatible = getattr(text_encoder, "compatible_ardy_models", None)
    artifact_config = getattr(text_encoder, "artifact_config", None)
    if compatible is None and isinstance(artifact_config, dict):
        compatible = artifact_config.get("compatible_ardy_models")
    if compatible is not None:
        if not isinstance(compatible, (list, tuple)) or not all(isinstance(name, str) and name for name in compatible):
            raise ValueError("compatible_ardy_models must be null or a list of non-empty strings")
        compatible = list(compatible)

    if output_dim == LEGACY_LLM2VEC_DIM:
        condition_schema = LEGACY_CONDITION_SCHEMA
    elif output_dim == DIRECT_ROOT_BODY_DIM:
        condition_schema = DIRECT_CONDITION_SCHEMA
        if not compatible:
            raise ValueError(
                "A 2048-dimensional direct-condition encoder must disclose at least one compatible ARDY model."
            )
    else:
        raise ValueError(
            f"Unsupported text encoder output_dim {output_dim}; expected "
            f"{LEGACY_LLM2VEC_DIM} or {DIRECT_ROOT_BODY_DIM}."
        )

    namespace = getattr(text_encoder, "cache_namespace", None)
    if not isinstance(namespace, str) or not namespace:
        namespace_payload = f"{type(text_encoder).__module__}.{type(text_encoder).__qualname__}:{output_dim}"
        namespace = f"ardy-text-encoder-v1:{hashlib.sha256(namespace_payload.encode('utf-8')).hexdigest()}"

    return {
        "schema_version": TEXT_ENCODER_METADATA_VERSION,
        "output_dim": output_dim,
        "condition_schema": condition_schema,
        "compatible_ardy_models": compatible,
        "cache_namespace": namespace,
    }


class TextEncoderAPI:
    """Text encoder API client for motion generation."""

    def __init__(self, url: str):
        self.client = Client(url, verbose=False)
        self.url = url
        self.device = "cpu"
        self.dtype = torch.float
        self.output_dim: int | None = None
        self.compatible_ardy_models: list[str] | None = None
        self.condition_schema: str | None = None
        self.server_metadata: dict[str, Any] | None = None
        self._metadata_established = False
        self._url_fingerprint = _url_fingerprint(url)
        self.cache_namespace = f"ardy-text-api-v1:{self._url_fingerprint}:unprobed"

    def _create_np_random_name(self):
        import uuid

        return str(uuid.uuid4()) + ".npy"

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    @staticmethod
    def _response_metadata(result) -> dict[str, Any] | None:
        if not isinstance(result, (list, tuple)) or len(result) < 4:
            return None
        metadata = result[3]
        if isinstance(metadata, dict) and set(metadata) == {"value"}:
            metadata = metadata["value"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as error:
                raise RuntimeError("Text encoder server returned invalid metadata JSON") from error
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise TypeError(f"Text encoder server metadata must be an object, got {type(metadata).__name__}.")
        return metadata

    @staticmethod
    def _download_path(result) -> str:
        if not isinstance(result, (list, tuple)) or not result:
            raise RuntimeError("Text encoder server returned an empty response")
        download = result[0]
        if isinstance(download, str):
            return download
        if isinstance(download, dict):
            value = download.get("value")
            if isinstance(value, str):
                return value
        raise RuntimeError("Text encoder server response does not contain a downloadable embedding")

    def _establish_metadata(
        self,
        metadata: dict[str, Any] | None,
        observed_dim: int,
    ) -> None:
        if metadata is None:
            # Servers predating the distilled encoder returned no metadata.
            # Preserve those 4096-dimensional LLM2Vec services, but never infer
            # that an anonymous 2048 vector has checkpoint-specific semantics.
            if observed_dim != LEGACY_LLM2VEC_DIM:
                raise RuntimeError(
                    "A legacy text encoder server returned "
                    f"{observed_dim} features without schema metadata. "
                    "Only metadata-free 4096-dimensional LLM2Vec servers are "
                    "accepted; upgrade the server for direct root/body conditions."
                )
            normalized = {
                "schema_version": 0,
                "output_dim": LEGACY_LLM2VEC_DIM,
                "condition_schema": LEGACY_CONDITION_SCHEMA,
                "compatible_ardy_models": None,
                "cache_namespace": (f"legacy-server-{self._url_fingerprint}:{LEGACY_LLM2VEC_DIM}"),
            }
        else:
            try:
                schema_version = int(metadata["schema_version"])
                output_dim = int(metadata["output_dim"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Text encoder server metadata requires integer schema_version and output_dim fields"
                ) from error
            if schema_version != TEXT_ENCODER_METADATA_VERSION:
                raise RuntimeError(
                    f"Unsupported text encoder metadata version {schema_version}; "
                    f"expected {TEXT_ENCODER_METADATA_VERSION}."
                )
            if output_dim != observed_dim:
                raise RuntimeError(
                    f"Text encoder metadata declares {output_dim} features, but "
                    f"the downloaded embedding has {observed_dim}."
                )

            condition_schema = metadata.get("condition_schema")
            expected_schema = (
                LEGACY_CONDITION_SCHEMA
                if output_dim == LEGACY_LLM2VEC_DIM
                else DIRECT_CONDITION_SCHEMA
                if output_dim == DIRECT_ROOT_BODY_DIM
                else None
            )
            if expected_schema is None or condition_schema != expected_schema:
                raise RuntimeError(
                    f"Unsupported text condition schema {condition_schema!r} for output_dim {output_dim}."
                )

            compatible = metadata.get("compatible_ardy_models")
            if compatible is not None and (
                not isinstance(compatible, list) or not all(isinstance(name, str) and name for name in compatible)
            ):
                raise RuntimeError(
                    "Text encoder server compatible_ardy_models must be null or a list of non-empty strings"
                )
            if output_dim == DIRECT_ROOT_BODY_DIM and not compatible:
                raise RuntimeError(
                    "A direct root/body text encoder must disclose a non-empty compatible_ardy_models list."
                )

            namespace = metadata.get("cache_namespace")
            if not isinstance(namespace, str) or not namespace:
                raise RuntimeError("Text encoder server metadata requires a non-empty cache_namespace")
            normalized = {
                "schema_version": schema_version,
                "output_dim": output_dim,
                "condition_schema": condition_schema,
                "compatible_ardy_models": compatible,
                "cache_namespace": namespace,
            }

        if self._metadata_established:
            if normalized != self.server_metadata:
                raise RuntimeError(
                    "Text encoder server identity changed while the client was "
                    "running; restart with a fresh embedding cache."
                )
            return

        self.server_metadata = normalized
        self.output_dim = int(normalized["output_dim"])
        self.condition_schema = str(normalized["condition_schema"])
        compatible = normalized["compatible_ardy_models"]
        self.compatible_ardy_models = None if compatible is None else list(compatible)
        server_namespace = str(normalized["cache_namespace"])
        self.cache_namespace = f"ardy-text-api-v1:{self._url_fingerprint}:{server_namespace}"
        self._metadata_established = True

    def ensure_metadata(self) -> None:
        """Probe the service once so compatibility and cache identity are known."""

        if not self._metadata_established:
            self(["healthcheck"])

    def assert_compatible(self, ardy_model_name: str) -> None:
        self.ensure_metadata()
        if self.compatible_ardy_models is not None and ardy_model_name not in self.compatible_ardy_models:
            raise ValueError(
                f"Remote text encoder is trained for {self.compatible_ardy_models}, not {ardy_model_name!r}."
            )

    def __call__(self, texts):
        """Encode text prompts into tensors.

        Args:
            texts (str | list[str]): text prompts to encode

        Returns:
            tuple[torch.Tensor, list[int]]: encoded text tensors and their lengths
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("texts must contain at least one prompt")

        tensors = []
        lengths = []
        for text in texts:
            filename = self._create_np_random_name()

            result = self.client.predict(
                text=text,
                filename=filename,
                api_name="/DemoWrapper",
            )
            path = self._download_path(result)
            tensor = np.load(path)
            if tensor.ndim != 2:
                raise RuntimeError(f"Text encoder server embedding must have shape [T, D], got {tensor.shape}.")
            self._establish_metadata(
                self._response_metadata(result),
                observed_dim=int(tensor.shape[-1]),
            )
            length = tensor.shape[0]

            tensors.append(tensor)
            lengths.append(length)

        padded_tensor = np.zeros((len(lengths), max(lengths), tensors[0].shape[-1]), dtype=tensors[0].dtype)
        for idx, (tensor, length) in enumerate(zip(tensors, lengths)):
            padded_tensor[idx, :length] = tensor

        padded_tensor = torch.from_numpy(padded_tensor)
        padded_tensor = padded_tensor.to(device=self.device, dtype=self.dtype)
        return padded_tensor, lengths
