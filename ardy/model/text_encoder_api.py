# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote text encoder API client (Gradio) for motion generation."""

import logging

import numpy as np
import torch
from gradio_client import Client

# Suppress the [httpx] logs (GET requests)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress internal gradio_client logs
logging.getLogger("gradio_client").setLevel(logging.WARNING)


class TextEncoderAPI:
    """Text encoder API client for motion generation."""

    def __init__(self, url: str):
        self.client = Client(url, verbose=False)
        self.device = "cpu"
        self.dtype = torch.float

    def _create_np_random_name(self):
        import uuid

        return str(uuid.uuid4()) + ".npy"

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

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
            path = result[0]["value"]
            tensor = np.load(path)
            length = tensor.shape[0]

            tensors.append(tensor)
            lengths.append(length)

        padded_tensor = np.zeros((len(lengths), max(lengths), tensors[0].shape[-1]), dtype=tensors[0].dtype)
        for idx, (tensor, length) in enumerate(zip(tensors, lengths)):
            padded_tensor[idx, :length] = tensor

        padded_tensor = torch.from_numpy(padded_tensor)
        padded_tensor = padded_tensor.to(device=self.device, dtype=self.dtype)
        return padded_tensor, lengths
