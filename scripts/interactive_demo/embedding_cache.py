# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed cache for text-prompt embeddings.

Wraps the demo's shared text encoder (built once in ``run_demo.py`` and reused across every model
load — see ``loading.py``) so repeated or preset prompts skip re-encoding, both within a session and
across demo restarts.

Adapted from the pattern in ``kimodo/kimodo/demo/embedding_cache.py``, trimmed for this demo: no
index.json bookkeeping, no contextvars-based session-cache layer, and no prewarm-marker files (a
disk-cache hit already makes prewarm idempotent).
"""

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

# Relative to the working directory the demo is launched from, matching the
# other ".cache/..." defaults used throughout interactive_demo (gui/io.py,
# gui/visualize.py, motion_io.py).
DEFAULT_CACHE_DIR = os.path.join(".cache", "text_embeddings")
DEFAULT_MAX_MEM_ENTRIES = 128


def _normalize_texts(texts) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


class EmbeddingCache:
    """Thread-safe in-memory LRU + per-prompt disk cache for text embeddings.

    Each entry stores one prompt's UNPADDED embedding (``tensor[i, :length]``) as a float32 numpy
    array, so cached entries can be re-padded batch-wise on read no matter how prompts were
    originally grouped into batches.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_mem_entries: int = DEFAULT_MAX_MEM_ENTRIES,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_mem_entries = max_mem_entries
        self._lock = threading.Lock()
        self._mem_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._dir_ready = False  # cache_dir created lazily, on first disk write

    @staticmethod
    def _make_key(text: str) -> str:
        # Keyed on the raw prompt text only. The text encoder is shared
        # across every loaded model, and entries are always stored as
        # float32 regardless of the encoder's current precision — serving a
        # float32-stored embedding under either the bf16 or fp32 dropdown
        # setting is acceptable for this demo, so the key doesn't need to
        # encode device/dtype.
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _mem_get(self, key: str) -> Optional[np.ndarray]:
        value = self._mem_cache.get(key)
        if value is not None:
            self._mem_cache.move_to_end(key)
        return value

    def _mem_put(self, key: str, value: np.ndarray) -> None:
        self._mem_cache[key] = value
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > self.max_mem_entries:
            self._mem_cache.popitem(last=False)

    def _disk_load(self, key: str) -> Optional[np.ndarray]:
        path = self._entry_path(key)
        if not os.path.exists(path):
            return None
        try:
            return np.load(path)
        except Exception:
            return None

    def _disk_save(self, key: str, value: np.ndarray) -> None:
        # Disk persistence is best-effort (matching _disk_load): on a full or
        # read-only volume the entry stays in the memory LRU and the call
        # still succeeds, degrading to memory-only caching.
        try:
            if not self._dir_ready:
                os.makedirs(self.cache_dir, exist_ok=True)
                self._dir_ready = True
            np.save(self._entry_path(key), value)
        except Exception as e:
            print(f"[text-embedding cache] Disk write failed (non-fatal, entry kept in memory): {e!r}")

    def contains(self, text: str) -> bool:
        """Whether ``text`` is already cached (memory or disk).

        Read-only; used only to report prewarm hit/miss counts.
        """
        key = self._make_key(text)
        with self._lock:
            if key in self._mem_cache:
                return True
        return os.path.exists(self._entry_path(key))

    def get_or_encode(self, texts: list[str], encoder) -> tuple[torch.Tensor, list[int]]:
        """Return ``(padded_tensor, lengths)`` for ``texts``, encoding only the prompts that aren't
        already cached.

        ``encoder`` is called at most once, batched over every miss.
        """
        if not texts:
            # Preserve whatever the wrapped encoder does with an empty
            # batch instead of inventing our own (kimodo's equivalent path
            # calls the shape-less ``torch.empty()``, which raises).
            return encoder(texts)

        arrays: list[Optional[np.ndarray]] = [None] * len(texts)
        lengths: list[int] = [0] * len(texts)
        misses: list[tuple[int, str]] = []

        with self._lock:
            for idx, text in enumerate(texts):
                key = self._make_key(text)
                cached = self._mem_get(key)
                if cached is None:
                    cached = self._disk_load(key)
                    if cached is not None:
                        self._mem_put(key, cached)
                if cached is not None:
                    arrays[idx] = cached
                    lengths[idx] = int(cached.shape[0])
                else:
                    misses.append((idx, key))

        if misses:
            miss_texts = [texts[idx] for idx, _key in misses]
            # Encoding runs outside the lock: it's the slow part (network
            # round-trip or GPU forward pass) and unrelated cache lookups
            # from other threads shouldn't block on it. If two threads race
            # to encode the same miss, both do redundant work but write
            # back identical content under the lock below — harmless, last
            # write wins.
            miss_tensor, miss_lengths = encoder(miss_texts)
            miss_tensor = miss_tensor.detach().to(device="cpu", dtype=torch.float32)
            miss_arrays = miss_tensor.numpy()

            with self._lock:
                for miss_pos, (idx, key) in enumerate(misses):
                    length = int(miss_lengths[miss_pos])
                    arr = np.ascontiguousarray(miss_arrays[miss_pos, :length])
                    arrays[idx] = arr
                    lengths[idx] = length
                    self._mem_put(key, arr)
                    self._disk_save(key, arr)

        feat_dim = arrays[0].shape[-1]
        max_len = max(lengths)
        padded = np.zeros((len(texts), max_len, feat_dim), dtype=np.float32)
        for idx, arr in enumerate(arrays):
            padded[idx, : arr.shape[0]] = arr

        return torch.from_numpy(padded), lengths


def _probe_device_dtype(encoder) -> tuple:
    """Best-effort initial ``(device, dtype)`` for ``encoder``.

    Tries ``.device``/``.dtype`` attributes first (present on ``TextEncoderAPI``,
    ardy/model/text_encoder_api.py), then the first parameter of the encoder itself or of a nested
    ``.model`` submodule (covers ``LLM2VecEncoder``, ardy/model/llm2vec/llm2vec_wrapper.py, which
    stores its real ``nn.Module`` in ``.model`` and exposes neither attribute directly), and finally
    falls back to cpu/float32.
    """
    device = getattr(encoder, "device", None)
    dtype = getattr(encoder, "dtype", None)
    if device is None or dtype is None:
        for params_source in (encoder, getattr(encoder, "model", None)):
            params = getattr(params_source, "parameters", None)
            if not callable(params):
                continue
            try:
                first = next(params())
            except (StopIteration, TypeError):
                continue
            device = device or first.device
            dtype = dtype or first.dtype
            break
    return device or torch.device("cpu"), dtype or torch.float32


class CachedTextEncoder:
    """Wraps a text encoder with a disk-backed :class:`EmbeddingCache`.

    Preserves the wrapped encoder's ``(tensor, lengths)`` contract (see
    ``ardy/model/text_encoder_api.py``): cached entries are re-padded batch-wise and the returned
    tensor is always cast to the encoder's CURRENT device/dtype before returning, mirroring
    ``TextEncoderAPI.__call__``'s final ``.to(device=self.device, dtype=self.dtype)`` — bfloat16
    round-trips losslessly through the cache's float32 storage.
    """

    def __init__(
        self,
        encoder,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_mem_entries: int = DEFAULT_MAX_MEM_ENTRIES,
    ) -> None:
        self.encoder = encoder
        self.cache = EmbeddingCache(cache_dir=cache_dir, max_mem_entries=max_mem_entries)
        self._device, self._dtype = _probe_device_dtype(encoder)

    def __call__(self, texts):
        texts = _normalize_texts(texts)
        tensor, lengths = self.cache.get_or_encode(texts, self.encoder)
        if not texts:
            # Empty batch was fully delegated to the wrapped encoder
            # (see EmbeddingCache.get_or_encode) — return it untouched
            # rather than re-casting a result we didn't produce.
            return tensor, lengths
        return tensor.to(device=self._device, dtype=self._dtype), lengths

    def prewarm(self, texts) -> None:
        """Warm the cache for ``texts`` (e.g. the Prompt List presets) so the first click of a
        preset button after startup is instant.

        Meant to run on a background daemon thread: never raises, so a
        slow or unreachable encoder can't take the demo down.
        """
        texts = _normalize_texts(texts)
        try:
            already_cached = sum(1 for text in texts if self.cache.contains(text))
            self(texts)
            print(
                f"[text-embedding cache] Prewarm complete: "
                f"{len(texts) - already_cached} encoded, "
                f"{already_cached} already cached."
            )
        except Exception as e:
            print(f"[text-embedding cache] Prewarm failed (non-fatal): {e!r}")

    def to(self, device=None, dtype=None):
        if hasattr(self.encoder, "to"):
            self.encoder.to(device=device, dtype=dtype)
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return self

    def __getattr__(self, name):
        return getattr(self.encoder, name)
