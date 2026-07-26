# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-window sizing for the interactive demo.

Kept free of demo/torch imports so it can be unit-tested standalone.
"""

import math
from typing import Optional


def compute_window_num_frames(
    history_length: int,
    gen_horizon_len: int,
    num_frames_per_token: int,
    max_window_len: int,
    history_start_idx: int = 0,
    max_constraint_idx: Optional[int] = None,
    future_crop_length: int = 0,
) -> int:
    """Number of frames of the sequence visible to the model in one step.

    Args:
        history_length: frames of history included in the window.
        gen_horizon_len: frames generated per autoregressive step.
        num_frames_per_token: window length must be a multiple of this.
        max_window_len: total window budget in frames (the size TRT engines
            are built for); multiple of ``num_frames_per_token``.
        history_start_idx: absolute frame index where the window starts.
        max_constraint_idx: absolute frame index of the furthest timeline
            constraint beyond the history, or None if there is none.
        future_crop_length: max frames of future context to expose for
            constraints past the generation horizon.
    """
    num_frames = history_length + gen_horizon_len

    if max_constraint_idx is not None:
        # Grow the window so the furthest constraint is visible as future
        # context, bounded by the future-crop setting.
        num_frames = max(num_frames, max_constraint_idx - history_start_idx + 1)
        num_frames = min(num_frames, future_crop_length + history_length + gen_horizon_len)
        num_frames = math.ceil(num_frames / num_frames_per_token) * num_frames_per_token

    # The future-crop slider max is derived against the horizon alone, so
    # future_crop + history + horizon can exceed the budget by up to
    # history_length frames — past what a TRT engine was built for. Trim the
    # future context back to the budget, but never below history + horizon:
    # the model cannot generate a window smaller than that.
    num_frames = max(min(num_frames, max_window_len), history_length + gen_horizon_len)

    return num_frames
