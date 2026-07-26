# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu


class TrajectorySmoother:
    """A class for modifying trajectories to hit specific values at specific frames while respecting
    soft constraints.

    This class modifies a trajectory to hit specific values at specific
    frames, while respecting the following soft constraints:
    * Preserve the original positions
    * Bring the accelerations as close to zero as possible

    The weights of the position soft constraints are specified in pos_weight.

    This is posed as a minimization problem:
    E(x) = pos_weight * |x - x_orig|^2 +
           |A x|^2

    where you minimize E(x) subject to specified values at indices where
    "mask" is equal to 1. A is a matrix that computes the N-2 accelerations
    associated with frames n-1, n and n+1.
    """

    """
    min f(x) + g(z)
    s.t. I x - z = 0

    x --> argmin_x (f(x) + p/2 ||I x - z + u||^2)
    z --> argmin_z (g(z) + p/2 ||I x - z + u||^2)
    u --> u + I x - z

    f(x) = pos_weight * |x - x_orig|^2 + |A x|^2
    g(z) = inf if any(|z-t| > margin) else 0

    x minimization:

    E(x) = wp/2 * |x - x_orig|^2 + 1/2 |A x|^2 + p/2 |I x - z + u|^2
    E(x) = wp/2 * (x - x_orig)^T (x - x_orig) + x^T A^T A x + p/2 (I x - z + u)^T (I x - z + u)
    E(x) = wp/2 * (x^T x - x_orig^T x - x^T x_orig + x_orig^T x_orig) +
           1/2 x^T A^T A x +
           p/2 (x^T I^T I x - x^T I^T z + x^T I^T u - z^T I x + z^T z - z^T u + u^T I x - u^T z + u^T u)

    argmin E(x) = argmin [
            wp/2 * (x^T x - 2 x^T x_orig) +
            1/2 x^T A^T A x +
            p/2 (x^T I^T I x + 2 x^T I^T (u - z) )
    ]
    = argmin [
            x^T wp/2 * I * x - wp * x^T x_orig +
            1/2 x^T A^T A x +
            x^T p/2 I^T I x + p x^T I^T (u - z)
    ]
    = argmin [
            1/2 x^T (wp * I + A^T A + p I^T I) x - x^T (wp * x_orig + p I^T (z - u))
    ]

    x = (wp * I + A^T A + p I)^-1 (wp * x_orig + p (z - u))


    """

    def __init__(
        self,
        margins,
        pos_weight=0.0,
        loop=False,
        admm_iters=100,
        alpha_overrelax=1.0,
        circle_project=False,
    ):
        """Initialize the TrajectorySmoother.

        Args:
            margins: Array of margin values for each frame.
                    margins[i] < 0: unconstrained
                    margins[i] == 0: pinned on this frame
                    margins[i] > 0: can deviate within the margin
            pos_weight: Weight for position preservation
            loop: Whether the trajectory should loop
            admm_iters: Number of ADMM iterations
        """
        self.pos_weight = pos_weight
        self.admm_iters = admm_iters
        self.alpha_overrelax = alpha_overrelax
        self.circle_project = circle_project
        N = len(margins)

        # Store margin information as numpy arrays
        self.margin_vals = margins

        # Build acceleration matrix A
        a_data = []
        a_rows = []
        a_cols = []

        for i in range(1, N - 1):
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([i, i, i])
            a_cols.extend([i - 1, i, i + 1])

        if loop:
            # Add periodic accelerations
            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([0, 0, 0])
            a_cols.extend([N - 1, 0, 1])

            scale = 1.0
            a_data.extend([-scale, 2.0 * scale, -scale])
            a_rows.extend([N - 1, N - 1, N - 1])
            a_cols.extend([N - 2, N - 1, 0])

        A = sparse.csr_matrix((a_data, (a_rows, a_cols)), shape=(N, N))

        # Build identity matrix
        identity_matrix = sparse.eye(N)

        # Build system matrix M
        M = pos_weight * identity_matrix + A.T @ A

        # Calculate ADMM step size
        diag_max = max(abs(M.diagonal()))
        self.admm_stepsize = 0.25 * np.sqrt(diag_max)

        M = M + self.admm_stepsize * identity_matrix
        self.system_lu = splu(M.tocsc())

    def smooth(self, targets, x0):
        """Interpolate between reference positions while satisfying constraints.

        Args:
            observations: Target positions for constrained frames (numpy array)
            ref_positions: Reference positions defining original shape
                         (numpy array)

        Returns:
            Interpolated positions (numpy array)
        """
        x_target = targets.copy()
        x = x0.copy()
        z = np.zeros_like(x)
        u = np.zeros_like(x)

        for _ in range(self.admm_iters):
            self.z_update(z, x, x_target, u)
            self.u_update(u, x, z)
            self.x_update(x, z, u, x_target)

        return x

    def x_update(self, x, z, u, x_t):
        """Update x in the ADMM iteration."""
        # x = (wp * I + A^T A + p I)^-1 (wp * x_orig + p (z - u))
        r = self.pos_weight * x_t + self.admm_stepsize * (z - u)
        x[:] = self.system_lu.solve(r)

    def z_update(self, z, x, z_t, u):
        """Update z in the ADMM iteration using vectorized operations."""
        # Compute the difference from target for all margin locations at once
        z[:] = x + u - z_t

        # Check if we need to project back to margin
        z_diff_norms = np.linalg.norm(z, axis=1)
        mask = z_diff_norms > self.margin_vals
        if np.any(mask):
            scale_factors = self.margin_vals[mask] / z_diff_norms[mask]
            z[mask] *= scale_factors[:, np.newaxis]

        # Add back the target
        z[:] += z_t

        if self.circle_project:
            z[:] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1.0e-6)

    def u_update(self, u, x, z):
        """Update u in the ADMM iteration using vectorized operations."""
        u[:] += self.alpha_overrelax * (x - z)


def smooth_signal(x, margins, pos_weight=0, alpha_overrelax=1.8, admm_iters=500, circle_project=False):
    x_smoothed = x.copy()
    x_smoothed[:] = x.mean(axis=0, keepdims=True)

    # smooth the signal, multigrid style by starting out coarse,
    # doubling the resolution and repeating until we're at the full
    # resolution, using the previous result as the initial guess.
    levels = int(math.floor(math.log2(len(x))))
    levels = max(levels - 4, 1)

    stepsize = 2**levels
    while True:
        # smooth signals at this level:
        num_steps = len(x_smoothed[::stepsize])
        smoother = TrajectorySmoother(
            margins=margins[::stepsize],
            pos_weight=pos_weight,
            alpha_overrelax=alpha_overrelax,
            admm_iters=admm_iters,
            circle_project=circle_project,
        )
        x_smoothed[::stepsize] = smoother.smooth(x[::stepsize], x_smoothed[::stepsize])

        # interpolate to next level:
        next_stepsize = stepsize // 2
        num_interleaved = len(x_smoothed[next_stepsize::stepsize])
        if num_interleaved == num_steps:
            # linearly extrapolate the last value if we have to:
            x_smoothed[next_stepsize::stepsize][-1] = (
                x_smoothed[::stepsize][-1] + (x_smoothed[::stepsize][-1] - x_smoothed[::stepsize][-2]) / 2
            )
            num_interleaved = num_interleaved - 1

        # linearly interpolate the remaining values:
        x_smoothed[next_stepsize::stepsize][:num_interleaved] = (
            x_smoothed[::stepsize][:-1] + x_smoothed[::stepsize][1:]
        ) / 2

        if stepsize == 1:
            break

        stepsize //= 2

    return x_smoothed


def get_smooth_root_pos(hip_translations):
    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]

    batch_size, nframes = root_translations_xz.shape[:2]
    margins = np.full(root_translations_xz.shape[1], 0.06)

    root_translations_smoothed_xz = []
    for batch in range(batch_size):
        root_translations_smoothed_xz.append(
            smooth_signal(root_translations_xz[batch].detach().cpu().numpy(), margins)[None]
        )

    root_translations_smoothed_xz = torch.tensor(np.concatenate(root_translations_smoothed_xz))

    root_translations = torch.cat(
        [
            root_translations_smoothed_xz.to(root_translations_y.device),
            root_translations_y,
        ],
        dim=-1,
    )[..., [0, 2, 1]]

    return root_translations


def smooth_trajectory_gpu(
    positions: torch.Tensor,
    margin: float = 0.06,
    smoothness_weight: float = 1.0,
    position_weight: float = 0.01,
    num_iters: int = 50,
) -> torch.Tensor:
    """GPU-accelerated trajectory smoothing using gradient descent.

    Args:
        positions: [N, D] tensor of positions to smooth
        margin: Maximum allowed deviation from original positions
        smoothness_weight: Weight for smoothness term (minimizes acceleration)
        position_weight: Weight for position preservation term
        num_iters: Number of optimization iterations

    Returns:
        Smoothed positions [N, D]
    """
    device = positions.device
    n_frames = positions.shape[0]

    if n_frames < 3:
        return positions.clone()

    # Initialize smoothed positions with original
    smoothed = positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([smoothed], lr=0.01)

    for _ in range(num_iters):
        optimizer.zero_grad()

        # Smoothness loss: minimize second derivative (acceleration)
        vel = smoothed[1:] - smoothed[:-1]
        acc = vel[1:] - vel[:-1]
        smoothness_loss = smoothness_weight * torch.sum(acc**2)

        # Position preservation loss (soft constraint)
        position_loss = position_weight * torch.sum((smoothed - positions) ** 2)

        # Total loss
        loss = smoothness_loss + position_loss

        loss.backward()
        optimizer.step()

        # Project back to margin constraints
        with torch.no_grad():
            diff = smoothed - positions
            diff_norm = torch.norm(diff, dim=1, keepdim=True)
            mask = diff_norm > margin
            if mask.any():
                scale = margin / (diff_norm[mask] + 1e-8)
                smoothed[mask] = positions[mask] + diff[mask] * scale

    return smoothed.detach()


def smooth_trajectory_gpu_fast(
    positions: torch.Tensor,
    margin: float = 0.06,
    smoothness_weight: float = 100.0,
    position_weight: float = 0.1,
) -> torch.Tensor:
    """
    Fast GPU-accelerated trajectory smoothing using direct least squares solution.
    No iterative optimization - solves in one shot.

    Args:
        positions: [N, D] tensor of positions to smooth
        margin: Maximum allowed deviation from original positions
        smoothness_weight: Weight for smoothness term (minimizes acceleration)
        position_weight: Weight for position preservation term

    Returns:
        Smoothed positions [N, D]
    """
    device = positions.device
    dtype = positions.dtype
    n_frames = positions.shape[0]
    n_dims = positions.shape[1]

    if n_frames < 3:
        return positions.clone()

    # Build acceleration matrix A for second derivative
    # A[i] computes: x[i-1] - 2*x[i] + x[i+1]
    A = torch.zeros((n_frames - 2, n_frames), device=device, dtype=dtype)
    for i in range(n_frames - 2):
        A[i, i] = 1.0
        A[i, i + 1] = -2.0
        A[i, i + 2] = 1.0

    # Solve: minimize smoothness_weight * ||Ax||^2 + position_weight * ||x - x_orig||^2
    # Solution: x = (smoothness_weight * A^T A + position_weight * I)^(-1) * (position_weight * x_orig)

    ATA = A.T @ A  # [N, N]
    M = smoothness_weight * ATA + position_weight * torch.eye(n_frames, device=device, dtype=dtype)

    # Solve for each dimension independently
    smoothed = torch.zeros_like(positions)
    for d in range(n_dims):
        b = position_weight * positions[:, d]  # [N]
        # Use torch.linalg.solve for numerical stability
        smoothed[:, d] = torch.linalg.solve(M, b)

    # Apply margin constraints
    diff = smoothed - positions
    diff_norm = torch.norm(diff, dim=1)  # [N] - no keepdim to avoid shape mismatch
    mask = diff_norm > margin  # [N] boolean mask
    if mask.any():
        scale = margin / (diff_norm[mask] + 1e-8)  # [M] where M = number of True in mask
        # Need to broadcast scale [M] to match diff[mask] which is [M, D]
        smoothed[mask] = positions[mask] + diff[mask] * scale[:, None]

    return smoothed


def get_smooth_root_pos_gpu(hip_translations: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated version of get_smooth_root_pos.

    Args:
        hip_translations: [B, T, 3] root positions

    Returns:
        Smoothed root positions [B, T, 3]
    """
    device = hip_translations.device
    root_translations_xz = hip_translations[..., [0, 2]]  # [B, T, 2]
    root_translations_y = hip_translations[..., [1]]  # [B, T, 1]

    batch_size = root_translations_xz.shape[0]

    # Smooth XZ components
    root_translations_smoothed_xz = []
    for batch_idx in range(batch_size):
        smoothed_xz = smooth_trajectory_gpu(
            root_translations_xz[batch_idx],  # [T, 2]
            margin=0.06,
            smoothness_weight=1.0,
            position_weight=0.01,
            num_iters=50,
        )
        root_translations_smoothed_xz.append(smoothed_xz[None])  # [1, T, 2]

    root_translations_smoothed_xz = torch.cat(root_translations_smoothed_xz, dim=0)  # [B, T, 2]

    # Combine smoothed XZ with original Y
    root_translations = torch.cat(
        [
            root_translations_smoothed_xz,
            root_translations_y,
        ],
        dim=-1,
    )  # [B, T, 3] with order [x, z, y]

    # Reorder to [x, y, z]
    root_translations = root_translations[..., [0, 2, 1]]

    return root_translations


def get_smooth_root_pos_gpu_fast(hip_translations: torch.Tensor) -> torch.Tensor:
    """Fast GPU-accelerated version of get_smooth_root_pos using direct solver.

    Args:
        hip_translations: [B, T, 3] root positions

    Returns:
        Smoothed root positions [B, T, 3]
    """
    device = hip_translations.device
    root_translations_xz = hip_translations[..., [0, 2]]  # [B, T, 2]
    root_translations_y = hip_translations[..., [1]]  # [B, T, 1]

    batch_size = root_translations_xz.shape[0]

    # Smooth XZ components using fast method
    root_translations_smoothed_xz = []
    for batch_idx in range(batch_size):
        smoothed_xz = smooth_trajectory_gpu_fast(
            root_translations_xz[batch_idx],  # [T, 2]
            margin=0.06,
            smoothness_weight=100.0,
            position_weight=0.1,
        )
        root_translations_smoothed_xz.append(smoothed_xz[None])  # [1, T, 2]

    root_translations_smoothed_xz = torch.cat(root_translations_smoothed_xz, dim=0)  # [B, T, 2]

    # Combine smoothed XZ with original Y
    root_translations = torch.cat(
        [
            root_translations_smoothed_xz,
            root_translations_y,
        ],
        dim=-1,
    )  # [B, T, 3] with order [x, z, y]

    # Reorder to [x, y, z]
    root_translations = root_translations[..., [0, 2, 1]]

    return root_translations
