"""Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the LICENSE file in the root directory
of this source tree.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from e3nn import o3

YTOL = 0.999999


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(
    lv: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    _Jd: list[torch.Tensor],
) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    frequencies = torch.arange(lv, -lv - 1, -1, device=angle.device, dtype=angle.dtype)
    phases = angle.unsqueeze(-1) * frequencies
    diagonal = torch.diag_embed(torch.cos(phases))
    anti_diagonal = torch.flip(torch.diag_embed(torch.sin(phases)), dims=(-1,))
    return diagonal + anti_diagonal


def rotation_to_wigner(
    edge_rot_mat: torch.Tensor,
    start_lmax: int,
    end_lmax: int,
    Jd: list[torch.Tensor],
    rot_clip: bool = False,
) -> torch.Tensor:
    """Set <rot_clip=True> to handle gradient instability when using gradient-based force/stress
    prediction."""
    x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = o3.xyz_to_angles(x)
    R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ edge_rot_mat
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

    if rot_clip:
        yprod = x[..., 1].detach()
        mask = (yprod > -YTOL) & (yprod < YTOL)
        alpha = torch.where(mask, alpha, alpha.detach())
        gamma = torch.where(mask, gamma, gamma.detach())
        beta_detach = beta.detach()
        beta_detach = torch.where(yprod > YTOL, torch.zeros_like(beta_detach), beta_detach)
        beta_detach = torch.where(
            yprod < -YTOL,
            torch.full_like(beta_detach, math.pi),
            beta_detach,
        )
        beta = torch.where(mask, beta, beta_detach)

    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = edge_rot_mat.new_zeros(len(alpha), size, size)
    start = 0
    for lv in range(start_lmax, end_lmax + 1):
        block = wigner_D(lv, alpha, beta, gamma, Jd).to(wigner.dtype)
        block_size = block.size(-1)
        pad = (start, size - start - block_size, start, size - start - block_size)
        wigner = wigner + F.pad(block, pad)
        start = start + block_size

    if rot_clip:
        return wigner
    else:
        return wigner.detach()
