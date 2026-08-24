"""Shared S2 activation helpers used by both the fast and legacy eSEN blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class S2ActivationM(nn.Module):
    """Apply a pointwise nonlinearity on an S2 grid and project back to m-primary coefficients."""

    def __init__(self, lmax: int, mmax: int, SO3_grid, to_m: torch.Tensor) -> None:
        super().__init__()
        self.act = torch.nn.SiLU()
        to_grid_mat = SO3_grid["lmax_mmax"].get_to_grid_mat(device=None)
        from_grid_mat = SO3_grid["lmax_mmax"].get_from_grid_mat(device=None)
        self.register_buffer(
            "to_grid_mat_m",
            torch.einsum("ji,bai->jba", to_m, to_grid_mat),
            persistent=False,
        )
        self.register_buffer(
            "from_grid_mat_m",
            torch.einsum("ji,bai->baj", to_m, from_grid_mat),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_grid = torch.einsum("iba,zic->zbac", self.to_grid_mat_m, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai,zbac->zic", self.from_grid_mat_m, x_grid)


class SeparableS2ActivationM(nn.Module):
    """Apply separate scalar and tensor activations while preserving the m-primary layout."""

    def __init__(self, lmax: int, mmax: int, SO3_grid, to_m: torch.Tensor) -> None:
        super().__init__()
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2ActivationM(lmax, mmax, SO3_grid, to_m)

    def forward(self, input_scalars: torch.Tensor, input_tensors: torch.Tensor) -> torch.Tensor:
        output_scalars = self.scalar_act(input_scalars).reshape(
            input_scalars.shape[0], 1, input_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return torch.cat((output_scalars, output_tensors[:, 1:]), dim=1)
