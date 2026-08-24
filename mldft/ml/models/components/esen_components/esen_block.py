"""Legacy eSEN block kept for checkpoint compatibility.

Fresh eSEN models use the fast-path implementation in `fast_layers.py`.
This file remains because old Lightning checkpoints deserialize concrete
`eSEN_Block` instances from here.

Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the LICENSE file in the root directory
of this source tree.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mldft.ml.models.components.equiformerv2_components.activation import (
    GateActivation,
)
from mldft.ml.models.components.esen_components.layer_norm import (
    get_normalization_layer,
)
from mldft.ml.models.components.esen_components.radial import PolynomialEnvelope
from mldft.ml.models.components.esen_components.s2_activation import (
    SeparableS2ActivationM,
)
from mldft.ml.models.components.esen_components.so2_layers import SO2_Convolution
from mldft.ml.models.components.esen_components.so3_layers import SO3_Linear


class Edgewise(torch.nn.Module):
    """Legacy edge update used only by old deserialized eSEN blocks."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        mappingReduced,
        SO3_grid,
        cutoff,
        act_type="gate",
        use_envelope: bool = True,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.act_type = act_type

        if self.act_type == "gate":
            self.act = GateActivation(
                lmax=self.lmax,
                mmax=self.mmax,
                num_channels=self.hidden_channels,
                mappingReduced=self.mappingReduced,
            )
            extra_m0_output_channels = self.lmax * self.hidden_channels
        elif self.act_type == "s2":
            self.act = SeparableS2ActivationM(
                lmax=self.lmax,
                mmax=self.mmax,
                SO3_grid=self.SO3_grid,
                to_m=self.mappingReduced.to_m,
            )
            extra_m0_output_channels = self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        self.use_envelope = use_envelope
        if self.use_envelope:
            self.cutoff = cutoff
            self.envelope = PolynomialEnvelope(exponent=5)

        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(self.lmax, self.mmax)

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_m,
        wigner_inv,
        node_offset: int = 0,
    ):
        """Run the legacy edge update in the edge frame and scatter the result back to nodes."""
        x_source = x.index_select(0, edge_index[0])
        x_target = x.index_select(0, edge_index[1])
        x_message = torch.cat((x_source, x_target), dim=2)
        if wigner_m.dtype != x_message.dtype:
            wigner_m = wigner_m.to(x_message.dtype)
        x_message = torch.bmm(wigner_m, x_message)

        # New modules use m-primary activations. Legacy pickled s2 activations are l-primary,
        # so keep a compatibility path for old checkpoints that stored that object.
        use_m_primary = self.act_type == "gate" or isinstance(self.act, SeparableS2ActivationM)
        x_message, x_0_gating = self.so2_conv_1(
            x_message,
            x_edge,
            input_m_primary=True,
            output_m_primary=use_m_primary,
        )
        if self.act_type == "gate":
            x_message = self.act(x_0_gating, x_message, m_primary=True)
        else:
            x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(
            x_message,
            x_edge,
            input_m_primary=use_m_primary,
            output_m_primary=True,
        )

        if self.use_envelope:
            env = self.envelope(edge_distance / self.cutoff)
            x_message = x_message * env.view(-1, 1, 1)

        if wigner_inv.dtype != x_message.dtype:
            wigner_inv = wigner_inv.to(x_message.dtype)
        x_message = torch.bmm(wigner_inv, x_message)
        new_embedding = torch.zeros(
            (x.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_message)
        return new_embedding


class SpectralAtomwise(torch.nn.Module):
    """Legacy atom-wise residual MLP in coefficient space."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(self.sphere_channels, self.hidden_channels, lmax=self.lmax)
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(self.hidden_channels, self.sphere_channels, lmax=self.lmax)

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class GridAtomwise(torch.nn.Module):
    """Legacy atom-wise residual MLP implemented through an S2 grid projection."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.grid_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.sphere_channels, bias=False),
        )

    def forward(self, x):
        # Project to grid
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        # Perform point-wise operations
        x_grid = self.grid_mlp(x_grid)
        # Project back to spherical harmonic coefficients
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)


class eSEN_Block(torch.nn.Module):
    """Legacy residual eSEN block preserved for checkpoint compatibility."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced,
        SO3_grid,
        edge_channels_list: list[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        mlp_type: str,
        use_envelope: bool,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        self.use_envelope = use_envelope

        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            cutoff=cutoff,
            act_type=act_type,
            use_envelope=use_envelope,
        )

        self.norm_2 = get_normalization_layer(
            norm_type, lmax=self.lmax, num_channels=sphere_channels
        )

        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif mlp_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        else:
            raise ValueError(f"Unknown MLP type {mlp_type}")

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner,
        wigner_inv,
        node_offset: int = 0,
    ):
        """Apply the legacy edge residual branch followed by the atom-wise residual branch."""
        x_res = x
        x = self.norm_1(x)

        x = self.edge_wise(
            x,
            x_edge,
            edge_distance,
            edge_index,
            wigner,
            wigner_inv,
            node_offset,
        )
        x = x + x_res

        x_res = x
        x = self.norm_2(x)

        x = self.atom_wise(x)
        return x + x_res
