"""Legacy edge-degree embedding kept for checkpoint compatibility.

Fresh eSEN models use the fast-path implementation in `fast_layers.py`.
This file remains because old Lightning checkpoints deserialize concrete
`EdgeDegreeEmbedding` instances from here.

Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the LICENSE file in the root directory
of this source tree.
"""

from __future__ import annotations

import copy

import torch

from mldft.ml.models.components.esen_components.radial import (
    PolynomialEnvelope,
    RadialMLP,
)


class EdgeDegreeEmbedding(torch.nn.Module):
    """Legacy edge-degree embedding kept only for unpickled checkpoints.

    Args:
        sphere_channels (int):      Number of spherical channels

        lmax (int):                 degrees (l)
        mmax (int):                 orders (m)

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
        cutoff (float):             Cutoff distance for the radial function

        mappingReduced (CoefficientMapping): Class to convert l and m indices once node embedding is rotated
        out_mask (torch.Tensor):    Mask to select the output irreps
        use_envelope (bool):        Whether to use envelope function
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        max_num_elements: int,
        edge_channels_list,
        rescale_factor,
        cutoff,
        mappingReduced,
        out_mask,
        use_envelope,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced

        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(self.edge_channels_list)

        self.rescale_factor = rescale_factor

        self.use_envelope = use_envelope
        if self.use_envelope:
            self.cutoff = cutoff
            self.envelope = PolynomialEnvelope(exponent=5)

        self.out_mask = out_mask

    def forward(
        self,
        x,
        x_edge,
        edge_distance,
        edge_index,
        wigner_inv,
        node_offset=0,
    ):
        """Project the legacy radial embedding into spherical channels and scatter it to target
        nodes."""
        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)

        # FairChem only rotates the populated m=0 block instead of padding all m values.
        wigner_inv_m0 = wigner_inv.narrow(2, 0, self.m_0_num_coefficients)
        if self.use_envelope:
            env = self.envelope(edge_distance / self.cutoff)
            wigner_inv_m0 = wigner_inv_m0 * env.view(-1, 1, 1)
        if wigner_inv_m0.dtype != x_edge_m_0.dtype:
            wigner_inv_m0 = wigner_inv_m0.to(x_edge_m_0.dtype)

        x_edge_embedding = torch.bmm(wigner_inv_m0, x_edge_m_0).to(x.dtype)
        x.index_add_(0, edge_index[1] - node_offset, x_edge_embedding / self.rescale_factor)
        return x
