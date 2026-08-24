"""Fast-path eSEN layers used by fresh models.

The legacy classes in `embedding.py` and `esen_block.py` are still kept so old
Lightning checkpoints that pickle those concrete module types can be deserialized.
New models should use the classes in this file instead.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.nn import Linear
from torch.profiler import record_function

from mldft.ml.models.components.equiformerv2_components.activation import GateActivation
from mldft.ml.models.components.esen_components.layer_norm import (
    get_normalization_layer,
)
from mldft.ml.models.components.esen_components.radial import (
    RadialMLP,
)
from mldft.ml.models.components.esen_components.s2_activation import (
    SeparableS2ActivationM,
)
from mldft.ml.models.components.esen_components.so3 import CoefficientMapping
from mldft.ml.models.components.esen_components.so3_layers import SO3_Linear


class FastExecutionBackend:
    """Small tensor kernels used by the fast path to avoid repeated layout boilerplate."""

    @staticmethod
    def prepare_wigner(
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        mapping_reduced: CoefficientMapping,
        coefficient_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert l-primary Wigner matrices to the reduced m-primary layout used by the fast
        blocks."""
        if coefficient_index is not None:
            wigner = wigner.index_select(1, coefficient_index)
            wigner_inv = wigner_inv.index_select(2, coefficient_index)

        to_m = mapping_reduced.to_m.to(dtype=wigner.dtype)
        wigner = torch.einsum("mk,nkj->nmj", to_m, wigner)
        wigner_inv = torch.einsum("njk,mk->njm", wigner_inv, to_m)
        return wigner, wigner_inv

    @staticmethod
    def node_to_edge_wigner_permute(
        x_full: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Gather source/target node features per edge and rotate them into the edge frame."""
        x_source = x_full.index_select(0, edge_index[0])
        x_target = x_full.index_select(0, edge_index[1])
        x_message = torch.cat((x_source, x_target), dim=2)
        if wigner.dtype != x_message.dtype:
            wigner = wigner.to(x_message.dtype)
        return torch.bmm(wigner, x_message)

    @staticmethod
    def permute_wigner_inv_edge_to_node(
        x_message: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_offset: int,
    ) -> torch.Tensor:
        """Rotate edge messages back to the node frame and scatter-add them to target nodes."""
        if wigner_inv.dtype != x_message.dtype:
            wigner_inv = wigner_inv.to(x_message.dtype)
        x_rotated = torch.bmm(wigner_inv, x_message)
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding.index_add_(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(
        x: torch.Tensor,
        radial_output: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_index: torch.Tensor,
        m_0_num_coefficients: int,
        sphere_channels: int,
        rescale_factor: float,
        node_offset: int,
    ) -> torch.Tensor:
        """Project the radial edge-degree embedding into spherical coefficients and accumulate per
        node."""
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        if wigner_inv_m0.dtype != radial.dtype:
            wigner_inv_m0 = wigner_inv_m0.to(radial.dtype)
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)
        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding.to(x.dtype) / rescale_factor,
        )


class SO2_m_Conv(torch.nn.Module):
    def __init__(
        self,
        m: int,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax

        assert self.mmax >= m
        num_coefficients = self.lmax - m + 1
        num_channels = num_coefficients * self.sphere_channels
        self.out_channels_half = self.m_output_channels * (num_channels // self.sphere_channels)
        self.fc = Linear(num_channels, 2 * self.out_channels_half, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = self.fc(x_m)
        x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(x_m.shape[0], -1, self.out_channels_half).split(
            1, dim=1
        )
        x_m_r = x_r_0 - x_i_1
        x_m_i = x_r_1 + x_i_0
        return (
            x_m_r.view(x_m.shape[0], -1, self.m_output_channels),
            x_m_i.view(x_m.shape[0], -1, self.m_output_channels),
        )


class FastSO2Convolution(torch.nn.Module):
    """SO(2) convolution operating directly on the packed m-primary representation."""

    def __init__(
        self,
        sphere_channels: int,
        m_output_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        internal_weights: bool = True,
        edge_channels_list: list[int] | None = None,
        extra_m0_output_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels
        self.edge_channels_list = copy.deepcopy(edge_channels_list) if edge_channels_list else None

        num_channels_m0 = (self.lmax + 1) * self.sphere_channels
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.sphere_channels)
        if self.extra_m0_output_channels is not None:
            m0_output_channels += self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        self.so2_m_conv = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(
                    m,
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax,
                    self.mmax,
                )
            )
            num_channels_rad += self.so2_m_conv[-1].fc.in_features

        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            edge_channels_list = copy.deepcopy(self.edge_channels_list)
            edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialMLP(edge_channels_list)

        self.m_split_sizes = [self.mappingReduced.m_size[0]] + (
            2 * torch.tensor(self.mappingReduced.m_size[1:])
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply per-m convolutions, optionally modulated by edge-conditioned radial weights."""
        if self.rad_func is not None:
            if x_edge is None:
                raise ValueError("FastSO2Convolution requires x_edge when internal_weights=False.")
            x_edge = self.rad_func(x_edge)

        x_by_m = x.split(self.m_split_sizes, dim=1)
        x_edge_by_m = x_edge.split(self.edge_split_sizes, dim=1) if x_edge is not None else None
        num_edges = x.shape[0]

        x_0 = x_by_m[0].view(num_edges, -1)
        if x_edge_by_m is not None:
            x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        if self.extra_m0_output_channels is not None:
            x_0_extra, x_0 = x_0.split(
                (
                    self.extra_m0_output_channels,
                    self.fc_m0.out_features - self.extra_m0_output_channels,
                ),
                -1,
            )

        out = [x_0.view(num_edges, -1, self.m_output_channels)]
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if x_edge_by_m is not None:
                x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            out.extend(self.so2_m_conv[m - 1](x_m))

        out_tensor = torch.cat(out, dim=1)
        if self.extra_m0_output_channels is not None:
            return out_tensor, x_0_extra
        return out_tensor


class FastEdgeDegreeEmbedding(torch.nn.Module):
    """Fast edge-degree embedding that only materializes the populated m=0 channels."""

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        rescale_factor: float,
        mappingReduced: CoefficientMapping,
        backend: FastExecutionBackend,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.backend = backend
        self.m_0_num_coefficients = self.mappingReduced.m_size[0]

        edge_channels_list = copy.deepcopy(edge_channels_list)
        edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(edge_channels_list)
        self.rescale_factor = rescale_factor

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_inv_envelope: torch.Tensor,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """Compute the invariant radial features and scatter them into the equivariant node
        state."""
        radial = self.rad_func(x_edge)
        return self.backend.edge_degree_scatter(
            x,
            radial,
            wigner_inv_envelope,
            edge_index,
            self.m_0_num_coefficients,
            self.sphere_channels,
            self.rescale_factor,
            node_offset,
        )


class FastEdgewise(torch.nn.Module):
    """Edge update block for the fast path using m-primary SO(2) convolutions."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        mappingReduced: CoefficientMapping,
        SO3_grid: nn.ModuleDict,
        backend: FastExecutionBackend,
        act_type: str = "gate",
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.backend = backend
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
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

        self.so2_conv_1 = FastSO2Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )
        self.so2_conv_2 = FastSO2Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )
        self.out_mask = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(self.lmax, self.mmax)

    def forward(
        self,
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv_envelope: torch.Tensor,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """Rotate node features to each edge frame, apply the edge MLP, and scatter back."""
        with record_function("esen.edgewise"):
            x_message = self.backend.node_to_edge_wigner_permute(x, edge_index, wigner)
            x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
            if self.act_type == "gate":
                x_message = self.act(x_0_gating, x_message, m_primary=True)
            else:
                x_message = self.act(x_0_gating, x_message)
            x_message = self.so2_conv_2(x_message)
            return self.backend.permute_wigner_inv_edge_to_node(
                x_message,
                wigner_inv_envelope,
                edge_index,
                x.shape[0],
                node_offset,
            )


class SpectralAtomwise(torch.nn.Module):
    """Atom-wise residual MLP implemented directly in coefficient space."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.sphere_channels, self.lmax * self.hidden_channels, bias=True),
            nn.SiLU(),
        )
        self.so3_linear_1 = SO3_Linear(self.sphere_channels, self.hidden_channels, lmax=self.lmax)
        self.act = GateActivation(
            lmax=self.lmax,
            mmax=self.lmax,
            num_channels=self.hidden_channels,
        )
        self.so3_linear_2 = SO3_Linear(self.hidden_channels, self.sphere_channels, lmax=self.lmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class GridAtomwise(torch.nn.Module):
    """Atom-wise residual MLP implemented by projecting to an S2 grid and back."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        SO3_grid: nn.ModuleDict,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        x_grid = self.grid_mlp(x_grid)
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)


class FastESENBlock(torch.nn.Module):
    """Fast residual eSEN block combining one edge update and one atom-wise update."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mappingReduced: CoefficientMapping,
        SO3_grid: nn.ModuleDict,
        edge_channels_list: list[int],
        norm_type: str,
        act_type: str,
        mlp_type: str,
        backend: FastExecutionBackend,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
        )
        self.edge_wise = FastEdgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            backend=backend,
            act_type=act_type,
        )
        self.norm_2 = get_normalization_layer(
            norm_type,
            lmax=self.lmax,
            num_channels=sphere_channels,
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
        x: torch.Tensor,
        x_edge: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
        wigner_inv_envelope: torch.Tensor,
        node_offset: int = 0,
    ) -> torch.Tensor:
        """Run the edge residual branch followed by the atom-wise residual branch."""
        del edge_distance

        x_res = x
        x = self.norm_1(x)
        x = self.edge_wise(
            x,
            x_edge,
            edge_index,
            wigner,
            wigner_inv_envelope,
            node_offset,
        )
        x = x + x_res

        x_res = x
        x = self.norm_2(x)
        x = self.atom_wise(x)
        return x + x_res
