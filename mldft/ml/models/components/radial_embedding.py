from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import relu

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData


class RadialEmbedding(nn.Module):
    """RadialEmbedding module for computing embeddings based on radial distances."""

    def __init__(self, basis_info: BasisInfo, directed: bool = True) -> None:
        """Initialize the RadialEmbedding module.

        Args:
            basis_info (BasisInfo): Used to determine the number of atom types.
            directed (bool, optional): Whether the learned means and stds are directed, i.e.,
                whether a C-H edge gets the same embedding as the reverse H-C edge.
        """
        super().__init__()

        self.basis_info = basis_info
        self.directed = directed
        self.n_types = basis_info.n_types
        self.n_edge_types = (
            basis_info.n_types**2
            if directed
            else basis_info.n_types * (basis_info.n_types + 1) // 2
        )

        self.mul = nn.Embedding(self.n_edge_types, 1)
        self.bias = nn.Embedding(self.n_edge_types, 1)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def get_rescaled_distances(self, sample: OFData) -> tuple[Tensor, Tensor]:
        """Function to compute distances between nodes.

        Calculates the distances of the edges. An affine linear transformation is applied to the
        distances where the multiplication and bias are dependent on the edge type, i.e., the atom
        types of the atoms connected by the edge.

        Args:
            sample (OFData): Sample containing the data.

        Returns:
            tuple[Tensor, Tensor]: Tuple containing the linearly mapped and the raw distances
                of the edges.
        """
        pos = sample.pos
        edge_index = sample.edge_index

        # get the position of the atom i and j from the edge index
        # i is the first row of the edge index and j is the second row
        pos_i = pos.index_select(0, edge_index[0])
        pos_j = pos.index_select(0, edge_index[1])

        # same for atom types
        atom_i = sample.atom_ind.index_select(0, edge_index[0])
        atom_j = sample.atom_ind.index_select(0, edge_index[1])
        if self.directed:
            edge_types = atom_i * self.n_types + atom_j
        else:
            # effectively, we sort the atom types to make the edge type symmetric
            edge_types = torch.min(atom_i, atom_j) * self.n_types + torch.max(atom_i, atom_j)

        # calculate the edge lengths
        x = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        distances = x.clone()

        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x + bias

        return x, distances

    def forward(self, sample: OFData) -> Tensor:
        """Forward pass for RadialEmbedding module."""
        raise NotImplementedError


class GaussianEmbedding(RadialEmbedding):
    """Class for Radial Embedding with Gaussian distribution
    (see [M-OFDFT]_, Supplementary Section `B.1.2 <https://arxiv.org/html/2309.16578v2#A2.SS1.SSS2>`_
    and [STRUCTURES25]_, Supplementary Section `S.2 <https://arxiv.org/html/2503.00443v2#Ax1.SS2>`_).
    """

    def __init__(
        self,
        basis_info: BasisInfo,
        out_channels: int = 128,
        normalized: bool = True,
        directed: bool = True,
        init_radius_range: tuple[float, float] = (0, 3),
        std_init_radius_range: tuple[float, float] = (0.1, 3),
    ) -> None:
        """Initialize the GaussianEmbedding.

        Args:
            basis_info (BasisInfo): Used to determine the number of atom types.
            out_channels (int, optional): Number of gaussians to use. Defaults to 128.
            normalized (bool, optional): Whether to normalize the gaussians. Defaults to True.
            directed (bool, optional): Whether the learned means and stds are directed, i.e. whether a C-H edge gets
                the same embedding as the reverse H-C edge. Defaults to True.
            init_radius_range (Tuple[float, float], optional): Range for the initialization of the means.
                Defaults to (0, 3).
            std_init_radius_range (Tuple[float, float], optional): Range for the initialization of the standard deviations.
                Defaults to (0.1, 3).
        """
        super().__init__(basis_info, directed)

        self.out_channels = out_channels

        # Sample both means and stds
        self.means = nn.Embedding(1, out_channels)
        self.stds = nn.Embedding(1, out_channels)
        self.normalized = normalized

        # Initialization of the means and stds
        nn.init.uniform_(self.means.weight, *init_radius_range)
        nn.init.uniform_(self.stds.weight, *std_init_radius_range)

    def gaussian(self, x, mean, std):
        """Gaussian kernel."""
        a = (2 * np.pi) ** 0.5
        return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (std * a)

    def rbf(self, x, mean, std):
        """Radial basis function kernel, a Gaussian without the normalization."""
        return torch.exp(-0.5 * (((x - mean) / std) ** 2))

    def forward(self, sample: OFData) -> tuple[Tensor, Tensor]:
        """Embed using Gaussians."""
        x, distances = self.get_rescaled_distances(sample)

        x = x.expand(-1, self.out_channels)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5

        if self.normalized:
            return self.gaussian(x.float(), mean, std).type_as(self.means.weight), distances
        else:
            return self.rbf(x.float(), mean, std).type_as(self.means.weight), distances


def envelope_default(d: torch.Tensor, p: int = 6) -> torch.Tensor:
    """Default envelope function.

    From J Gasteiger, J Groß, S Günnemann: Directional message passing for molecular graphs,
    https://openreview.net/forum?id=B1eWbxStPH. The envelope function is one at the origin and zero
    at d=1. The first and second derivatives are zero at d=1.

    Args:
       d: Distance tensor.
       p: Power of the envelope function.
    """
    polynomial = (
        1
        - (p + 1) * (p + 2) / 2 * d**p
        + p * (p + 2) * d ** (p + 1)
        - p * (p + 1) / 2 * d ** (p + 2)
    )
    return relu(polynomial)


class BesselEmbedding(RadialEmbedding):
    """Class for Radial Embedding with Bessel functions of first kind."""

    def __init__(
        self,
        basis_info: BasisInfo,
        out_channels: int = 128,
        directed: bool = True,
        learn_frequencies: bool = True,
        init_frequency_range: tuple[float, float] = (0.5, 10),
        envelope: Callable[[torch.Tensor], torch.Tensor] | None = envelope_default,
        cut_off_radius: float = 5.0,
    ) -> None:
        """Initialize the BesselEmbedding.

        Args:
            basis_info: Used to determine the number of atom types.
            out_channels: Number of bessel functions to use.
            directed: Whether the learned means and stds are directed, i.e. whether a C-H edge gets
                the same embedding as the reverse H-C edge.
            learn_frequencies: Whether to learn frequencies. Defaults to True.
            init_frequency_range: Range of the frequencies.
            envelope: Optional envelope function.
            cut_off_radius: Cut off radius. Only used for the envelope.
        """
        super().__init__(basis_info, directed)

        self.out_channels = out_channels
        self.cut_off = cut_off_radius

        self.frequencies = torch.linspace(*init_frequency_range, out_channels)
        if learn_frequencies:
            self.frequencies = nn.Parameter(self.frequencies)

        self.envelope = envelope

    def forward(self, sample: OFData) -> tuple[Tensor, Tensor]:
        """Embed using bessel functions."""
        x, distances = self.get_rescaled_distances(sample)
        x = x.expand(-1, self.out_channels)

        embedding = torch.sin(2 * np.pi * self.frequencies * x) / (x + 1e-9)
        if self.envelope is not None:
            embedding = 2 / self.cut_off * self.envelope(x / self.cut_off) * embedding

        return embedding, distances
