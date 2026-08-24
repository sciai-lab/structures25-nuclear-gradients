# This file is copied from https://github.com/atomicarchitects/equiformer_v2 licensed under the MIT License.

import numpy as np
import torch
from e3nn.o3 import Irreps, Linear, Norm
from torch import Tensor, nn

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


class InitialGuessDelta(nn.Module):
    """InitialGuessDelta class."""

    def __init__(
        self,
        density_embedding_lmax_list: list[int],
        n_channels: int,
        basis_info: BasisInfo,
        dataset_statistics: DatasetStatistics = None,
        weigher_key: str = None,
    ) -> None:
        """Initializes the InitialGuessDelta class.

        Args:
            density_embedding_lmax_list (list[int]): List of lmax values for the density embedding.
            n_channels (int): Number of channels.
            basis_info (BasisInfo): BasisInfo object.
            dataset_statistics (DatasetStatistics, optional): DatasetStatistics object. Defaults to None.
            weigher_key (str, optional): Weigher key for rescaling the initial guess. Defaults to None.
        """
        super().__init__()
        self.density_embedding_lmax_list = density_embedding_lmax_list
        self.n_channels = n_channels

        irrep_list = [
            (n_channels, (l_angular, -(l_angular % 2 * 2) + 1))
            for lmax in self.density_embedding_lmax_list
            for l_angular in range(lmax + 1)
        ]
        self.embedding_irreps = Irreps(irrep_list)

        irreps_in_per_atomtype = basis_info.irreps_per_atom

        self.linear = Linear(
            irreps_in=self.embedding_irreps,
            irreps_out=irreps_in_per_atomtype.sum(),
            internal_weights=True,
            shared_weights=True,
        )

        self.with_scale_shift = dataset_statistics is not None
        if self.with_scale_shift:
            mean = dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/mean")
            std = dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/std")

            scalar_mask = basis_info.l_per_basis_func == 0

            mean[~scalar_mask] = 0

            irreps = "+".join([str(irreps) for irreps in basis_info.irreps_per_atom])
            norm = Norm(Irreps(irreps))
            std = norm(torch.as_tensor(std)).numpy()
            std = std[basis_info.basis_func_to_shell] / np.sqrt(
                2 * basis_info.l_per_basis_func + 1
            )

            self.register_buffer(
                "mean",
                torch.as_tensor(mean),
            )
            self.register_buffer(
                "std",
                torch.as_tensor(std),
            )

    def forward(
        self, embedding: Tensor, basis_function_ind: Tensor, coeff_ind_to_node_ind: Tensor
    ) -> Tensor:
        """Forward pass.

        Args:
            embedding (Tensor): Embedding tensor.
            basis_function_ind (Tensor): Basis function indices from the corresponding OFData object.
            coeff_ind_to_node_ind (Tensor): Coefficient indices to node indices from the corresponding OFData object.
        """

        # Convert the embedding from the equiformerv2 format to the e3nn format.
        # e3nn stores all the irreps in sequence, while the equiformer uses channels. This means for the equiformer there is always
        # the same number of features for each l_angular, because there is exactly one feature per l_angular and channel.
        unchanneled_embedding = torch.empty(
            embedding.shape[0],
            embedding.shape[1] * embedding.shape[2],
            device=embedding.device,
            dtype=embedding.dtype,
        )
        slices = self.embedding_irreps.slices()
        for l_max in self.density_embedding_lmax_list:
            offset = 0
            for l_angular in range(l_max + 1):
                unchanneled_embedding[..., slices[l_angular]].view(
                    embedding.shape[0], -1, 2 * l_angular + 1
                )[:] = embedding[..., offset : offset + 2 * l_angular + 1, :].permute(0, 2, 1)
                offset += 2 * l_angular + 1

        # Linear transformation of the embedding to get the initial guess delta
        inital_guess_delta = self.linear(unchanneled_embedding)

        out = inital_guess_delta[coeff_ind_to_node_ind, basis_function_ind]

        # Rescale the initial guess delta using the dataset statistics
        if self.with_scale_shift:
            out *= self.std[basis_function_ind]
            out += self.mean[basis_function_ind]

        return out
