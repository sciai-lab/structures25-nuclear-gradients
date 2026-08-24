from typing import Union

import numpy as np
import torch
from e3nn.o3 import Linear, Norm
from torch import Tensor, nn

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.models.components.tensorframes_components.irreps_conversion import (
    convert_tensorframes_irreps_to_e3nn,
)
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from tensorframes.reps import Irreps
from tensorframes.reps.utils import parse_reps


class InitialGuessDelta(nn.Module):
    """InitialGuessDelta class."""

    def __init__(
        self,
        embedding_irreps: Union[Irreps, str],
        basis_info: BasisInfo,
        dataset_statistics: DatasetStatistics = None,
        weigher_key: str = None,
    ) -> None:
        """Initializes the InitialGuessDelta class.

        Args:
            embedding_irreps (Union[Irreps, str]): Irreps object for the embedding.
            basis_info (BasisInfo): BasisInfo object.
            dataset_statistics (DatasetStatistics, optional): DatasetStatistics object. Defaults to None.
            weigher_key (str, optional): Weigher key. Defaults to None.
        """
        super().__init__()
        self.embedding_irreps = parse_reps(embedding_irreps)
        self.embedding_irreps = convert_tensorframes_irreps_to_e3nn(self.embedding_irreps)

        self.irreps_out = basis_info.irreps_per_atom.sum()
        self.linear = Linear(
            irreps_in=self.embedding_irreps,
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.with_scale_shift = dataset_statistics is not None
        if self.with_scale_shift:
            mean = dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/mean")
            std = dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/std")

            scalar_mask = basis_info.l_per_basis_func == 0

            mean[~scalar_mask] = 0

            norm = Norm(self.irreps_out)
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

        # Linear transformation of the embedding to get the initial guess delta
        inital_guess_delta = self.linear(embedding)

        out = inital_guess_delta[coeff_ind_to_node_ind, basis_function_ind]

        # Rescale the initial guess delta using the dataset statistics
        if self.with_scale_shift:
            out *= self.std[basis_function_ind]
            out += self.mean[basis_function_ind]

        return out
