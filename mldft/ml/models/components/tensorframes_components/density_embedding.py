from e3nn.o3 import Linear
from torch import Tensor, nn

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.density_coeff_embedding import AtomHotEmbedding
from mldft.ml.models.components.tensorframes_components.irreps_conversion import (
    convert_tensorframes_irreps_to_e3nn,
)
from tensorframes.reps import Irreps
from tensorframes.reps.utils import parse_reps


class DensityEmbedding(nn.Module):
    """DensityEmbedding class."""

    def __init__(
        self,
        embedding_irreps: Irreps,
        basis_info: BasisInfo,
    ) -> None:
        """Initializes the DensityEmbedding class.

        Args:
            embedding_irreps (Irreps): Irreps object for the embedding.
            basis_info (BasisInfo): BasisInfo object.
        """
        super().__init__()
        self.irreps_in = basis_info.irreps_per_atom.sum()
        self.embedding_irreps = parse_reps(embedding_irreps)
        self.embedding_irreps = convert_tensorframes_irreps_to_e3nn(self.embedding_irreps)

        self.atom_hot_embedding = AtomHotEmbedding(embed_dim=self.irreps_in.dim)
        self.linear = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.embedding_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, batch: OFData, rescaled_coeffs: Tensor = None) -> Tensor:
        """Forward pass.

        Args:
            batch (OFData): Data batch.
            rescaled_coeffs (Tensor, optional): Rescaled coefficients. If None, batch.coeffs is used. Defaults to None.
        """
        if rescaled_coeffs is None:
            coeffs = batch.coeffs
        else:
            coeffs = rescaled_coeffs

        atom_hot_embedding = self.atom_hot_embedding(
            coeffs,
            batch.basis_function_ind,
            batch.n_basis_per_atom,
            batch.coeff_ind_to_node_ind,
        )
        embedding = self.linear(atom_hot_embedding)
        return embedding
