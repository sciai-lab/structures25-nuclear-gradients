import torch
from e3nn.o3 import Irreps, Linear
from loguru import logger
from torch import Tensor, nn

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.density_coeff_embedding import AtomHotEmbedding
from mldft.ml.models.components.equiformerv2_components.layer_norm import (
    get_normalization_layer,
)
from mldft.utils.local_frames import pyscf_to_e3nn_matrix


class DensityEmbedding(nn.Module):
    """DensityEmbedding class."""

    def __init__(
        self,
        density_embedding_lmax_list: list[int],
        n_channels: int,
        basis_info: BasisInfo,
        norm_type: str = None,
        apply_pyscf_to_e3nn_conversion: bool = False,
    ) -> None:
        """Initializes the DensityEmbedding class.

        Args:
            density_embedding_lmax_list (list[int]): List of lmax values for the density embedding.
            n_channels (int): Number of channels.
            basis_info (BasisInfo): BasisInfo object.
            norm_type (str, optional): Type of normalization. One of 'layer_norm', 'layer_norm_sh', 'rms_norm_sh', or None. Defaults to None.
            apply_pyscf_to_e3nn_conversion (bool): If True, applies the pyscf->e3nn Wigner rotation
                inside the model. Set to False if the data was already converted by ToE3NNConvention.
                Defaults to False.
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

        self.atom_hot_embedding = AtomHotEmbedding(
            sum(basis_info.basis_dim_per_atom),
        )
        self.linear = Linear(
            irreps_in=irreps_in_per_atomtype.sum(),
            irreps_out=self.embedding_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        if norm_type is not None:
            self.use_norm = True
            self.norm = get_normalization_layer(
                norm_type, lmax=max(self.density_embedding_lmax_list), num_channels=self.n_channels
            )
        else:
            self.use_norm = False

        self.num_elements = sum([irr.dim for irr in self.embedding_irreps])
        self.apply_pyscf_to_e3nn_conversion = apply_pyscf_to_e3nn_conversion
        if apply_pyscf_to_e3nn_conversion:
            # Register each wigner matrix as an individual buffer
            l_max = max(self.density_embedding_lmax_list)
            for l in range(l_max + 1):
                to_e3nn_wm = pyscf_to_e3nn_matrix(Irreps(f"{l}e"))
                self.register_buffer(
                    f"to_e3nn_wigner_matrix_{l}", to_e3nn_wm.to(torch.get_default_dtype())
                )
            logger.info(
                "DensityEmbedding: applying pyscf->e3nn convention conversion inside the model. "
                "Make sure ToE3NNConvention is NOT in the data pipeline."
            )
        self.gamma = 0.0

        # The conversion mask is used to convert the embedding from e3nn format to the format used in the equiformerv2.
        # e3nn stores all the irreps in sequence, while the equiformer uses channels. This means for the equiformer there is always
        # the same number of features for each l_angular, because there is exactly one feature per l_angular and channel.
        self.conversion_mask = torch.zeros(
            n_channels * (density_embedding_lmax_list[0] + 1) ** 2, dtype=torch.int
        )
        for l_max in density_embedding_lmax_list:
            for l_angular in range(l_max + 1):
                for m in range(0, 2 * l_angular + 1):
                    for c in range(n_channels):
                        self.conversion_mask[
                            ..., l_angular**2 * (n_channels) + m * (n_channels) + c
                        ] = (l_angular**2 * (n_channels) + m + c * (2 * l_angular + 1))

    def forward(self, batch: OFData, rescaled_coeffs: Tensor = None) -> Tensor:
        """Forward pass.

        Args:
            batch (OFData): Data batch.
            rescaled_coeffs (Tensor, optional): Rescaled coefficients. If None, batch.coeffs is used. Defaults to None.
        """
        if rescaled_coeffs is not None:
            coeffs = rescaled_coeffs
        else:
            coeffs = batch.coeffs

        atom_hot_embedding = self.atom_hot_embedding(
            coeffs,
            batch.basis_function_ind,
            batch.n_basis_per_atom,
            batch.coeff_ind_to_node_ind,
        )

        embedding = self.linear(atom_hot_embedding)

        # Convert the embedding from e3nn format to the format used in the equiformerv2
        channeled_embedding = embedding[..., self.conversion_mask].view(
            embedding.shape[0], -1, self.n_channels
        )

        # Apply the wigner matrices (batched per l) to the embedding to convert to the e3nn convention
        if hasattr(self, "apply_pyscf_to_e3nn_conversion") and self.apply_pyscf_to_e3nn_conversion:
            for l_max in self.density_embedding_lmax_list:
                offset = 0
                for l_angular in range(l_max + 1):
                    channeled_embedding[..., offset : offset + 2 * l_angular + 1, :] = (
                        torch.einsum(
                            "ij,...jk->...ik",
                            getattr(self, f"to_e3nn_wigner_matrix_{l_angular}"),
                            channeled_embedding[..., offset : offset + 2 * l_angular + 1, :],
                        )
                    )
                    offset += 2 * l_angular + 1

        if self.use_norm:
            channeled_embedding = self.norm(channeled_embedding)

        return channeled_embedding
