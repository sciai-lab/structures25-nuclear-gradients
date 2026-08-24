import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.pool import global_add_pool

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.dimension_wise_rescaling import DimensionWiseRescaling
from mldft.ml.models.components.model_output import ModelOutput
from mldft.ml.models.components.tensorframes_components.density_embedding import (
    DensityEmbedding,
)
from mldft.ml.models.components.tensorframes_components.initial_guess_delta_module import (
    InitialGuessDelta,
)
from mldft.utils.log_utils.logging_mixin import LoggingMixin

# class AtomicNumbersConverter:

#     def __init__(self, dataset_atomic_numbers) -> None:
#         self._atomic_numbers_mapping = torch.zeros(dataset_atomic_numbers.max() + 1, dtype=torch.int64)
#         for i, atomic_number in enumerate(dataset_atomic_numbers):
#             self._atomic_numbers_mapping[atomic_number] = i

#     def __call__(self, batch: OFData) -> OFData:
#         batch.z = self._atomic_numbers_mapping[batch.atomic_numbers]
#         return batch


def _convert_to_tensorframes_data(batch: OFData, rescaled_coeffs: Tensor = None) -> Batch:
    """Converts the input data to the format used by the TensorFrames architectures.

    Args:
        batch (OFData): Data batch.

    Returns:
        Batch (Batch): The converted data.
    """
    coeffs = batch.coeffs if rescaled_coeffs is None else rescaled_coeffs

    if hasattr(batch, "batch_size"):
        data = Batch(
            x=coeffs,
            pos=batch.pos,
            atomic_numbers=batch.atomic_numbers,
            z=batch.atom_ind,
            batch=batch.batch,
            edge_index=batch.edge_index,
        )
    else:
        data = Batch.from_data_list(
            [
                Data(
                    x=coeffs,
                    pos=batch.pos,
                    atomic_numbers=batch.atomic_numbers,
                    edge_index=batch.edge_index,
                    z=batch.atom_ind,
                )
            ]
        )

    return data


class TensorFramesNet(nn.Module, LoggingMixin):
    """A wrapper to bring the inputs into the correct format for tensorframes based
    architectures."""

    def __init__(
        self,
        net_main: torch.nn.Module,
        initial_guess_module: InitialGuessDelta,
        energy_mlp: nn.Sequential,
        atom_ref_module: AtomRef,
        dimension_wise_rescaling_module: DimensionWiseRescaling,
        density_embedding: DensityEmbedding,
    ) -> None:
        super().__init__()
        """Initializes the TensorFramesNet class.

        Args:
            net_main (torch.nn.Module): The main neural network module.
            initial_guess_module (InitialGuessDelta): The initial guess module.
            energy_mlp (nn.Sequential): The energy MLP module.
            atom_ref_module (AtomRef): The atom reference module.
            dimension_wise_rescaling_module (DimensionWiseRescaling): The dimension-wise rescaling module.
            density_embedding (DensityEmbedding): The density embedding module.
        """
        self.net_main = net_main
        self.atom_ref_module = atom_ref_module
        self.initial_guess_module = initial_guess_module
        self.density_embedding = density_embedding
        self.energy_mlp = energy_mlp
        self.dimension_wise_rescaling_module = dimension_wise_rescaling_module

    def forward(self, batch: OFData) -> ModelOutput:
        """Calculates the forward pass of the module according to OFDFT training strategy.

        Args:
            batch (OFData): The batch / data object containing the input data.

        Returns:
            ModelOutput: Typed outputs with ``pred_energy`` and ``pred_diff``.
        """

        # dimensionwise rescaling
        rescaled_coeffs = self.dimension_wise_rescaling_module(batch)
        if self.density_embedding is not None:
            rescaled_coeffs = self.density_embedding(batch=batch, rescaled_coeffs=rescaled_coeffs)

        data = _convert_to_tensorframes_data(batch=batch, rescaled_coeffs=rescaled_coeffs)
        node_features = self.net_main(data)
        energies_per_atom = self.energy_mlp(node_features)
        energies = global_add_pool(energies_per_atom, batch.batch)[:, 0]

        # lower part of figure 6
        energies_atom_ref = self.atom_ref_module.sample_forward(batch)
        self.log(energy_atom_ref=energies_atom_ref.sum(), energy_g3d=energies.sum())
        energies += energies_atom_ref
        # output branch initial guess
        initial_guess_delta = self.initial_guess_module(
            embedding=node_features,
            basis_function_ind=batch.basis_function_ind,
            coeff_ind_to_node_ind=batch.coeff_ind_to_node_ind,
        )

        return ModelOutput(
            pred_energy=energies,
            pred_gradient=None,
            pred_diff=initial_guess_delta,
        )
