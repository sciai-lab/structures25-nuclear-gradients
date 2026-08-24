"""Atomic reference module (see [M-OFDFT]_, Section `4.3 paragraph 3 <https://arxiv.org/html/2309.16578v2#S4.SS3.SSS0.Px3>`_)."""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData


class AtomRef(nn.Module):
    """Atomic reference module
    (see [M-OFDFT]_, Section `4.3 paragraph 3 <https://arxiv.org/html/2309.16578v2#S4.SS3.SSS0.Px3>`_).
    """

    def __init__(
        self,
        t_z: torch.Tensor | np.ndarray,
        t_global: float,
        g_z: torch.Tensor | np.ndarray,
    ) -> None:
        """Initialize the atomic reference module. Alternatively use
        :meth:`from_dataset_statistics`.

        Args:
            t_z: Kinetic energy per atomic species. Shape (n_atom_types,).
            t_global: Global bias.
            g_z: Mean gradient vector, concatenated over atomic species. Shape (basis,).
        """
        super().__init__()
        # Register large energy biases as float64 for better precision
        self.register_buffer("t_z", torch.as_tensor(t_z, dtype=torch.float64))
        self.t_global = float(t_global)
        self.register_buffer("g_z", torch.as_tensor(g_z, dtype=torch.float64))

    def _apply(self, fn):
        """Apply module-wide tensor transforms while preserving float64 atom refs."""
        atom_ref_buffers = {
            name: self._buffers.pop(name) for name in ("t_z", "g_z") if name in self._buffers
        }
        super()._apply(fn)
        for name, buffer in atom_ref_buffers.items():
            device_probe = fn(torch.empty(0, device=buffer.device))
            self.register_buffer(name, buffer.to(device=device_probe.device, dtype=torch.float64))
        return self

    @classmethod
    def from_dataset_statistics(
        cls,
        dataset_statistics,
        basis_info: BasisInfo = None,
        weigher_key: str = "has_energy_label",
        mode: Literal["all", "scalar_only", "energy_only", "zero"] = "all",
    ) -> "AtomRef":
        """Initialize using a :class:`~mldft.ml.preprocess.dataset_statistics.DatasetStatistics`
        object.

        Args:
            dataset_statistics: Dataset statistics object.
            basis_info: Basis information object. Only required for mode='scalar_only' to infer
                which features are scalars.
            weigher_key (str): Selects the sample weigher for the dataset statistics.
            mode: Which setting to use:
                'all' to include energy and all gradient contributions,
                'scalar_only' to only multiply gradients to scalars for equivariance
                'energy_only' to only use the energy contributions without the gradient
                'zero' to turn off the atomref with less struggle
        """
        if mode == "all":
            return cls(
                t_z=dataset_statistics.load_statistic(weigher_key, "atom_ref_atom_type_bias"),
                t_global=float(
                    dataset_statistics.load_statistic(weigher_key, "atom_ref_global_bias")
                ),
                g_z=dataset_statistics.load_statistic(weigher_key, "gradient_label/mean"),
            )
        elif mode == "scalar_only":
            assert basis_info is not None, "basis_info must be provided if scalar_only is True"
            scalar_mask = basis_info.l_per_basis_func == 0
            g_mean = dataset_statistics.load_statistic(weigher_key, "gradient_label/mean")
            g_z = g_mean * scalar_mask
            return cls(
                t_z=dataset_statistics.load_statistic(
                    weigher_key, "scalar_atom_ref_atom_type_bias"
                ),
                t_global=float(
                    dataset_statistics.load_statistic(weigher_key, "scalar_atom_ref_global_bias")
                ),
                g_z=g_z,
            )
        elif mode == "energy_only":
            return cls(
                t_z=dataset_statistics.load_statistic(
                    weigher_key, "atom_ref_energy_only_atom_type_bias"
                ),
                t_global=dataset_statistics.load_statistic(
                    weigher_key, "atom_ref_energy_only_global_bias"
                ),
                g_z=np.zeros_like(
                    dataset_statistics.load_statistic(weigher_key, "gradient_label/mean")
                ),
            )

        elif mode == "zero":
            return cls(
                t_z=np.zeros_like(
                    dataset_statistics.load_statistic(weigher_key, "atom_ref_atom_type_bias")
                ),
                t_global=0.0,
                g_z=np.zeros_like(
                    dataset_statistics.load_statistic(weigher_key, "gradient_label/mean")
                ),
            )
        else:
            raise KeyError(f"Configured mode {mode} unknown.")

    def forward(
        self,
        atom_ind: torch.Tensor,
        coeffs: torch.Tensor,
        basis_function_ind: torch.Tensor,
        atom_batch: (
            torch.Tensor | None
        ),  # do not set default to avoid passing data from a batch on accident
        coeffs_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Calculates the energy for one molecule, or a batch of molecules, according to a linear
        fit, e.g. computed by :class:`~mldft.ml.preprocess.dataset_statistics.DatasetStatistics`.

        Args:
            atom_ind: Atom types in the molecule (indexing the atomic numbers present in the basis). Shape (n_atoms,).
            coeffs: Coefficients of the molecule. Shape (n_basis,).
            basis_function_ind: Indices of the basis functions in the molecule.
                See :attr:`mldft.ml.data.components.of_data.OFData.basis_function_ind`. Shape (n_basis,).
            atom_batch: Batch index for the atoms. Shape (n_atoms,), or None if not batched.
            coeffs_batch: Batch index for the coefficients. Shape (n_basis,), or None if not batched.
            num_graphs: Number of graphs in the batch, or None if not batched.
        """
        # compute the gradient times the coefficients for the molecule, i.e. the part proportional to the coefficients
        g_m = self.g_z[basis_function_ind]
        g_times_p = global_add_pool(g_m * coeffs.to(torch.float64), coeffs_batch, num_graphs)

        # compute the bias per molecule, by summing the bias per atom
        t_m = global_add_pool(self.t_z[atom_ind], atom_batch, num_graphs)

        # return the sum of all contributions
        return g_times_p + t_m + self.t_global

    def sample_forward(self, sample: OFData) -> torch.Tensor:
        """Calculate the expected energy for a sample (or batch) with a linear fit, e.g. computed
        by :class:`~mldft.ml.preprocess.dataset_statistics.DatasetStatistics`.

        Args:
            sample: OFData object containing the molecule.
        """
        if isinstance(sample, Batch):
            coeffs_batch = sample.coeffs_batch
            atom_batch = sample.batch
            num_graphs = sample.num_graphs
        else:
            coeffs_batch = None
            atom_batch = None
            num_graphs = None
        return self(
            atom_ind=sample.atom_ind,
            coeffs=sample.coeffs,
            atom_batch=atom_batch,
            coeffs_batch=coeffs_batch,
            basis_function_ind=sample.basis_function_ind,
            num_graphs=num_graphs,
        )


class SimpleQuadraticAtomRef(nn.Module):
    def __init__(self, ground_state_coeff_mean: Tensor, factor: float):
        """
        Args:
            ground_state_coeff_mean: The mean of the ground state coefficients per atom type.
            factor: The factor to multiply the quadratic term with.
        """
        super().__init__()
        self.ground_state_coeff_mean = ground_state_coeff_mean
        self.factor = factor

    def forward(
        self, coeffs: Tensor, basis_function_ind: Tensor, coeffs_batch: Tensor | None
    ) -> Tensor:
        """Compute an isotropic quadratic energy around the mean ground state coeffs.

        Args:
            coeffs: The coefficients of the molecule. Shape (n_basis,).
            basis_function_ind: Indices of the basis functions in the molecule.
                See :attr:`mldft.ml.data.components.of_data.OFData.basis_function_ind`. Shape (n_basis,).
            coeffs_batch: Batch index for the coefficients. Shape (n_basis,), or None if not batched.
        """
        coeff_delta = coeffs - self.ground_state_coeff_mean[basis_function_ind]
        return self.factor * global_add_pool(coeff_delta**2, coeffs_batch)

    def sample_forward(self, sample: OFData) -> Tensor:
        """Compute the energy for a sample.

        Args:
            sample: The sample to compute the energy for.
        """
        return self(sample.coeffs, sample.basis_function_ind, sample.coeffs_batch)
