from abc import ABC

import torch
from loguru import logger
from pyscf import gto

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.density_optimization import density_optimization
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.nuclear_gradients import get_gradient_method
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.ofdft.run_density_optimization import SampleGenerator


class BaseNumHessian(ABC):
    """Base class for numerical Hessian and semi-numerical Hessian calculations.

    This class provides a common interface for computing the Hessian matrix using numerical
    methods, which can be extended for specific implementations.
    """

    def __init__(
        self,
        sample_generator: SampleGenerator,
        functional_factory: FunctionalFactory | None = None,
        optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
        epsilon: float = 1e-4,
    ) -> None:
        """Initializes the BaseNumHessian class."""
        if functional_factory is None:
            functional_factory = sample_generator.get_functional_factory()
        self.func_factory = functional_factory
        self.sample_generator = sample_generator
        self.optimizer = optimizer
        self.epsilon = epsilon

    def __call__(self, sample: OFData, coeffs: torch.Tensor) -> torch.Tensor:
        """Computes the Hessian matrix for a given sample and coefficients.

        Args:
            sample (OFData): The sample data containing the necessary inputs.
            coeffs (torch.Tensor): The coefficients in the AO basis used as initialization for the density
                optimization in the numerical procedure, advised to be set to the converged coeffs
                of the molecular structure whose hessian is to be evaluated.

        Returns:
            torch.Tensor: The computed Hessian matrix.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class NumericalHessian(BaseNumHessian):
    """Numerical Hessian calculation class.

    This version of the nuclear Hessian is less stable and efficient than the
    semi-numerical version. Usually larger step sizes and higher precision and convergence is required.

    .. math::

        H_{ij} = \\frac{\\partial^2 E}{\\partial R_i \\partial R_j} \\approx
        \\frac{1}{4\\epsilon^2} \\left( E_{ij}^{++} - E_{ij}^{+-} - E_{ij}^{-+} + E_{ij}^{--} \\right)

    Attributes:
        sample_generator (SampleGenerator): The generator for sample data.
        func_factory (FunctionalFactory | None): The factory for functional objects.
        optimizer (Optimizer): The optimizer used for density optimization.
        epsilon (float): The perturbation size for numerical differentiation.
    """

    def __init__(
        self,
        sample_generator: SampleGenerator,
        functional_factory: FunctionalFactory | None = None,
        optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
        epsilon: float = 5e-2,
    ) -> None:
        """Initializes the NumericalHessian class.

        Args:
            sample_generator (SampleGenerator): The generator for sample data.
            functional_factory (FunctionalFactory | None): The factory for functional objects.
            optimizer (Optimizer): The optimizer used for density optimization.
            epsilon (float): The perturbation size for numerical differentiation.
        """

        super().__init__(
            sample_generator,
            functional_factory=functional_factory,
            optimizer=optimizer,
            epsilon=epsilon,
        )

    def __call__(self, sample: OFData, coeffs: torch.Tensor) -> torch.Tensor:
        """Computes the numerical nuclear Hessian for a given sample and coefficients.

        Args:
            sample (OFData): The sample data containing the necessary inputs.
            coeffs (torch.Tensor): The coefficients in the AO basis used as initialization for the density
                optimization in the numerical procedure, advised to be set to the converged coeffs
                of the molecular structure whose hessian is to be evaluated.

        Returns:
            torch.Tensor: The computed Hessian matrix.
        """

        mol: gto.Mole = sample.mol
        coords = mol.atom_coords(unit="Bohr")

        hessian = torch.zeros((mol.natm, mol.natm, 3, 3), dtype=torch.float64)

        for i in range(mol.natm):
            for i_coord in range(3):
                for j in range(i, mol.natm):
                    for j_coord in range(3):
                        logger.trace(f"Computing Hessian element ({i}, {i_coord}, {j}, {j_coord})")
                        energies = torch.zeros(4, dtype=torch.float64)
                        for k, (sign_i, sign_j) in enumerate([(1, 1), (-1, 1), (1, -1), (-1, -1)]):
                            # Perturb the nuclear positions
                            perturbed_coords = coords.copy()
                            perturbed_coords[i, i_coord] += sign_i * self.epsilon
                            perturbed_coords[j, j_coord] += sign_j * self.epsilon
                            perturbed_mol = mol.set_geom_(
                                perturbed_coords, unit="Bohr", inplace=False
                            )

                            perturbed_sample = self.sample_generator.get_sample_from_mol(
                                perturbed_mol
                            )

                            transformed_initial_guess = transform_tensor_with_sample(
                                perturbed_sample,
                                coeffs.to(perturbed_sample.transformation_matrix.device),
                                representation=Representation.VECTOR,
                            )

                            energy, _, converged, _ = density_optimization(
                                perturbed_sample,
                                perturbed_mol,
                                self.optimizer,
                                self.func_factory,
                                initialization=transformed_initial_guess,
                                disable_pbar=True,
                                disable_printing=True,
                            )
                            if not converged:
                                logger.critical(
                                    f"Denop did not converge for coords: {i} {i_coord}, {j} {j_coord}"
                                )
                            # logger.debug(f"{i} {i_coord} {j} {j_coord} - {k} {sign_i} {sign_j} {energy.total_energy}")
                            energies[k] = energy.total_energy

                        hessian[i, j, i_coord, j_coord] = (
                            energies[0] - energies[1] - energies[2] + energies[3]
                        ) / (4 * (self.epsilon) ** 2)
                        # set lower triangular part
                        hessian[j, i, j_coord, i_coord] = hessian[i, j, i_coord, j_coord]
                        logger.trace(
                            f"{i} {i_coord} {j} {j_coord} - {hessian[i, j, i_coord, j_coord]}"
                        )

        return hessian


class SemiNumericalHessian(BaseNumHessian):
    """Semi-numerical Hessian calculation class.

    This is the standard approach used in QM codes for numerical nuclear Hessians if gradients are
    available. It is both more stable and efficient than the fully numerical version.

    .. math::
        H_{i\\cdot} = \\frac{\\partial^2 E}{\\partial R_i \\partial R} \\approx
        \\frac{1}{2\\epsilon} \\left(\\nabla_i^+ - \\nabla_i^-\\right)

    where :math:`\\nabla_i^+` and :math:`\\nabla_i^-` are the gradients

    Attributes:
        sample_generator (SampleGenerator): The generator for sample data.
        functional_factory (FunctionalFactory | None): The factory for functional objects.
        optimizer (Optimizer): The optimizer used for density optimization.
        epsilon (float): The perturbation size for numerical differentiation.
        grad (callable): The gradient computation method.
    """

    def __init__(
        self,
        sample_generator: SampleGenerator,
        functional_factory: FunctionalFactory | None = None,
        optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
        epsilon: float = 5e-3,
        gradients="analytical",
    ) -> None:
        """Initializes the SemiNumericalHessian class.

        Args:
            sample_generator (SampleGenerator): The generator for sample data.
            functional_factory (FunctionalFactory | None): The factory for functional objects.
            optimizer (Optimizer): The optimizer used for density optimization.
            epsilon (float): The perturbation size for numerical differentiation.
            gradients (str): The method to compute gradients, either 'analytical' or 'numerical'.
        """
        super().__init__(
            sample_generator,
            functional_factory=functional_factory,
            optimizer=optimizer,
            epsilon=epsilon,
        )
        self.grad = get_gradient_method(gradients, self.sample_generator, self.func_factory)

    def __call__(self, sample: OFData, coeffs: torch.Tensor) -> torch.Tensor:
        """Computes the semi-numerical nuclear Hessian for a given sample and coefficients.

        Args:
            sample (OFData): The sample data containing the necessary inputs.
            coeffs (torch.Tensor): The coefficients in the AO basis used as initialization for the density
                optimization in the numerical procedure, advised to be set to the converged coeffs
                of the molecular structure whose hessian is to be evaluated.

        Returns:
            torch.Tensor: The computed Hessian matrix.
        """
        mol: gto.Mole = sample.mol
        coords = mol.atom_coords(unit="Bohr")

        hessian = torch.zeros((mol.natm, mol.natm, 3, 3), dtype=torch.float64)

        for i in range(3 * mol.natm):
            i_coord = i % 3
            i_atom = i // 3
            grads = torch.zeros((2, 3 * mol.natm), dtype=torch.float64)
            for k, sign in enumerate([1, -1]):
                # Perturb the nuclear positions
                perturbed_coords = coords.copy()
                perturbed_coords[i_atom, i_coord] += sign * self.epsilon
                perturbed_mol = mol.set_geom_(perturbed_coords, unit="Bohr", inplace=False)

                # Get the perturbed sample
                perturbed_sample = self.sample_generator.get_sample_from_mol(perturbed_mol)
                # Transform the initial guess
                transformed_initial_guess = transform_tensor_with_sample(
                    perturbed_sample,
                    coeffs.to(perturbed_sample.transformation_matrix.device),
                    representation=Representation.VECTOR,
                )
                # Compute the energy
                _, final_coeffs, converged, _ = density_optimization(
                    perturbed_sample,
                    perturbed_mol,
                    self.optimizer,
                    self.func_factory,
                    initialization=transformed_initial_guess,
                    disable_pbar=True,
                    disable_printing=True,
                )
                if not converged:
                    logger.critical(f"Denop did not converge for coord: {i}")
                grads[k] = self.grad(
                    perturbed_sample,
                    final_coeffs,
                )

            hessian[i_atom, :, i_coord, :] = ((grads[0] - grads[1]) / (2 * self.epsilon)).reshape(
                mol.natm, 3
            )

        # symmetrize the Hessian
        hessian = (hessian + hessian.movedim((0, 1, 2, 3), (1, 0, 3, 2))) / 2

        return hessian
