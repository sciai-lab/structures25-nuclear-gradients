from collections.abc import Callable

import numpy as np
import pytest
import torch
from pyscf.hessian import thermo

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.energies import Energies
from mldft.ofdft.nuclear_hessian import NumericalHessian, SemiNumericalHessian
from mldft.ofdft.optimizer import Optimizer, TorchOptimizer
from mldft.ofdft.run_density_optimization import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype

FORCE_CONSTANT = torch.tensor(0.4, dtype=torch.float64)


class MockOptimizer(Optimizer):
    """A mock optimizer that returns a simple harmonic potential for H2 molecule with the set force
    constant and equilibirum distance."""

    # H2  equilibirum in Bohr
    re = torch.tensor(1.3984, dtype=torch.float64)
    k = FORCE_CONSTANT

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable[[OFData], tuple[Energies, torch.Tensor]],
        callback: Callable | None = None,
        disable_pbar: bool = False,
        **kwargs,
    ) -> tuple[Energies, bool]:
        """Optimize the positions of the nuclei in the sample using a harmonic potential."""
        mass = torch.as_tensor(sample.mol.atom_mass_list(isotope_avg=False))
        reduced_mass = mass[0] * mass[1] / mass.sum()
        r = torch.linalg.norm(sample.pos[0] - sample.pos[1])
        energy = (self.re - r) ** 2 * self.k / 2 * reduced_mass
        return Energies(nuclear_repulsion=energy), True


@set_default_torch_dtype(torch.float64)
def test_model_hessian(ml_model_data):
    """This test checks that the numerical Hessian method returns the correct Hessian for a simple
    model where the force constant is known by construction."""
    ml_model, data, transforms, cfg = ml_model_data
    sample_generator = SampleGenerator(cfg, ml_model)
    func_factory = sample_generator.get_functional_factory()

    optimizer = MockOptimizer()

    hessian_method = NumericalHessian(
        sample_generator, functional_factory=func_factory, optimizer=optimizer, epsilon=1e-6
    )
    sample = data.val_set[0]
    sample = transforms(sample)
    coeffs = sample.coeffs.to(
        device=sample.transformation_matrix.device,
        dtype=sample.transformation_matrix.dtype,
    )
    sample.coeffs = coeffs
    hessian = hessian_method(sample, coeffs)
    mass = sample.mol.atom_mass_list(isotope_avg=False)
    res = thermo.harmonic_analysis(sample.mol, hessian.numpy(), imaginary_freq=False, mass=mass)

    assert np.allclose(res["force_const_au"], FORCE_CONSTANT)


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("HessianClass", [(NumericalHessian), (SemiNumericalHessian)])
def test_hessians(ml_model_data, HessianClass):
    """This test only check that the Hessian method actually runs through and returns a Hessian,
    but does not check the correctness of the Hessian itself."""
    ml_model, data, transforms, cfg = ml_model_data

    sample_generator = SampleGenerator(cfg, ml_model)
    func_factory = sample_generator.get_functional_factory()
    optimizer = TorchOptimizer(
        torch.optim.SGD,
        max_cycle=2,
        convergence_tolerance=10,
        lr=0.01,
    )

    hessian_method = HessianClass(
        sample_generator, functional_factory=func_factory, optimizer=optimizer
    )

    sample = data.val_set[0]
    sample = transforms(sample)
    sample.coeffs = sample.coeffs.to(
        device=sample.transformation_matrix.device,
        dtype=sample.transformation_matrix.dtype,
    )
    coeffs = transform_tensor_with_sample(
        sample, sample.coeffs, Representation.VECTOR, invert=True
    )

    hessian = hessian_method(sample, coeffs)
    assert hessian.shape == (sample.mol.natm, sample.mol.natm, 3, 3)
