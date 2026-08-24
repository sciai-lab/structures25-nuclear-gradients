from typing import Any

import numpy as np
import pytest
import torch
from pyscf import gto

from mldft.ml.data.components.of_data import Representation
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.geometry_optimization import (
    GeometryOptimizationNotConvergedError,
    GeomOpt,
)
from mldft.ofdft.optimizer import GradientDescent
from mldft.utils.sample_generator import SampleGenerator


def quadratic_model(mol: gto.Mole, last_coeffs=None):
    """Quadratic model for testing the geometry optimization.

    Assumes a quadratic potential between all atoms. This usually still converges for three atoms
    but fails for four or more, at least within a reasonable number of iterations.
    """
    k = 0.1
    re = 0.74
    energy = 0.0
    gradient = np.zeros(3 * mol.natm, dtype=float)

    for atom1_id in range(mol.natm):
        coords1 = mol.atom_coord(atom1_id)
        gradient1 = np.zeros(3, dtype=float)
        for atom2_id in range(atom1_id + 1, mol.natm):
            coords2 = mol.atom_coord(atom2_id)
            r = np.linalg.norm(coords1 - coords2)
            energy += k * (r - re) ** 2

            gradient1 += 2 * k * (r - re) * (coords1 - coords2) / r

        gradient[3 * atom1_id : 3 * atom1_id + 3] += gradient1

    return energy, gradient, torch.zeros(mol.nao), True


class MockSample:
    """Mock sample class for testing purposes."""

    def __init__(self, mol):
        self.mol = mol
        self.coeffs = torch.zeros(mol.nao_nr())

    def add_item(self, key: str, val: Any, representation: Representation):
        pass


class MockModelConfig:
    """Mock model configuration for testing purposes."""

    def get_sample_from_mol(self, mol):
        """Get a sample from a molecule with the appropriate transforms applied."""
        return MockSample(mol)


def test_GeometryOptimizer_with_quadratic_model(molecule_small):
    """Test the geometry optimization can actually converge on something with the quadratic
    model."""
    mol = molecule_small
    functional_factory = FunctionalFactory("hartree", "nuclear_attraction")
    model_config = MockModelConfig()
    sample = model_config.get_sample_from_mol(mol)

    if mol.natm == 1:
        with pytest.raises(ValueError):
            geom_opt = GeomOpt(sample, functional_factory, model_config, gradients=lambda x: x)
    else:
        geom_opt = GeomOpt(
            sample, functional_factory, model_config, maxsteps=100, gradients=lambda x: x
        )
        geom_opt.energy_and_gradient_calculator = quadratic_model

        optmol, _ = geom_opt.run()
        atom_coords = optmol.atom_coords()

        assert np.isclose(geom_opt.e_last, 0)
        dist_matrix = np.linalg.norm(atom_coords[:, np.newaxis] - atom_coords, axis=2)
        off_diagonal_mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
        assert np.allclose(dist_matrix[off_diagonal_mask], 0.74, atol=1e-2)


@pytest.mark.parametrize(
    "kwargs",
    [
        ({}),
        (
            {
                "convergence_energy": 1e-4,  # Eh
                "convergence_grms": 3e-3,  # Eh/Bohr
                "convergence_gmax": 4.5e-3,  # Eh/Bohr
                "convergence_drms": 1.2e-2,  # Angstrom
                "convergence_dmax": 1.8e-2,  # Angstrom
            }
        ),
    ],
)
def test_GeometryOptimizer_with_ofdft_model(ml_model_data, kwargs):
    """Test the geometry optimization can do a few cycles with an ofdft functional."""
    ml_model, data, transform, cfg = ml_model_data
    model_config = SampleGenerator(cfg, ml_model)

    mol = gto.M(atom="H 0 0 0; H 0 0 1")

    starting_coords = mol.atom_coords()
    functional_factory = model_config.get_functional_factory()
    sample = model_config.get_sample_from_mol(mol)

    if mol.natm == 1:
        with pytest.raises(ValueError):
            geom_opt = GeomOpt(sample, functional_factory, model_config)
    else:
        geom_opt = GeomOpt(
            sample,
            functional_factory,
            model_config,
            maxsteps=2,
            optimizer=GradientDescent(
                max_cycle=10, learning_rate=0.01, convergence_tolerance=1e-4
            ),
        )

        # this is necessary as this test model can return nan gradients and will fail the test
        # even when it actually works
        def test_function(i):
            try:
                optmol, _ = geom_opt.run(**kwargs)
                return optmol
            except np.linalg.LinAlgError as e:
                if i < 3:
                    return test_function(i + 1)
                else:
                    raise e

        # this is not even guaranteed to converge and definitely not in three steps, it just checks
        # that it can do a few cycles without raising other Errors
        with pytest.raises(GeometryOptimizationNotConvergedError):
            optmol = test_function(0)
            assert geom_opt.e_last != 0
            assert geom_opt.cycle > 0
            assert np.any(starting_coords != optmol.atom_coords())
