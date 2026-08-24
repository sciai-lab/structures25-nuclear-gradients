import pytest
import torch
from pyscf.scf.hf import dip_moment

from mldft.datagen.methods.density_fitting import (
    density_fitting_mol,
    ksdft_density_matrix,
)
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.ofdft.chemical_properties import (
    get_dipole_moment,
    get_dipole_moment_ks,
    get_hirshfeld_charges,
)
from mldft.utils.molecules import build_mol_with_even_tempered_basis


def test_get_dipole_moment(ksdft_results_small):
    """Check the dipole moment in the KS ansatz versus the OF ansatz."""
    results, initialisation, data_of_iteration, mol_orbital_basis = ksdft_results_small

    mo_coeffs = results["molecular_coeffs_orbitals"]
    mo_occ = results["occupation_numbers_orbitals"]
    gamma = ksdft_density_matrix(mo_coeffs, mo_occ)

    beta = 1.05  # small beta is crucial for higher accuracy
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital_basis, beta)
    coeffs = density_fitting_mol(gamma, mol_orbital_basis, mol_rho, method="hartree+external")
    coeffs = torch.as_tensor(coeffs, dtype=OFDFT_DTYPE)

    dipole_orbital = get_dipole_moment_ks(mol_orbital_basis, gamma)
    dipole_rho = get_dipole_moment(mol_rho, coeffs)
    dipole_pyscf = dip_moment(mol_orbital_basis, gamma, unit="au", verbose=0)

    assert dipole_orbital == pytest.approx(dipole_pyscf)
    assert dipole_orbital == pytest.approx(dipole_rho, abs=2e-4, rel=5e-3)


def test_get_hirshfeld_charges(molecule_medium_with_density_basis):
    """Check that the Hirshfeld partial charges sum up to zero."""
    mol = molecule_medium_with_density_basis

    g = torch.Generator()
    g.manual_seed(42)
    coeffs = torch.rand(mol.nao, generator=g)
    normalization_vector = get_normalization_vector(mol)
    coeffs *= mol.nelectron / (coeffs @ normalization_vector)

    partial_charges = get_hirshfeld_charges(mol, coeffs)
    assert sum(partial_charges) == pytest.approx(0, abs=1e-4)
