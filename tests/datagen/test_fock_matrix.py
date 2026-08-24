import numpy as np
import pytest
from pyscf import dft, gto, scf

from mldft.datagen.methods.density_fitting import (
    get_KSDFT_Hartree_potential,
    ksdft_density_matrix,
)
from mldft.datagen.methods.fock_matrix import (
    compute_fock_matrix,
    fock_matrix_to_coefficients,
)

ATOL = 1e-7


@pytest.fixture
def converged_ksdft_data(ksdft_results_small) -> tuple[gto.Mole, str, np.ndarray, np.ndarray]:
    """Converged KS data derived from ksdft_results_small."""
    results, _initialization, data_of_iteration, mol = ksdft_results_small

    data = data_of_iteration[-1]
    gamma = ksdft_density_matrix(
        data["molecular_coeffs_orbitals"], data["occupation_numbers_orbitals"]
    )
    xc = results["name_xc_functional"].decode()
    overlap_matrix = mol.intor_symmetric("int1e_ovlp")
    return mol, xc, gamma, overlap_matrix


def test_compute_fock_matrix(converged_ksdft_data) -> None:
    """Checks the Fock matrix against a manual Hartree + XC + hcore decomposition."""
    mol, xc, gamma, _ = converged_ksdft_data
    fock_matrix = compute_fock_matrix(mol, gamma, xc)

    h1e = scf.hf.get_hcore(mol)
    hartree_potential = get_KSDFT_Hartree_potential(mol, gamma)
    grid = dft.Grids(mol)
    grid.build()
    _n, _exc, xc_potential = dft.numint.NumInt().nr_rks(mol, grid, xc, gamma)
    fock_matrix_manual = h1e + hartree_potential + xc_potential

    assert fock_matrix.shape == (mol.nao, mol.nao)
    assert np.allclose(fock_matrix, fock_matrix.T, atol=ATOL, rtol=0.0)
    assert np.allclose(fock_matrix, fock_matrix_manual, atol=ATOL, rtol=0.0)


def test_fock_matrix_to_coefficients(converged_ksdft_data) -> None:
    """Checks coefficients from the generalized eigenvalue equation F C = S C e."""
    mol, xc, gamma, overlap_matrix = converged_ksdft_data
    fock_matrix = compute_fock_matrix(mol, gamma, xc)

    coefficients = fock_matrix_to_coefficients(fock_matrix, overlap_matrix)
    assert coefficients.shape == (mol.nao, mol.nao)

    ortho = coefficients.T @ overlap_matrix @ coefficients
    assert np.allclose(ortho, np.eye(mol.nao), atol=ATOL, rtol=0.0)

    eigvals = np.diag(coefficients.T @ fock_matrix @ coefficients)
    residual = fock_matrix @ coefficients - overlap_matrix @ coefficients @ np.diag(eigvals)
    assert np.allclose(residual, 0.0, atol=ATOL, rtol=0.0)

    num_orbitals = max(1, mol.nao - 1)
    reduced_coefficients = fock_matrix_to_coefficients(
        fock_matrix, overlap_matrix, num_orbitals=num_orbitals
    )
    assert reduced_coefficients.shape == (mol.nao, num_orbitals)

    with pytest.raises(ValueError, match="num_orbitals must be in"):
        fock_matrix_to_coefficients(fock_matrix, overlap_matrix, num_orbitals=0)

    with pytest.raises(ValueError, match="num_orbitals must be in"):
        fock_matrix_to_coefficients(fock_matrix, overlap_matrix, num_orbitals=mol.nao + 1)
