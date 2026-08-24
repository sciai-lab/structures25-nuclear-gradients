import numpy as np
import scipy
from pyscf import dft, gto, scf


def compute_fock_matrix(mol: gto.Mole, gamma: np.ndarray, xc: str) -> np.ndarray:
    """Computes the Fock matrix for a given density matrix.

    Args:
        mol: The molecule in the orbital basis.
        gamma: Density matrix, shape (n_ao, n_ao).
        xc: Exchange-correlation functional string.

    Returns:
        Fock matrix of shape (n_ao, n_ao).
    """
    h1e = scf.hf.get_hcore(mol)
    mf = dft.RKS(mol, xc=xc)
    v_eff = mf.get_veff(mol, gamma)
    return h1e + v_eff


def fock_matrix_to_coefficients(
    fock_matrix: np.ndarray,
    overlap_matrix: np.ndarray,
    num_orbitals: int | None = None,
) -> np.ndarray:
    """Solves F C = S C e and returns molecular orbital coefficients.

    Args:
        fock_matrix: Fock matrix of shape (n, n).
        overlap_matrix: Overlap matrix S of shape (n, n).
        num_orbitals: Number of lowest-energy orbitals to return. If None, returns all.

    Returns:
        MO coefficients of shape (n, num_orbitals).
    """
    fock_matrix = np.asarray(fock_matrix)
    overlap_matrix = np.asarray(overlap_matrix)

    subset = None
    if num_orbitals is not None:
        if not (1 <= num_orbitals <= fock_matrix.shape[0]):
            raise ValueError(
                f"num_orbitals must be in [1, {fock_matrix.shape[0]}], got {num_orbitals}."
            )
        subset = [0, num_orbitals - 1]

    _eigenvalues, coefficients = scipy.linalg.eigh(
        fock_matrix, overlap_matrix, subset_by_index=subset
    )
    return coefficients
