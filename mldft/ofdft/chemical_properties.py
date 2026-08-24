import numpy as np
import torch
from pyscf import gto, scf
from pyscf.scf.hf import dip_moment

from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import get_basis_dipole
from mldft.utils.coeffs_to_grid import coeffs_to_rho
from mldft.utils.grids import grid_setup


def get_dipole_moment(mol: gto.Mole, coeffs: torch.Tensor) -> torch.Tensor:
    r"""Calculate the dipole moment of a molecule in the OF ansatz.

    Args:
        mol: PySCF molecule object.
        coeffs: Coefficients or tensor of coefficients. Last index runs over the basis functions.

    Returns:
        Dipole moment(s) of the molecule. Last index runs over the x, y, z components.
    """
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    nuclear_dipole = charges @ coords
    nuclear_dipole = torch.as_tensor(nuclear_dipole, dtype=OFDFT_DTYPE, device=coeffs.device)

    basis_dipole = get_basis_dipole(mol)
    basis_dipole = torch.as_tensor(basis_dipole, dtype=OFDFT_DTYPE, device=coeffs.device)
    electronic_dipole = coeffs @ basis_dipole.T
    return nuclear_dipole - electronic_dipole


def get_dipole_moment_ks(mol: gto.Mole, gamma: np.array) -> np.array:
    r"""Calculate the dipole moment of a molecule in the KS ansatz.

    Args:
        mol: PySCF molecule object.
        gamma: Density matrix.

    Returns:
        Dipole moment of the molecule.
    """
    return dip_moment(mol, gamma, unit="au", verbose=0)


def get_hirshfeld_charges(
    mol: gto.Mole, coeffs: torch.Tensor, ks_basis: str = "6-31G(2df,p)"
) -> np.array:
    """Calculate Hirshfeld charges from the OF coefficients.

    The required atomic densities are obtained via the pySCF atomic initial guess which uses
    spherically averaged Hartree-Fock calculations.

    Args:
        mol: PySCF molecule object.
        coeffs: Coefficients in the untransformed basis.
        ks_basis: Basis set used for the atomic densities.

    Returns:
        Hirshfeld partial charges in the same order as the atoms in the mol object.
    """
    grid = grid_setup(mol, grid_level=3)  # seems to be enough for Hirshfeld charges

    ao = mol.eval_gto("GTOval", grid.coords)
    molecular_density = coeffs_to_rho(coeffs, ao)

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    atomic_densities = np.zeros((mol.natm, grid.weights.size))

    for i in range(mol.natm):
        atom = [[charges[i], coords[i]]]
        atom_mol = gto.M(atom=atom, spin=None, basis=ks_basis, unit="bohr")
        atom_ao = atom_mol.eval_gto("GTOval", grid.coords)

        mf = scf.RHF(atom_mol)
        atom_dm = mf.get_init_guess()
        if atom_dm.ndim == 3:
            atom_dm = atom_dm.sum(axis=0)
        atom_density = np.einsum("ri,ij,rj->r", atom_ao, atom_dm, atom_ao)
        atom_norm = np.sum(atom_density * grid.weights)
        # normalize the atomic density as the accuracy of the initial guess is not perfect
        atomic_densities[i] = atom_density * charges[i] / atom_norm

    promolecular_density = np.sum(atomic_densities, axis=0)
    weight_functions = atomic_densities / promolecular_density

    molecular_deformation_density = molecular_density - promolecular_density
    atomic_deformation_densities = weight_functions * molecular_deformation_density
    return -atomic_deformation_densities @ grid.weights
