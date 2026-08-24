from typing import Literal

import numpy as np
import scipy
import torch
from pyscf import gto

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices_torch,
)
from mldft.ofdft.basis_integrals import get_overlap_matrix, get_overlap_tensor
from mldft.ofdft.torch_functionals import eval_torch_functionals


def get_v_eff_from_gradient(
    gradient: torch.Tensor,
    target_key: str,
    coeffs: torch.Tensor | None,
    aos: torch.Tensor | None = None,
    grid_weights: torch.Tensor | None = None,
    xc: str | None = None,
) -> torch.Tensor:
    """Obtain the effective potential from the gradient of the model output.

    Note that the gradient, the coeffs and the atomic orbitals must be in the same transformation.

    Args:
        gradient: Gradient of the model output.
        target_key: Target key of the model.
        coeffs: The coefficients in the density basis. Only needed if the target key is not "kin".
        aos: Atomic orbitals evaluated on the grid. Only needed if the target key is not "kin".
        grid_weights: Weights of the grid points. Only needed if the target key is not "kin".
        xc: The exchange-correlation functional to use. Only needed for the "kin_plus_xc".

    Returns:
        The effective potential function integrated over the density basis.
    """
    if target_key != "kin":
        assert all(
            x is not None for x in [coeffs, aos, grid_weights]
        ), f"coeffs, aos, grid_weights must be provided for target_key {target_key}"

    if target_key == "kin":
        pass
    elif target_key == "kin_minus_apbe":
        gradient += eval_torch_functionals(coeffs, aos, grid_weights, ["APBE"])["APBE"][1]
    elif target_key == "kin_plus_xc":
        assert xc is not None, "xc must be provided for target_key 'kin_plus_xc'"
        gradient -= eval_torch_functionals(coeffs, aos, grid_weights, [xc])[xc][1]
    else:
        raise ValueError(f"Unknown target key {target_key}")

    v_eff = -gradient
    return v_eff


def get_gamma_from_v_eff(
    v_eff: torch.Tensor,
    mol_density: gto.Mole,
    mol_orbital: gto.Mole,
    method: Literal["eigh", "lobpcg", "scipy"] = "eigh",
) -> torch.Tensor:
    """Get the density matrix from the effective potential.

    Perform a single SCF step with the given effective potential and return the density matrix.

    Args:
        v_eff: Effective potential integrated over the density basis.
        mol_density: Molecule object in the density basis.
        mol_orbital: Molecule object in the Kohn-Sham basis.
        method: Method to use for the diagonalization of the Fock matrix.
            See :py:func:`single_ks_step`.

    Returns:
        The density matrix.
    """
    tensor_type = {"dtype": v_eff.dtype, "device": v_eff.device}

    overlap_density = torch.as_tensor(get_overlap_matrix(mol_density), **tensor_type)
    overlap_tensor = torch.as_tensor(get_overlap_tensor(mol_density, mol_orbital), **tensor_type)
    overlap_orbital = torch.as_tensor(get_overlap_matrix(mol_orbital), **tensor_type)
    kinetic_matrix = torch.as_tensor(mol_orbital.intor_symmetric("int1e_kin"), **tensor_type)
    # calculate the condition number of the overlap matrix
    # overlap_condition = torch.linalg.cond(overlap_density)
    # print(f"Condition number of the overlap matrix: {overlap_condition}")

    v_eff_coeffs = torch.linalg.inv(overlap_density) @ v_eff
    v_eff_matrix = torch.einsum("i,ijk->jk", v_eff_coeffs, overlap_tensor)

    # L = overlap_tensor.transpose(0, 2)
    # L = L.reshape(-1, L.shape[-1])
    # new_v_eff_coeffs = torch.linalg.lstsq(L, v_eff_matrix.flatten())[0]
    # l2_norm = (v_eff_coeffs - new_v_eff_coeffs) @ overlap_density @ (v_eff_coeffs - new_v_eff_coeffs)
    # assert l2_norm < 1e-15, f"l2 norm of the difference between the two v_eff_coeffs: {l2_norm}"

    torch.testing.assert_close(overlap_tensor, overlap_tensor.transpose(-2, -1))
    torch.testing.assert_close(
        v_eff_matrix, v_eff_matrix.T
    ), "Effective potential matrix is not symmetric"
    assert torch.allclose(kinetic_matrix, kinetic_matrix.T), "Kinetic matrix is not symmetric"

    fock = kinetic_matrix + v_eff_matrix
    gamma = single_ks_step(fock, overlap_orbital, mol_density.nelectron, method)

    return torch.as_tensor(gamma, **tensor_type)


def single_ks_step(
    fock: torch.Tensor,
    overlap_orbital: torch.Tensor,
    n_electron: int,
    method: Literal["lobpcg", "eigh", "scipy"],
) -> torch.Tensor:
    """Perform a single Kohn-Sham step.

    To be differentiable using backpropagation using "lobpcg" and "eigh", the generalized
    eigenvalue problem is reduced to a standard eigenvalue problem by diagonalizing the orbital
    overlap matrix.

    Args:
        fock: Fock matrix.
        overlap_orbital: Overlap matrix in the Kohn-Sham basis.
        n_electron: Number of electrons.
        method: Method to use for the diagonalization of the Fock matrix:

            * "eigh": torch's eigendecomposition of a symmetric/hermitian matrix.
            * "lobpcg": torch's LOBPCG (locally optimal block preconditioned conjugate gradient).
              This has the option to only compute the smallest eigenvalues, but seems to be
              much slower than "eigh", and sometimes fails to converge.
            * "scipy": scipy's `eigh` can solve the generalized eigenvalue problem, but does not
              support autograd.

    Returns:
        The density matrix.
    """
    tensor_type = {"dtype": fock.dtype, "device": fock.device}
    assert n_electron % 2 == 0, "Number of electrons is not even"
    torch.testing.assert_close(fock, fock.T, msg="Fock matrix is not symmetric")

    # orthonormalization
    _, transform_inv = natural_reparametrization_matrices_torch(overlap_orbital)
    transformed_fock = transform_inv @ fock @ transform_inv.T

    if method == "scipy":
        assert not fock.requires_grad, "scipy.linalg.eigh does not support autograd"
        epsilons, molecular_orbitals = scipy.linalg.eigh(fock, overlap_orbital)

        occupation = np.zeros(epsilons.shape)
        occupation[: n_electron // 2] = 2
        assert sum(occupation) == n_electron, "Number of electrons is not conserved"

        gamma = ksdft_density_matrix(molecular_orbitals, occupation)
        gamma = torch.as_tensor(gamma, **tensor_type)

    elif method == "lobpcg":
        epsilons, transformed_mo = torch.lobpcg(transformed_fock, k=n_electron // 2, largest=False)
        molecular_orbitals = transform_inv.T @ transformed_mo
        gamma = 2 * molecular_orbitals @ molecular_orbitals.T

    elif method == "eigh":
        epsilons, transformed_mo = torch.linalg.eigh(transformed_fock)
        molecular_orbitals = transform_inv.T @ transformed_mo

        occupation = torch.zeros_like(epsilons)
        occupation[: n_electron // 2] = 2
        gamma = ksdft_density_matrix(molecular_orbitals, occupation)

    else:
        raise ValueError(f"Unknown diagonalization method {method}")

    return gamma


if __name__ == "__main__":
    from mldft.utils.molecules import build_mol_with_even_tempered_basis

    ethanol_structure = [
        ["H", (1.8853, -0.0401, 1.0854)],
        ["C", (1.2699, -0.0477, 0.1772)],
        ["H", (1.5840, 0.8007, -0.4449)],
        ["H", (1.5089, -0.9636, -0.3791)],
        ["C", (-0.2033, 0.0282, 0.5345)],
        ["H", (-0.4993, -0.8287, 1.1714)],
        ["H", (-0.4235, 0.9513, 1.1064)],
        ["O", (-0.9394, 0.0157, -0.6674)],
        ["H", (-1.8540, 0.0626, -0.4252)],
    ]
    mol_orbital_ = gto.M(atom=ethanol_structure, unit="angstrom", basis="6-31G(2df,p)")
    mol_density_ = build_mol_with_even_tempered_basis(mol_orbital_)

    for i in range(100):
        v_eff_ = torch.rand(mol_density_.nao)
        gamma_ = get_gamma_from_v_eff(v_eff_, mol_density_, mol_orbital_)
