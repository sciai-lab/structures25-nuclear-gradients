"""Utilities to compute KS/OF energies and OF gradients.

Mirrors the logic from :func:`~mldft.datagen.methods.label_generation.get_energies_and_gradients`
and adds a reusable context object for repeated evaluations (e.g. perturbations
around a fixed ground state).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from loguru import logger
from pyscf import dft, gto

from mldft.datagen.methods.label_generation import (
    compute_remaining_ks_energies,
    eval_density_functionals,
    validate_density_fitting,
)
from mldft.ofdft.libxc_functionals import nr_rks


@dataclass
class EnergyGradientContext:
    """Reusable context for repeated energy/gradient evaluations."""

    grad_external_energy_c: np.ndarray
    grad_external_energy_p: np.ndarray
    kinetic_and_external_energy_hamiltonian: np.ndarray
    coulomb_matrix: np.ndarray
    mol_orbital: gto.Mole
    mol_density: gto.Mole
    density_fitting: Callable
    grid: dft.Grids
    aos: np.ndarray | None
    aos_torch: torch.Tensor | None
    grid_weights: torch.Tensor | None
    xc: str
    use_torch: bool
    use_blocked: bool
    max_memory: int
    numint: dft.numint.NumInt | None = None

    def __post_init__(self):
        if self.numint is None:
            self.numint = dft.numint.NumInt()


def _compute_xc_energy_and_coeffs(
    gamma: np.ndarray,
    context: EnergyGradientContext,
) -> tuple[float, np.ndarray]:
    """Computes KS XC energy and density-fit coefficients."""
    assert gamma.ndim == 2, f"Expected 2D gamma, got shape {gamma.shape}."

    xc_energy = nr_rks(
        context.numint,
        context.mol_orbital,
        context.grid,
        context.xc,
        gamma,
        max_memory=context.max_memory,
    )
    logger.trace("Starting density fitting.")
    coeffs_p = context.density_fitting(gamma)
    logger.trace("Density fitting done.")

    return xc_energy, coeffs_p  # type: ignore


def _compute_of_energies_and_gradients(
    coeffs_p: np.ndarray,
    context: EnergyGradientContext,
) -> tuple[dict, dict]:
    """Computes OF energies and gradients from density coefficients."""
    of_energy: dict = {}
    gradient: dict = {}

    logger.trace("Starting evaluation of functionals on the grid.")
    results = eval_density_functionals(
        coeffs_p,
        functionals=[context.xc, "APBE"],
        mol=context.mol_density,
        grid=context.grid,
        aos=context.aos,
        aos_torch=context.aos_torch,
        grid_weights=context.grid_weights,
        use_torch=context.use_torch,
        use_blocked=context.use_blocked,
        max_memory=context.max_memory,
    )
    logger.trace("Finished evaluation of functionals on the grid.")
    coeffs_total = coeffs_p
    of_energy["kinapbe"], gradient["kinapbe"] = results["APBE"]
    of_energy["xc"], gradient["xc"] = results[context.xc]
    gradient["hartree"] = context.coulomb_matrix @ coeffs_total
    of_energy["hartree"] = coeffs_total @ gradient["hartree"] / 2
    gradient["external"] = context.grad_external_energy_p
    of_energy["external"] = coeffs_total @ context.grad_external_energy_p
    gradient["effective_potential"] = gradient["hartree"] + gradient["xc"] + gradient["external"]

    return of_energy, gradient


def get_energies_and_gradients_using_context(
    gamma: np.ndarray,
    context: EnergyGradientContext,
    e_nuc_nuc: float,
    e_hartree_c: float | None,
    number_of_iteration: int,
    iteration: int,
    sanity_check_fit: bool = False,
) -> tuple[dict, dict, np.ndarray, dict]:
    """Evaluates KS/OF energies and OF gradients using a reusable context."""
    if isinstance(gamma, list):
        gamma = np.array(gamma)

    logger.trace("Computing Kohn-Sham XC energy.")
    num_electrons = context.mol_density.nelectron
    ks_energy: dict = {}

    ks_energy["xc"], coeffs_p = _compute_xc_energy_and_coeffs(gamma, context)

    ks_energy = compute_remaining_ks_energies(
        gamma,
        context.grad_external_energy_c,
        context.kinetic_and_external_energy_hamiltonian,
        ks_energy,
        context.mol_orbital,
        e_nuc_nuc,
        e_hartree_c,
    )

    validate_density_fitting(
        coeffs_p,
        context.mol_orbital,
        context.mol_density,
        context.grid,
        gamma,
        iteration,
        number_of_iteration,
        sanity_check_fit,
        num_electrons,
    )

    of_energy, gradient = _compute_of_energies_and_gradients(coeffs_p, context)

    of_energy["electronic"] = ks_energy["electronic"]
    of_energy["total"] = of_energy["electronic"] + e_nuc_nuc
    of_energy["kinetic"] = (
        of_energy["electronic"] - of_energy["hartree"] - of_energy["external"] - of_energy["xc"]
    )

    kin_diff = np.abs(of_energy["kinetic"] - ks_energy["kinetic"])
    hartree_diff = np.abs(of_energy["hartree"] - ks_energy["hartree"])
    external_diff = np.abs(of_energy["external"] - ks_energy["external"])
    electronic_diff = np.abs(of_energy["electronic"] - ks_energy["electronic"])

    if kin_diff > 1e-3:
        logger.warning(
            "KS and OF kinetic energies differ by "
            f"{kin_diff * 1e3:.2g} mHa at iteration {iteration} "
            f"(total number of iteration={number_of_iteration})"
        )

    if hartree_diff > 1e-3:
        logger.warning(
            "KS and OF hartree energies differ by "
            f"{hartree_diff * 1e3:.2g} mHa at iteration {iteration} "
            f"(total number of iteration={number_of_iteration})"
        )

    if external_diff > 1e-3:
        logger.warning(
            "KS and OF external energies differ by "
            f"{external_diff * 1e3:.2g} mHa at iteration {iteration} "
            f"(total number of iteration={number_of_iteration})"
        )

    if electronic_diff > 1e-3:
        logger.warning(
            "KS and OF electronic energies differ by "
            f"{electronic_diff * 1e3:.2g} mHa at iteration {iteration} "
            f"(total number of iteration={number_of_iteration})"
        )

    return ks_energy, of_energy, coeffs_p, gradient
