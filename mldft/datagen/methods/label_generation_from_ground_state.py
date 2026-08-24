"""Generates labels from the ground state of the KSDFT calculation.

Instead of perturbing the system during SCF iterations, this module starts from the converged
ground state and applies perturbations to the Fock matrix to generate nearby non-equilibrium
densities as additional training data.

Refer to :mod:`mldft.datagen.methods.label_generation` for more details and especially the
mathematical background.
"""

import os

import numpy as np
import torch
from loguru import logger
from pyscf import dft, gto, scf

from mldft.datagen.methods.density_fitting import (
    _check_data_format,
    get_density_fitting_function,
    ksdft_density_matrix,
)
from mldft.datagen.methods.energies_and_gradients import (
    EnergyGradientContext,
    get_energies_and_gradients_using_context,
)
from mldft.datagen.methods.fock_matrix import (
    compute_fock_matrix,
    fock_matrix_to_coefficients,
)
from mldft.datagen.methods.label_generation import (
    get_data_dict,
    ks_energy_names,
    of_energy_names,
    of_gradient_names,
)
from mldft.datagen.methods.perturbation import (
    build_distribution_perturbation_sampler,
)
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
    get_overlap_tensor,
)
from mldft.ofdft.torch_functionals import str_to_torch_functionals
from mldft.utils.grids import grid_setup


def _get_ground_state_data(
    data_of_iteration: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extracts data from the last SCF iteration (ground state)."""
    data_gs = data_of_iteration[-1]
    mo_coeff_gs = np.asarray(data_gs["molecular_coeffs_orbitals"])
    mo_occ_gs = np.asarray(data_gs["occupation_numbers_orbitals"])
    gamma_gs = ksdft_density_matrix(mo_coeff_gs, mo_occ_gs)
    e_hartree_gs = data_gs["coulomb_energy"]
    return mo_coeff_gs, mo_occ_gs, gamma_gs, e_hartree_gs


def calculate_labels_with_ground_state_perturbations(
    results: dict,
    initialization: dict,
    data_of_iteration: list[dict],
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    perturbation_cfg: dict,
    density_fitting_method: str = "hartree+external_mofdft",
) -> dict:
    """Calculates labels by perturbing the converged ground state Fock matrix.

    Starts from the KSDFT ground state (last SCF iteration) and generates
    ``num_perturbations`` perturbed densities by adding random noise to
    the Fock matrix (sampled in the density basis). For each perturbed state the same
    energy and gradient labels as in :mod:`mldft.datagen.methods.label_generation` are
    computed.

    The kinetic-energy gradient label follows from the optimality condition at the
    ground state with a perturbation :math:`\\delta v` (in the density basis):

    .. math::

        \\frac{\\partial T}{\\partial p}
        = -v_{\\mathrm{eff}}(p^*) - S_{\\mathrm{density}}\\,\\delta v

    where :math:`p^*` is the ground-state density and
    :math:`S_{\\mathrm{density}}` is the density-basis overlap.

    Args:
        results: Parameters of the KSDFT calculation (from .chk file).
        initialization: First-iteration density matrix and total energy (unused here,
            kept for API compatibility with label_generation.py).
        data_of_iteration: List of per-SCF-iteration data dicts.
        mol_orbital_basis: Molecule in the orbital basis.
        mol_density_basis: Molecule in the density basis.
        perturbation_cfg: Distribution perturbation config.
        density_fitting_method: Density fitting method string.

    Returns:
        Dict of all label arrays (same schema as label_generation.get_data_dict).
        Perturbed samples come first; the unperturbed ground-state sample is last.
    """
    assert len(data_of_iteration) > 0, "No SCF iterations found in checkpoint."
    _check_data_format(results, data_of_iteration, mol_orbital_basis)
    assert gto.mole.is_same_mol(mol_density_basis, mol_orbital_basis, cmp_basis=False)

    mo_coeff_gs, mo_occ_gs, gamma_gs, e_hartree_gs = _get_ground_state_data(data_of_iteration)
    num_orbitals = mo_coeff_gs.shape[-1]

    num_perturbations = perturbation_cfg["num_perturbations"]
    num_samples = num_perturbations + 1  # +1 for the unperturbed ground state

    xc: str = results["name_xc_functional"].decode()
    use_torch = xc in str_to_torch_functionals
    grid = grid_setup(
        mol_density_basis,
        results["grid_level"],
        results["prune_method"].decode(),
    )
    grid_weights_np = np.asarray(grid.weights)
    max_memory = int(os.getenv("PYSCF_MAX_MEMORY", 4000))
    estimated_xc_mem_usage = 8 * mol_density_basis.nao * grid_weights_np.size * 8 / 1048576
    use_blocked = estimated_xc_mem_usage > max_memory

    coulomb_matrix = get_coulomb_matrix(mol_density_basis)
    grad_external_energy_c = get_nuclear_attraction_matrix(mol_orbital_basis)
    grad_external_energy_p = get_nuclear_attraction_vector(mol_density_basis)
    kin_ext_hamiltonian = scf.hf.get_hcore(mol_orbital_basis)
    density_fitting = get_density_fitting_function(
        density_fitting_method,
        mol_orbital_basis,
        mol_density_basis,
        coulomb_matrix,
        grad_external_energy_c,
        grad_external_energy_p,
        max_memory=max_memory,
    )
    e_nuc_nuc = mol_orbital_basis.energy_nuc()
    num_coeffs_p = mol_density_basis.nao_nr()

    fock_gs = compute_fock_matrix(mol_orbital_basis, gamma_gs, xc)
    overlap_orbital = mol_orbital_basis.intor_symmetric("int1e_ovlp")
    overlap_density = get_overlap_matrix(mol_density_basis)
    overlap_tensor = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
    perturbation_sampler = build_distribution_perturbation_sampler(
        perturbation_cfg, overlap_density
    )

    aos = aos_torch = grid_weights = None
    if not use_blocked:
        aos = dft.numint.eval_ao(mol_density_basis, grid.coords, deriv=1)
        if use_torch:
            aos_torch = torch.as_tensor(aos, dtype=torch.float64)
    if use_torch:
        grid_weights = torch.as_tensor(grid_weights_np, dtype=torch.float64)

    of_coeffs = np.zeros((num_samples, num_coeffs_p))
    of_gradients = {name: np.zeros((num_samples, num_coeffs_p)) for name in of_gradient_names}
    ks_energies = {name: np.zeros(num_samples) for name in ks_energy_names}
    of_energies = {name: np.zeros(num_samples) for name in of_energy_names}
    has_energy_label = np.ones(num_samples, dtype=bool)

    energy_gradient_context = EnergyGradientContext(
        grad_external_energy_c=grad_external_energy_c,
        grad_external_energy_p=grad_external_energy_p,
        kinetic_and_external_energy_hamiltonian=kin_ext_hamiltonian,
        coulomb_matrix=coulomb_matrix,
        mol_orbital=mol_orbital_basis,
        mol_density=mol_density_basis,
        density_fitting=density_fitting,
        grid=grid,
        aos=aos,
        aos_torch=aos_torch,
        grid_weights=grid_weights,
        xc=xc,
        use_torch=use_torch,
        use_blocked=use_blocked,
        max_memory=max_memory,
    )

    logger.debug("Computing ground state labels.")
    ks_energy, of_energy, of_coeff, gradient = get_energies_and_gradients_using_context(
        gamma=gamma_gs,
        context=energy_gradient_context,
        e_nuc_nuc=e_nuc_nuc,
        e_hartree_c=e_hartree_gs,
        number_of_iteration=num_samples,
        iteration=num_samples - 1,
    )
    of_coeffs[-1] = of_coeff
    for name in gradient:
        if name in of_gradients:
            of_gradients[name][-1] = gradient[name]
    for name in ks_energies:
        ks_energies[name][-1] = ks_energy.get(name, 0.0)
    for name in of_energies:
        of_energies[name][-1] = of_energy.get(name, 0.0)

    v_eff_gs = gradient["effective_potential"]
    of_gradients["kinetic"][-1] = -v_eff_gs

    if not results["converged"]:
        logger.warning("KS calculation did not converge; ground state labels may be inaccurate.")

    for i in range(num_perturbations):
        # Sample perturbation in density basis; project back to the Fock matrix
        delta_v = perturbation_sampler.sample(i, num_coeffs_p)
        delta_F = np.einsum("ijk,i->jk", overlap_tensor, delta_v, optimize=True)
        fock_perturbed = fock_gs + delta_F

        # Solve F_pert C = S C e for new MO coefficients
        mo_coeff_pert = fock_matrix_to_coefficients(
            fock_perturbed,
            overlap_orbital,
            num_orbitals,
        )
        gamma_pert = ksdft_density_matrix(mo_coeff_pert, mo_occ_gs)

        logger.debug(f"Computing labels for perturbation {i + 1}/{num_perturbations}")
        ks_energy, of_energy, of_coeff, gradient = get_energies_and_gradients_using_context(
            gamma=gamma_pert,
            context=energy_gradient_context,
            e_nuc_nuc=e_nuc_nuc,
            e_hartree_c=None,
            number_of_iteration=num_samples,
            iteration=i,
        )

        of_coeffs[i] = of_coeff
        for name in gradient:
            if name in of_gradients:
                of_gradients[name][i] = gradient[name]

        of_gradients["kinetic"][i] = -v_eff_gs - overlap_density @ delta_v

        for name in ks_energies:
            ks_energies[name][i] = ks_energy.get(name, 0.0)
        for name in of_energies:
            of_energies[name][i] = of_energy.get(name, 0.0)

    return get_data_dict(
        ks_energies,
        of_energies,
        of_coeffs,
        of_gradients,
        has_energy_label,
        e_nuc_nuc,
    )
