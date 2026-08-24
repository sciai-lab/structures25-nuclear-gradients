r"""This module calculates the value and gradient label from the KSDFT simulation results. The label generation
is described in [M-OFDFT]_, Supplementary Section `A.4 <https://arxiv.org/html/2309.16578v2#A1.SS4>`_ and
[STRUCTURES25]_, Supplementary Section `S.1.4 <https://arxiv.org/html/2503.00443v2#Ax1.SS1.SSS4>`_.

Coefficients of the orbitals and the density in the orbital basis are provided by the KSDFT calculation. The energies calculated include:

- Kinetic energy (:math:`E_{kin}(C) = T_s(C)`),
- Hartree energy (:math:`E_{H}(C)`),
- Exchange-correlation energy (:math:`E_{XC}(C)`),
- External energy (:math:`E_{ext}(C)`),
- Atomic nuclei repulsion energy (:math:`E_{nuc-nuc}`),
- Total energy (:math:`E_{tot}(C) = T_S(C) + E_H(C) + E_{XC}(C) + E_{ext}(C) + E_{nuc-nuc}`).

The coefficients of the basis functions in the density basis are determined by density fitting. From these, :math:`E_{H}(\mathbf{p})`, :math:`E_{XC}(\mathbf{p})`, and :math:`E_{ext}(\mathbf{p})` can be calculated, which now depend on the fitted density. The nuclear repulsion energy (:math:`E_{nuc-nuc}`) remains constant.

The label for the kinetic energy in these coordinates is indirectly calculated by:

.. math::

    T_S(\mathbf{p}) = T_S(C) + E_{H}(C) + E_{XC}(C) + E_{ext}(C) - E_{H}(\mathbf{p}) + E_{XC}(\mathbf{p}) + E_{ext}(\mathbf{p}).

We cannot calculate the gradient of the kinetic energy directly from this equation as the density :math:`p` depends on
the coefficients in the orbital basis :math:`C` in a nontrivial way.
Instead, we make use of the minimization procedure in the ksdft calculation. Each iteration the energy is minimized as follows:

.. math::

    \phi^{k} = \underset{\{\phi_i\}_{i=1}^n \text{orthonormal}}{\text{argmin}} \langle \psi_{\mathbf{\phi}} | \hat T_S | \psi_{\mathbf{\phi}} \rangle + \sum_{k'<k} \pi^{(k')}_k V_{eff}^{k'}[\rho_{[\mathbf{\phi}]}]

Where :math:`\pi^{(i)}_k` are the DIIS coefficients (see [M-OFDFT]_, Supplementary Section `A.4.1 Eq. S67 <https://arxiv.org/html/2309.16578v2#A1.E67>`_ ) of the KSDFT calculation and

.. math::

    V_{eff}^{k'}[\rho_{[\mathbf{\phi}]}] = \int \rho_{[\mathbf{\phi}]}(\mathbf{r}) V_{eff[\rho_{[\mathbf{\phi}^{k'}]}]}(r)dr

is the effective potential of the :math:`k'`-th iteration integrated over the density of the new density.
This is solved using Laplace multipliers:

.. math::

    \frac{\delta T_S[\rho_{[\mathbf{\phi}^{k}]}]}{\delta \rho}(\mathbf{r}) + \sum_{k'<k} \pi^{(k')}_k V_{eff}^{k'}(\mathbf{r}) = \mu^{(k)}

The projected gradient of the kinetic energy is then calculated as the DIIS weighted average over the projected last few effective potentials:

.. math::

    \nabla_\mathbf{p} T_S(\mathbf{p}) &= \int \frac{\delta T_S[\rho_{[\mathbf{\phi}^{k}]}]}{\delta \rho}(\mathbf{r}) \mathbf{\omega}(\mathbf{r}) d\mathbf{r} = -\int\sum_{k'<k} \pi^{(k')}_k V_{eff}^{k'}(\mathbf{r}) \mathbf{\omega}(\mathbf{r}) +\mu^{(k)}\mathbf{\omega}(\mathbf{r})d\mathbf{r} = -{\mathbf{v}}_{eff\{\mathbf{p}^{k'}\}_{k'< k}}+\mu^{(k)}\mathbf{w}\\
    \left( \textbf{I}-\frac{\textbf{w}^{(d)}{\textbf{w}^{(d)}}^T}{{\textbf{w}^{(d)}}^T
        \textbf{w}^{(d)}}\right) \nabla_\mathbf{p} T_S(\mathbf{p}) &= -\left( \textbf{I}-\frac{\textbf{w}^{(d)}{\textbf{w}^{(d)}}^T}{{\textbf{w}^{(d)}}^T
        \textbf{w}^{(d)}}\right){\mathbf{v}}_{eff\{\mathbf{p}^{k'}\}_{k'< k}}\\
    \mathbf{v}_{eff}(\mathbf{p}) &= \nabla_\mathbf{p} (E_{H}(\mathbf{p}) + E_{XC}(\mathbf{p}) + E_{ext}(\mathbf{p}))
"""

import os
from typing import Callable

import numpy as np
import torch
from loguru import logger
from pyscf import dft, gto, scf

from mldft.datagen.methods.density_fitting import (
    _check_data_format,
    get_density_fitting_function,
    get_KSDFT_Hartree_potential,
    ksdft_density_matrix,
)
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData
from mldft.ml.data.components.of_data import Representation as Rep
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_normalization_vector,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
)
from mldft.ofdft.inverse_kohn_sham import get_gamma_from_v_eff, get_v_eff_from_gradient
from mldft.ofdft.libxc_functionals import eval_libxc_functionals, nr_rks
from mldft.ofdft.torch_functionals import (
    eval_torch_functionals,
    eval_torch_functionals_blocked,
    str_to_torch_functionals,
)
from mldft.utils.grids import compute_l1_norm_orbital_vs_density_basis, grid_setup
from mldft.utils.instantiators import instantiate_model
from mldft.utils.multiprocess import get_memory_usage_in_mb

ks_energy_names = ["kinetic", "hartree", "xc", "external", "electronic", "total"]
of_energy_names = ks_energy_names + ["kinapbe"]
of_gradient_names = ["effective_potential", "kinetic", "kinapbe", "hartree", "xc", "external"]


def diis_weighted_average(
    iteration: int, diis_coeffs: np.ndarray, effective_potential_p: np.ndarray
) -> np.ndarray:
    r"""Calculates the DIIS weighted average of the effective potential needed for the calculation
    of the gradient label.

    .. math::

        {v}_{eff\{p^{k'}\}_{k'< k}} = \sum_{k'< k} \pi^{(k)}_{k'} v_{eff_{p^{(k')}}}

    :math:`\pi^{(k)}_{k'}` are the DIIS - coefficients of the KSDFT - calculation.

    Args:

        iteration: the iteration of which the weighted average of the effective potential should be computed.
            i=0 is the first iteration, not the initial guess.
        diis_coeffs: the DIIS coefficients of this iteration (d)
        effective_potential_p: the effective potentials of every iteration step (n_of_basis_functions)

    Returns:
        effective_potential_p_k: the DIIS weighted average of the effective potential or any other tensor that is
        computed for every iteration and is of the shape (n_iterations, ...)

    Raises:
        ValueError: if diis_c is longer then iterations have been computed so far
    """
    n_diis_coeffs = diis_coeffs.shape[0]
    if n_diis_coeffs > iteration + 1:
        raise ValueError(
            "There are more diis_coefficients then iterations have been computed so far"
        )
    effective_potential_p_k = np.einsum(
        "i,i...->...",
        diis_coeffs,
        effective_potential_p[iteration - n_diis_coeffs + 1 : iteration + 1],
        optimize=True,
        # just take the effective potentials of the last"n_diis_coeffs" iteration to be summed up
    )
    return effective_potential_p_k


def eval_density_functionals(
    coeffs: np.ndarray,
    functionals: list[str],
    mol: gto.Mole,
    grid: dft.Grids,
    aos: np.ndarray | None,
    aos_torch: torch.Tensor | None,
    grid_weights: torch.Tensor | None,
    use_torch: bool,
    use_blocked: bool,
    max_memory: float,
) -> dict:
    """Evaluate the classical density functionals.

    Args:
        coeffs: The coefficients of the basis functions in the density basis.
        functionals: The list of functionals to evaluate.
        mol: The molecule in the density basis.
        grid: The integration grid for the functionals.
        aos: The atomic orbitals.
        aos_torch: The atomic orbitals as a torch tensor.
        grid_weights: The weights of the grid.
        use_torch: Whether to use the torch implementation of the functionals.
        use_blocked: Whether to use the blocked implementation of the torch functionals.
        max_memory: The maximum memory for the functionals

    Returns:
        A dictionary with the functionals as keys and tuples of the energies and gradients as values.
    """
    if use_torch and not use_blocked:
        assert aos_torch is not None, f"aos_torch must be provided if {use_torch=}"
        assert grid_weights is not None, f"grid_weights must be provided if {use_torch=}"
    if not use_torch:
        assert aos is not None, f"aos must be provided if {use_torch=}"

    mem_used = get_memory_usage_in_mb()
    mem_available = max_memory - mem_used
    estimated_xc_mem_usage = 8 * mol.nao * grid.weights.size * 8 / 1048576  # MB
    use_blocked = mem_available < estimated_xc_mem_usage
    if use_torch:
        coeffs_torch = torch.tensor(coeffs, dtype=torch.float64, requires_grad=True)
        if use_blocked:
            max_memory = max(mem_available, 100)
            result = eval_torch_functionals_blocked(
                mol=mol,
                grid=grid,
                coeffs=coeffs_torch,
                functionals=functionals,
                pre_computed_aos=aos_torch,
                max_memory=max_memory,
            )
        else:
            result = eval_torch_functionals(coeffs_torch, aos_torch, grid_weights, functionals)
        result = {k: (e.numpy(force=True), g.numpy(force=True)) for k, (e, g) in result.items()}
    else:
        result = {}
        functionals_libxc = ["GGA_K_APBE" if f == "APBE" else f for f in functionals]
        for functional, functional_libxc in zip(functionals, functionals_libxc):
            # libxc functionals are evaluated separately to obtain all gradients
            result[functional] = eval_libxc_functionals(coeffs, functional_libxc, grid, aos, 1)

    return result


def compute_remaining_ks_energies(
    gamma: np.ndarray,
    grad_external_energy_c: np.ndarray,
    kinetic_and_external_energy_hamiltonian: np.ndarray,
    ks_energy: dict,
    mol_orbital: gto.Mole,
    e_nuc_nuc: float,
    e_hartree_c: float | None,
):
    """Computes the other Kohn-Sham energies.
    Args:
        gamma: The density matrix in the orbital basis.
        grad_external_energy_c: The gradient of the nuclear attraction energy in the orbital basis.
        kinetic_and_external_energy_hamiltonian: The one-electron Hamiltonian.
        ks_energy: The Kohn-Sham energies.
        mol_orbital: The molecule in the orbital basis.
        e_nuc_nuc: The nuclear-nuclear repulsion energy.
        e_hartree_c: The Hartree energy in the orbital basis (optional).

    Returns:
        The updated Kohn-Sham energies.
    """
    logger.trace("Computing other Kohn-Sham energies.")
    if gamma.ndim == 2:
        ks_energy["external"] = np.einsum("ij,ji->", gamma, grad_external_energy_c, optimize=True)
    else:
        gamma = gamma[0] + gamma[1]
        ks_energy["external"] = np.einsum("ij,ji->", gamma, grad_external_energy_c, optimize=True)
    # If the Hartree energy is not provided, we calculate it from the density matrix (expensive)
    if e_hartree_c is None:
        grad_hartree_energy_c = get_KSDFT_Hartree_potential(
            mol_orbital, gamma
        )  # for hatree term it should also work if we just add the gammas?
        ks_energy["hartree"] = (
            1 / 2 * np.einsum("ij,ji->", grad_hartree_energy_c, gamma, optimize=True)
        )
    else:
        ks_energy["hartree"] = e_hartree_c
    kin_ext = np.einsum(
        "ij,ji->", kinetic_and_external_energy_hamiltonian, gamma, optimize=True
    )  # also for the kin term
    ks_energy["kinetic"] = kin_ext - ks_energy["external"]
    ks_energy["electronic"] = (
        ks_energy["kinetic"] + ks_energy["hartree"] + ks_energy["external"] + ks_energy["xc"]
    )
    ks_energy["total"] = ks_energy["electronic"] + e_nuc_nuc
    return ks_energy


def validate_density_fitting(
    coeffs_p_local: np.ndarray,
    mol_orbital: gto.Mole,
    mol_density: gto.Mole,
    grid: dft.Grids,
    gamma: np.ndarray,
    iteration: int,
    number_of_iteration: int,
    sanity_check_fit: bool,
    n_elec_target: int,
):
    """Check fitted electron counts and optionally compute L1 fit errors."""
    assert coeffs_p_local.ndim == 1
    assert gamma.ndim == 2
    fitted = coeffs_p_local @ get_normalization_vector(mol_density)
    logger.debug(f"Electron number: {np.sum(fitted)} (should be {n_elec_target})")
    if np.abs((np.sum(fitted) - n_elec_target) / n_elec_target) > 1e-3:
        logger.warning(
            f"Fitted electron number is off by {np.sum(fitted) - n_elec_target:.2g} at iteration {iteration} (total number of iteration={number_of_iteration})"
        )
    if sanity_check_fit:
        l1_error = compute_l1_norm_orbital_vs_density_basis(
            mol_orbital, mol_density, grid, gamma, coeffs_p_local
        )
        l1_error_per_electron = l1_error / n_elec_target
        logger.debug(f"L1 error per electron: {l1_error_per_electron}")
        if l1_error_per_electron > 1e-1:
            logger.warning(
                f"L1 error per electron is {l1_error_per_electron:.2g} at iteration {iteration} (total number of iteration={number_of_iteration})"
            )


def get_energies_and_gradients(
    gamma: np.ndarray,
    grad_external_energy_c: np.ndarray,
    grad_external_energy_p: np.ndarray,
    kinetic_and_external_energy_hamiltonian: np.ndarray,
    coulomb_matrix: np.ndarray,
    e_nuc_nuc: float,
    e_hartree_c: float | None,
    mol_orbital: gto.Mole,
    mol_density: gto.Mole,
    density_fitting: callable,
    grid: dft.Grids,
    aos: np.ndarray | None,
    aos_torch: torch.Tensor | None,
    grid_weights: torch.Tensor | None,
    xc: str,
    use_torch: bool,
    use_blocked: bool,
    max_memory: float,
    number_of_iteration: int,
    iteration: int,
    sanity_check_fit: bool = False,
) -> tuple[dict, dict, np.ndarray, dict]:
    """Calculates the energies and gradients in the orbital and density basis.
    The gradient of the kinetic energy can't be calculated directly as the previous effective
    potentials are needed (for the formulas for the KS and OF energies and gradients see
    [M-OFDFT]_, Supplemmentary Section `A.3.1 <https://arxiv.org/html/2309.16578v2#A1.SS3.SSS1>`_,
    `A.3.2 <https://arxiv.org/html/2309.16578v2#A1.SS3.SSS2>`_,
    `A.4.2 <https://arxiv.org/html/2309.16578v2#A1.SS4.SSS2>`_ and
    `A.4.3 <https://arxiv.org/html/2309.16578v2#A1.SS4.SSS3>`_).

    Args:
        gamma: The density matrix in the orbital basis.
        grad_external_energy_c: The gradient of the nuclear attraction energy in the orbital basis.
        grad_external_energy_p: The gradient of the nuclear attraction energy in the density basis.
        kinetic_and_external_energy_hamiltonian: The one-electron Hamiltonian.
        coulomb_matrix: The Coulomb matrix (density basis).
        e_nuc_nuc: The nuclear-nuclear repulsion energy.
        e_hartree_c: The Hartree energy in the orbital basis (optional). If not provided, it is
            calculated from the density matrix (this is a bit more expensive since a four index
            tensor is involved).
        mol_orbital: The molecule in the orbital basis.
        mol_density: The molecule in the density basis.
        density_fitting: The density fitting function.
        grid: The integration grid for the XC and APBE functionals.
        aos: The atomic orbitals.
        aos_torch: The atomic orbitals as a torch tensor.
        grid_weights: The weights of the grid.
        xc: The exchange-correlation functional.
        use_torch: Whether to use the torch implementation of the XC functional.
        use_blocked: Whether to use the blocked implementation of the XC functional.
        max_memory: The maximum memory for the functionals.
        sanity_check_fit: Whether to perform a sanity check (compute l1 norm of density difference
            of the density fitting).

    Returns:
        A tuple of the KS energies, OF energies, the coefficients in the density basis, and the
        gradients in the density basis.
    """
    ks_energy = {}
    of_energy = {}
    gradient = {}
    if isinstance(gamma, list):
        gamma = np.array(gamma)
    # Calculation of the energies in the orbital basis
    logger.trace("Computing Kohn-Sham XC energy.")
    n_electrons = mol_density.nelectron
    if gamma.ndim != 2:
        raise ValueError("Only two-dimensional density matrices are supported.")
    ks_energy["xc"] = nr_rks(
        dft.numint.NumInt(), mol_orbital, grid, xc, gamma, max_memory=max_memory
    )
    logger.trace("Starting density fitting.")
    coeffs_p = density_fitting(gamma)
    logger.trace("Density fitting done.")

    ks_energy = compute_remaining_ks_energies(
        gamma,
        grad_external_energy_c,
        kinetic_and_external_energy_hamiltonian,
        ks_energy,
        mol_orbital,
        e_nuc_nuc,
        e_hartree_c,
    )

    validate_density_fitting(
        coeffs_p,
        mol_orbital,
        mol_density,
        grid,
        gamma,
        iteration,
        number_of_iteration,
        sanity_check_fit,
        n_electrons,
    )

    logger.trace("Starting evaluation of functionals on the grid.")
    classical_results = eval_density_functionals(
        coeffs_p,
        [xc, "APBE"],
        mol_density,
        grid,
        aos,
        aos_torch,
        grid_weights,
        use_torch,
        use_blocked,
        max_memory,
    )
    logger.trace("Finished evaluation of functionals on the grid.")
    of_energy["kinapbe"], gradient["kinapbe"] = classical_results["APBE"]
    of_energy["xc"], gradient["xc"] = classical_results[xc]
    gradient["hartree"] = coulomb_matrix @ coeffs_p
    of_energy["hartree"] = coeffs_p @ gradient["hartree"] / 2
    gradient["external"] = grad_external_energy_p
    of_energy["external"] = coeffs_p @ grad_external_energy_p
    gradient["effective_potential"] = gradient["hartree"] + gradient["xc"] + gradient["external"]

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
            f"KS and OF kinetic energies differ by {kin_diff * 1e3:.2g} mHa at iteration {iteration} (total number of iteration={number_of_iteration})"
        )

    if hartree_diff > 1e-3:
        logger.warning(
            f"KS and OF hartree energies differ by {hartree_diff * 1e3:.2g} mHa at iteration {iteration} (total number of iteration={number_of_iteration})"
        )

    if external_diff > 1e-3:
        logger.warning(
            f"KS and OF external energies differ by {external_diff * 1e3:.2g} mHa at iteration {iteration} (total number of iteration={number_of_iteration})"
        )

    if electronic_diff > 1e-3:
        logger.warning(
            f"KS and OF electronic energies differ by {electronic_diff * 1e3:.2g} mHa at iteration {iteration} (total number of iteration={number_of_iteration})"
        )

    return ks_energy, of_energy, coeffs_p, gradient


def get_data_dict(
    ks_energies: dict,
    of_energies: dict,
    of_coeffs: np.ndarray,
    of_gradients: dict,
    has_energy_label: np.ndarray,
    e_nuc_nuc: float,
):
    """Create a dict with all data that should be stored.

    Naming:

        * `of_` = orbital free
        * `ks_` = kohn sham
        * `e_` = energy
        * `grad_` = gradient

    Args:
        ks_energies: The energies in the Kohn-Sham basis.
        of_energies: The energies in the density basis.
        of_coeffs: The coefficients in the density basis.
        of_gradients: The gradients in the density basis.
        has_energy_label: Whether the energy label is available for an iteration.
        e_nuc_nuc: The nuclear-nuclear repulsion energy.
    """
    length = len(ks_energies["kinetic"])
    data = {
        # Scalars
        "ks_has_energy_label": has_energy_label,  # could also be in of, doesn't matter
        "ks_e_kin": ks_energies["kinetic"],
        "ks_e_hartree": ks_energies["hartree"],
        "ks_e_xc": ks_energies["xc"],
        "ks_e_ext": ks_energies["external"],
        "ks_e_nuc_nuc": np.repeat(e_nuc_nuc, length),  # could also be in of, doesn't matter
        "ks_e_electron": ks_energies["electronic"],
        "ks_e_tot": ks_energies["total"],
        "of_e_kin": of_energies["kinetic"],
        "of_e_hartree": of_energies["hartree"],
        "of_e_xc": of_energies["xc"],
        "of_e_ext": of_energies["external"],
        "of_e_kin_plus_xc": of_energies["kinetic"] + of_energies["xc"],
        "of_e_electron": of_energies["electronic"],
        "of_e_tot": of_energies["total"],
        # Arrays
        "of_coeffs": of_coeffs,
        "of_grad_hartree": of_gradients["hartree"],
        "of_grad_ext": of_gradients["external"],
        "of_grad_kin": of_gradients["kinetic"],
        "of_grad_xc": of_gradients["xc"],
        "of_grad_kin_plus_xc": of_gradients["kinetic"] + of_gradients["xc"],
        "of_grad_tot": of_gradients["kinetic"]
        + of_gradients["hartree"]
        + of_gradients["xc"]
        + of_gradients["external"],
    }

    data.update(
        {
            "of_e_kinapbe": of_energies["kinapbe"],
            "of_e_kin_minus_apbe": of_energies["kinetic"] - of_energies["kinapbe"],
            "of_grad_kinapbe": of_gradients["kinapbe"],
            "of_grad_kin_minus_apbe": of_gradients["kinetic"] - of_gradients["kinapbe"],
        }
    )

    assert all(len(value) == length for value in data.values()), f"{length=}\n" + "\n".join(
        [f"{key}: {len(value)}" for key, value in data.items()]
    )

    return data


def calculate_labels(
    results: dict,
    initialization: dict,
    data_of_iteration: list[dict],
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    density_fitting_method: str = "hartree+external_mofdft",
) -> dict:
    """Calculate labels from the results of a KSDFT calculation and return them in dict-form.
    (see [M-OFDFT]_ from Section `A.4.1 <https://arxiv.org/html/2309.16578v2#A1.SS4.SSS1>`_
    to `A.4.3 <https://arxiv.org/html/2309.16578v2#A1.SS4.SSS3>`_).

    Args:
        results: The parameters of a ksdft-calculation, provided in dict form.
        initialization: The density matrix and energy of the first iteration of the ksdft-calculation.
        data_of_iteration: List of the data of all ksdft iterations of the molecule.
        mol_orbital_basis: A gto.Mole object of the molecule in the orbital basis.
        mol_density_basis: A gto.Mole object of the molecule in the density basis.
        density_fitting_method: The method used for density fitting.

    Returns:
        A dict of all data relevant for ofdft calculations and comparison.

    Raises:
        KeyError: if one of the required keys is not inside the result dict
        ValueError: if the number of basis functions in data_of_iteration and the molecule differ

    Notes:
        All ks... values are calculated directly from the Kohn-Sham calculation. They are
        calculated at each iteration step of the ksdft-calculation.
        All of... values are calculated from the density "p" that was fitted to the density of the
        orbitals in ksdft.
    """
    _check_data_format(results, data_of_iteration, mol_orbital_basis)
    assert gto.mole.is_same_mol(mol_density_basis, mol_orbital_basis, cmp_basis=False)
    cycles = len(data_of_iteration)

    # set up classical functionals
    xc: str = results["name_xc_functional"].decode()  # strings are stored as bytes in .chk files
    use_torch = xc in str_to_torch_functionals
    grid = grid_setup(mol_density_basis, results["grid_level"], results["prune_method"].decode())
    max_memory = float(os.getenv("PYSCF_MAX_MEMORY", 4000))
    estimated_xc_mem_usage = 8 * mol_density_basis.nao * grid.weights.size * 8 / 1048576  # MB
    use_blocked = estimated_xc_mem_usage > max_memory
    aos = aos_torch = grid_weights = None

    # set up density fitting
    coulomb_matrix = get_coulomb_matrix(mol_density_basis)
    # All variables ending on _c are dependent on the coefficients in the orbital basis
    # While all variables ending on _p are dependent on the fitted density p
    grad_external_energy_c = get_nuclear_attraction_matrix(mol_orbital_basis)
    grad_external_energy_p = get_nuclear_attraction_vector(mol_density_basis)
    # kinetic and nuclei-electron-attraction energy hamiltonian (h1e in pyscf)
    kinetic_and_external_energy_hamiltonian = scf.hf.get_hcore(mol_orbital_basis)

    density_fitting = get_density_fitting_function(
        density_fitting_method,
        mol_orbital_basis,
        mol_density_basis,
        coulomb_matrix,
        grad_external_energy_c,
        grad_external_energy_p,
        max_memory=max_memory,
    )

    if use_torch:
        grid_weights = torch.as_tensor(grid.weights, dtype=torch.float64)
    if not use_blocked:
        aos = dft.numint.eval_ao(mol_density_basis, grid.coords, deriv=1)
        if use_torch:
            aos_torch = torch.as_tensor(aos, dtype=torch.float64)
    n_coeffs_p = mol_density_basis.nao_nr()
    of_coeffs = np.zeros((cycles + 1, n_coeffs_p))
    of_gradients = {name: np.zeros((cycles + 1, n_coeffs_p)) for name in of_gradient_names}
    ks_energies = {name: np.zeros(cycles + 1) for name in ks_energy_names}
    of_energies = {name: np.zeros(cycles + 1) for name in of_energy_names}
    e_nuc_nuc = mol_orbital_basis.energy_nuc()
    has_energy_label = np.array([True] * (cycles + 1))

    gamma_init = initialization["first_density_matrix"]
    logger.trace("Starting density fitting.")
    of_coeffs[0] = density_fitting(gamma_init)
    logger.trace("Density fitting done.")
    grad_hartree_0 = coulomb_matrix @ of_coeffs[0]
    logger.trace("Starting evaluation of functionals on the grid.")
    grad_xc_0 = eval_density_functionals(
        of_coeffs[0],
        [xc],
        mol_density_basis,
        grid,
        aos,
        aos_torch,
        grid_weights,
        use_torch,
        use_blocked,
        max_memory,
    )[xc][1]
    logger.trace("Finished evaluation of functionals on the grid.")
    of_gradients["effective_potential"][0] = grad_hartree_0 + grad_xc_0 + grad_external_energy_p
    has_energy_label[0] = False

    # Calculate Kohn-Sham and orbital-free energies for all iterations
    number_of_iteration = len(data_of_iteration)

    for i, data in enumerate(data_of_iteration, start=1):
        logger.trace(f"Processing SCF iteration {i}/{cycles}.")

        mo_coeff = data["molecular_coeffs_orbitals"]
        mo_occ = data["occupation_numbers_orbitals"]
        gamma = ksdft_density_matrix(mo_coeff, mo_occ)

        diis_coeffs = data.get("diis_coefficients")

        # For perturbed fock matrices (same for both cases)
        if "perturbation_coeffs" in data:
            overlap_matrix = get_overlap_matrix(mol_density_basis)
            gradient_perturbation = overlap_matrix @ data["perturbation_coeffs"].T
        else:
            gradient_perturbation = None

        hartree_energy_c = data["coulomb_energy"]

        ks_energy, of_energy, of_coeff, gradient = get_energies_and_gradients(
            gamma=gamma,
            grad_external_energy_c=grad_external_energy_c,
            grad_external_energy_p=grad_external_energy_p,
            kinetic_and_external_energy_hamiltonian=kinetic_and_external_energy_hamiltonian,
            coulomb_matrix=coulomb_matrix,
            e_nuc_nuc=e_nuc_nuc,
            e_hartree_c=hartree_energy_c,
            mol_orbital=mol_orbital_basis,
            mol_density=mol_density_basis,
            density_fitting=density_fitting,
            grid=grid,
            aos=aos,
            aos_torch=aos_torch,
            grid_weights=grid_weights,
            xc=xc,
            iteration=i,
            number_of_iteration=number_of_iteration,
            use_torch=use_torch,
            use_blocked=use_blocked,
            max_memory=max_memory,
        )

        # The gradient of the kinetic energy is the negative of the diis-summed effective potential
        of_gradients["kinetic"][i] = -1 * diis_weighted_average(
            i - 1, diis_coeffs, of_gradients["effective_potential"]
        )
        if gradient_perturbation is not None:
            of_gradients["kinetic"][i] -= gradient_perturbation
        # Comparison with the reference energies from the .chk file
        total_energy_reference = data["total_energy"]
        exchange_correlation_energy_reference = data["exchange_correlation_energy"]
        if np.abs(total_energy_reference - ks_energy["total"]) > 1e-04:
            logger.warning("Total energy does not match the reference energy from the .chk file")
        if np.abs(exchange_correlation_energy_reference - ks_energy["xc"]) > 1e-04:
            logger.warning("XC energy does not match the reference energy from the .chk file")

        # Initialize of_gradients for this iteration with gradient dict values
        for name in of_gradients:
            if name in gradient:
                of_gradients[name][i] = gradient[name]

        # Saving everything
        of_coeffs[i] = of_coeff
        for name in ks_energy_names:
            ks_energies[name][i] = ks_energy.get(name, ks_energies[name][i])

        for name in list(of_energies.keys()):
            of_energies[name][i] = of_energy.get(name, of_energies[name][i])

    # Compute the difference between final effective potential and final DIIS average. This is
    # also the gradient at the ground state in the density basis as the second summand is our
    # kinetic gradient label, and the first summand is the gradient of the other contributions to
    # the energy.
    ground_state_grad = of_gradients["effective_potential"][-1] + of_gradients["kinetic"][-1]
    ground_state_grad_max_abs = np.max(np.abs(ground_state_grad))
    if ground_state_grad_max_abs > 1e-05:
        logger.warning(
            f"Found non-zero gradient at the ground state: "
            f"max(abs(gradient)) = {ground_state_grad_max_abs*1e3:.2g} mHa"
        )

    return get_data_dict(
        ks_energies,
        of_energies,
        of_coeffs,
        of_gradients,
        has_energy_label,
        e_nuc_nuc,
    )


def active_learning(
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    checkpoint_path: str,
    master_transform: torch.nn.Module,
    basis_info: BasisInfo,
    data_of_iteration: list[dict],
    results: dict,
    iterations_per_mol: int,
    max_scale: float,
    density_fitting_method: str = "hartree+external",
    **_,
) -> dict:
    """Calculate the labels for active learning.

    Args:
        mol_orbital_basis: The molecule in the orbital basis.
        mol_density_basis: The molecule in the density basis.
        checkpoint_path: The path to the checkpoint of the model.
        master_transform: The master transform to be used for the calculation.
        basis_info: The basis information.
        data_of_iteration: List of the data of all ksdft iterations of the molecule. Needed for
            the ground state.
        results: The results dict containing the xc functional.
        iterations_per_mol: The number of sampled coefficients per molecule.
        max_scale: The maximum scale for the Gaussian perturbations.
        density_fitting_method: The density fitting method.

    Returns:
        The data dictionary.
    """
    _check_data_format(results, data_of_iteration, mol_orbital_basis)
    assert gto.mole.is_same_mol(mol_density_basis, mol_orbital_basis, cmp_basis=False)
    xc = results["name_xc_functional"].decode()
    use_torch = xc in str_to_torch_functionals
    grid = grid_setup(mol_density_basis, results["grid_level"], results["prune_method"].decode())
    max_memory = float(os.getenv("PYSCF_MAX_MEMORY", 4000))
    estimated_xc_mem_usage = 8 * mol_density_basis.nao * grid.weights.size * 8 / 1048576  # MB
    use_blocked = estimated_xc_mem_usage > max_memory
    aos = aos_torch = grid_weights = None
    if use_torch and not use_blocked:
        aos = dft.numint.eval_ao(mol_density_basis, grid.coords, deriv=1)
        aos_torch = torch.as_tensor(aos, dtype=torch.float64)
        grid_weights = torch.as_tensor(grid.weights, dtype=torch.float64)
    elif not use_torch:
        aos = dft.numint.eval_ao(mol_density_basis, grid.coords, deriv=1)
    e_nuc_nuc = mol_orbital_basis.energy_nuc()

    sample = OFData.minimal_sample_from_mol(mol_density_basis, basis_info, True)
    sample = master_transform(sample)
    model = instantiate_model(checkpoint_path, "cpu", torch.float64)

    attraction_vector = get_nuclear_attraction_vector(mol_density_basis)
    attraction_matrix = get_nuclear_attraction_matrix(mol_orbital_basis)
    coulomb_matrix = get_coulomb_matrix(mol_density_basis)
    overlap_matrix = torch.as_tensor(get_overlap_matrix(mol_density_basis), dtype=torch.float64)

    density_fitting = get_density_fitting_function(
        density_fitting_method,
        mol_orbital_basis,
        mol_density_basis,
        coulomb_matrix,
        attraction_matrix,
        attraction_vector,
        max_memory=max_memory,
    )

    n_coeffs_p = mol_density_basis.nao_nr()
    of_coeffs = np.zeros((iterations_per_mol + 1, n_coeffs_p))
    of_gradients = {
        name: np.zeros((iterations_per_mol + 1, n_coeffs_p)) for name in of_gradient_names
    }
    ks_energies = {name: np.zeros(iterations_per_mol + 1) for name in ks_energy_names}
    of_energies = {name: np.zeros(iterations_per_mol + 1) for name in of_energy_names}
    has_energy_label = np.array([True] * (iterations_per_mol + 1))

    settings = {
        "grad_external_energy_c": attraction_matrix,
        "grad_external_energy_p": attraction_vector,
        "kinetic_and_external_energy_hamiltonian": scf.hf.get_hcore(mol_orbital_basis),
        "coulomb_matrix": coulomb_matrix,
        "e_nuc_nuc": e_nuc_nuc,
        "e_hartree_c": None,
        "mol_orbital": mol_orbital_basis,
        "mol_density": mol_density_basis,
        "density_fitting": density_fitting,
        "grid": grid,
        "aos": aos,
        "aos_torch": aos_torch,
        "grid_weights": grid_weights,
        "xc": xc,
        "use_torch": use_torch,
        "use_blocked": use_blocked,
        "max_memory": max_memory,
    }

    # ground state labels
    mo_gs = data_of_iteration[-1]["molecular_coeffs_orbitals"]
    occ_gs = data_of_iteration[-1]["occupation_numbers_orbitals"]
    gamma_gs = ksdft_density_matrix(mo_gs, occ_gs)
    ks_energy, of_energy, coeffs_gs, of_gradient = get_energies_and_gradients(gamma_gs, **settings)
    # we could also use the diis averaged effective potential here, but due to self-consistency,
    # this should be the same (see above)
    of_gradient["kinetic"] = -of_gradient["effective_potential"]
    of_coeffs[-1] = coeffs_gs
    for name in ks_energy_names:
        ks_energies[name][-1] = ks_energy[name]
    for name in of_energy_names:
        of_energies[name][-1] = of_energy[name]
    for name in of_gradient_names:
        of_gradients[name][-1] = of_gradient[name]
    coeffs_gs = torch.as_tensor(coeffs_gs, dtype=torch.float64)
    coeffs_gs_transformed = transform_tensor_with_sample(sample, coeffs_gs, Rep.VECTOR)

    # np.random.seed(42)
    # direction = np.random.randn(mol_density_basis.nao)
    scales = torch.linspace(max_scale, 0, iterations_per_mol)
    for i, scale in enumerate(scales):
        scale_n = scale / np.sqrt(n_coeffs_p)  # scale by 1/sqrt(nao) to have constant std
        noise = torch.randn(mol_density_basis.nao, dtype=torch.float64)
        coeffs_transformed = coeffs_gs_transformed + scale_n * noise  # sample in transformed basis
        # coeffs_transformed = coeffs_gs_transformed + scale * direction

        sample.coeffs = coeffs_transformed.clone()
        gradient_transformed = model(sample)[1]
        # project the gradient to reduce density fitting error
        gradient_transformed = project_gradient(gradient_transformed, sample)

        gradient = transform_tensor_with_sample(
            sample, gradient_transformed, Rep.GRADIENT, invert=True
        )
        coeffs = transform_tensor_with_sample(sample, coeffs_transformed, Rep.VECTOR, invert=True)
        v_eff = get_v_eff_from_gradient(
            gradient, model.target_key, coeffs, aos_torch, grid_weights
        )

        # here, even a different Kohn-Sham basis could be used
        gamma = get_gamma_from_v_eff(v_eff, mol_density_basis, mol_orbital_basis)
        gamma = gamma.numpy(force=True)

        ks_energy, of_energy, of_coeff, of_gradient = get_energies_and_gradients(gamma, **settings)
        of_gradient["kinetic"] = -v_eff

        delta_coeff = torch.as_tensor(of_coeff, dtype=torch.float64) - coeffs_gs
        norm = torch.sqrt(delta_coeff @ overlap_matrix @ delta_coeff)
        print(f"scale: {scale:.5f}, L2: {norm:.5f}")

        of_coeffs[i] = of_coeff
        for name in ks_energy_names:
            ks_energies[name][i] = ks_energy[name]
        for name in of_energy_names:
            of_energies[name][i] = of_energy[name]
        for name in of_gradient_names:
            of_gradients[name][i] = of_gradient[name]

    return get_data_dict(
        ks_energies,
        of_energies,
        of_coeffs,
        of_gradients,
        has_energy_label,
        e_nuc_nuc=e_nuc_nuc,
    )


str_to_calculation_fct: dict[str, Callable[..., dict]] = {
    "calculate_labels": calculate_labels,
    "active_learning": active_learning,
}
