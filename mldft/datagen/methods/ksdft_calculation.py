r"""Wraps the KSDFT calculation and saves results to a .chk file. The theory of KSDFT is described in [M-OFDFT]_ in
Section `A.1 <https://arxiv.org/html/2309.16578v2#A1.SS1>`_
to Section `A.3 <https://arxiv.org/html/2309.16578v2#A1.SS3>`_
and [STRUCTURES25]_, Section `S.1.2 <https://arxiv.org/html/2503.00443v2#Ax1.SS1.SSS2>`_.

This module wraps the KSDFT-calculation as it is implemented in the pyscf library.
A Kohn Sham class is built with the molecule and calculation parameters.
To obtain the intermediate results after each iteration of the self-consistent-field
method, which are needed further down the line as additional datapoints for training the model, a callback function was
implemented. For the DIIS-coefficients it is further necessary to patch the extrapolate function of the CDIIS - class
in pyscf. All values are stored at runtime in a .chk file.

 The file format is as follows:

In the "Results" folder the initialization parameters of the calculation are stored:

 - **"converged"** : bool , if the scf-method converged on an energy.
 - **"total_energy"** : float , the total energy of the molecule after the last iteration.
 - **"occupation_numbers_orbitals"** : np.ndarray , the occupation numbers of each orbital after the last iteration.
 - **"molecular_coeffs_orbitals"** : np.ndarray , the coefficients of the orbitals after the last iteration.
 - **"max_cycle"** : int , the maximal number of iterations the calculation was initialized with.
 - **"name_xc_functional"** : string , the specified exchange correlation functional.
 - **"init_guess"** : string , the specified initial guess method.
 - **"convergence_tolerance"** : float , the specified convergence tolerance.
 - **"diis_start_cycle"** : int , the first cycle when diis is being used.
 - **"diis_space"** : int , the maximal number of Fock matrices to average in the diis scheme.
 - **"diis_method"** : string , the specified diis method.
 - **"grid_level"** : int , A parameter specifying the density of the grid used in the xc functional.
 - **"prune_method"** : string , A parameter specifying the pruning scheme of the grid used in the xc functional.
   Additionally, in the "Initialization" folder the initial density matrix and the total energy before the first
   iteration are stored, which are needed for the kinetic energy gradient.
 - **"first_density_matrix"** : np.ndarray , the density matrix before the first iteration.
 - **"first_total_energy"** : float , the total energy before the first iteration.
   for each cycle the intermediate results are saved into a "KS-iteration/{cycle}" folder:
 - **"diis_coefficients"** : np.ndarray , the diis-coefficients used to construct the current Fock matrix.
 - **"occupation_numbers_orbitals"** : np.ndarray , the occupations number of the orbitals.
 - **"molecular_coeffs_orbitals"** : np.ndarray , the coefficients of the orbitals in the specified basis.
 - **"total_energy"** : float , the total energy of the molecule in this iteration.
 - **"coulomb_energy"** : float , the total coulomb energy after this iteration (equal to the hartree energy).
 - **"exchange_correlation_energy"** : float , the energy calculated from the exchange correlation functional.
"""

import os
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import scipy
from omegaconf import DictConfig
from pyscf import dft, gto, scf
from pyscf.lib import logger, misc
from pyscf.lib.diis import BLOCK_SIZE

from mldft.ofdft.basis_integrals import get_overlap_tensor
from mldft.utils.grids import PRUNE_TO_STRING, STRING_TO_PRUNE
from mldft.utils.molecules import construct_aux_mol


def perturb_fock_matrix(
    fock_matrix: np.ndarray,
    mf: dft.rks,
    cycle: int,
    start_std: float,
    end_std: float,
    start_cycle: int,
    end_cycle: int,
    of_basis_set: str,
    std_sampling: str = "linear",
    reduction_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the Fock matrix by adding a random perturbation which is sampled in the density
    basis.

    Args:
        fock_matrix: The Fock matrix to perturb.
        mf: The pyscf mf object.
        cycle: The current cycle of the SCF iteration.
        start_std: The standard deviation of the perturbation at the first cycle.
        end_std: The standard deviation of the perturbation at the last cycle.
        start_cycle: The cycle at which the perturbation starts.
        end_cycle: The cycle at which the perturbation ends.
        of_basis_set: String of the density basis set.
        std_sampling: Sampling schedule for perturbation standard deviation.
            Supported values are "linear" and "exponential".
        reduction_factor: Multiplicative decay factor used when
            std_sampling == "exponential".
    """
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    mol_orbital = mf.mol
    mol_density = construct_aux_mol(mol_orbital, aux_basis_name=of_basis_set)

    if std_sampling == "linear":
        if end_cycle == start_cycle:
            pert_std = end_std
        else:
            pert_std = start_std * (end_cycle - cycle) / (end_cycle - start_cycle) + end_std
    elif std_sampling == "exponential":
        if reduction_factor is None:
            raise ValueError("Exponential perturbation std sampling requires reduction_factor.")
        pert_std = start_std * reduction_factor ** (cycle - start_cycle)
    else:
        raise ValueError(
            f"Unknown std_sampling '{std_sampling}'. Supported values are "
            "'linear' and 'exponential'."
        )

    overlap_tensor = get_overlap_tensor(mol_density, mol_orbital)

    n_ao = mol_orbital.nao
    if fock_matrix.shape[0] != n_ao**2:
        raise ValueError(f"Fock matrix has unexpected shape {fock_matrix.shape}.")
    perturbation_coeffs = np.random.normal(0, pert_std, overlap_tensor.shape[0])
    perturbation = np.einsum("ijk,i->jk", overlap_tensor, perturbation_coeffs, optimize=True)
    perturbed_fock_matrix = fock_matrix + perturbation.ravel()

    return perturbed_fock_matrix, perturbation_coeffs


def patched_extrapolate(
    self,
    n_d: int | None = None,
    use_perturbation: bool = False,
    pertubation_function: Callable | None = None,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> np.ndarray:
    """Monkeypatch for pyscf.lib.diis.extrapolate to extract the DIIS coefficients. Extrapolate the
    next Fock matrix based on the previous Fock matrices and errors.

    Args:
        n_d: Number of vectors to be used in the extrapolation. Default is
            all vectors.

    Returns:
        np.ndarray: A new Fock matrix.
    """
    if n_d is None:
        n_d = self.get_num_vec()
    if n_d == 0:
        raise RuntimeError("No vector found in DIIS object.")

    h = self._H[: n_d + 1, : n_d + 1]
    g = np.zeros(n_d + 1, h.dtype)
    g[0] = 1

    w, v = scipy.linalg.eigh(h)
    if np.any(abs(w) < 1e-14):
        logger.debug(self, "Linear dependence found in DIIS error vectors.")
        idx = abs(w) > 1e-14
        c = np.dot(v[:, idx] * (1.0 / w[idx]), np.dot(v[:, idx].T.conj(), g))
    else:
        try:
            c = np.linalg.solve(h, g)
        except np.linalg.linalg.LinAlgError as e:
            logger.warn(self, " diis singular, eigh(h) %s", w)
            raise e
    logger.debug1(self, "diis-c %s", c)

    xnew = None
    for i, ci in enumerate(c[1:]):
        xi = self.get_vec(i)
        if xnew is None:
            xnew = np.zeros(xi.size, c.dtype)
        for p0, p1 in misc.prange(0, xi.size, BLOCK_SIZE):
            xnew[p0:p1] += xi[p0:p1] * ci

    # -- modified from pyscf --
    # discard the first coefficient, corresponding to the lagrange multiplier for the coefficient sum
    diis_coeffs = c[1:]
    # reorder the coefficients to match the order of the SCF steps:
    # the last coefficient corresponds to the previous step, the first to the oldest step taken into account by DIIS.
    diis_coeffs = np.concatenate([diis_coeffs[self._head :], diis_coeffs[: self._head]])
    # save the coefficients in the DIIS object
    self.diis_coeffs = diis_coeffs

    if use_perturbation:
        # perturb the Fock matrix
        if hasattr(self, "cycle") and (self.cycle <= end_cycle) and (self.cycle >= start_cycle):
            xnew, self.perturbation_coeffs = pertubation_function(xnew, self.mf_aux, self.cycle)
        else:
            # reset the perturbation coefficients
            self.perturbation_coeffs = None
    # -- end of modification --
    return xnew


def compute_lumo_idx(molecular_occupation: np.ndarray) -> int:
    """Compute the index of the lowest unoccupied orbital. (If all orbitals are occupied, the
    highest occupied orbital is returned instead.)

    This assumes that all orbitals up to the last occupied one are occupied, which is handled by
    pyscf.
    """
    if np.array(molecular_occupation).ndim != 1:
        raise ValueError("Occupation numbers must be one-dimensional.")
    nonzero = np.flatnonzero(molecular_occupation)
    assert len(nonzero) > 0, "No occupied orbitals found."
    lumo_idx = min(nonzero[-1] + 1, len(molecular_occupation) - 1)
    assert np.all(molecular_occupation[:lumo_idx] != 0), "Occupied orbitals are not all occupied."
    return lumo_idx


def _save_scf_iteration_callback(envs: dict) -> None:
    """Save the data of each iteration in the chkfile.

    Args:
        envs: Dictionary with the local environment variables of the iteration.
    Returns:
        None
    """
    cycle = envs["cycle"]
    # for perturbation of the DIIS coefficients
    mf = envs["mf"]
    envs["mf_diis"].cycle = cycle
    envs["mf_diis"].mf_aux = mf
    # These are the initial values from before the first iteration. They are needed for the kinetic energy gradient
    if cycle == 0:
        infos_initialization = {
            "first_density_matrix": envs["dm_last"],
            "first_total_energy": envs["last_hf_e"],
        }
        scf.chkfile.save(mf.chkfile, "Initialization", infos_initialization)
    diis_coeffs = envs["mf_diis"].diis_coeffs  # only works if DIIS is enabled from the start
    lumo_idx = compute_lumo_idx(envs["mo_occ"])

    info = {
        "diis_coefficients": diis_coeffs,  # DIIS coefficients
        "occupation_numbers_orbitals": envs["mo_occ"][..., : lumo_idx + 1],  # Occupied orbitals
        "molecular_coeffs_orbitals": envs["mo_coeff"][..., : lumo_idx + 1],
        "total_energy": envs["e_tot"],
        "coulomb_energy": envs["vhf"].ecoul,
        "exchange_correlation_energy": envs["vhf"].exc,
    }
    if (
        hasattr(envs["mf_diis"], "perturbation_coeffs")
        and envs["mf_diis"].perturbation_coeffs is not None
    ):
        info["perturbation_coeffs"] = envs["mf_diis"].perturbation_coeffs
    scf.chkfile.save(mf.chkfile, f"KS-iteration/{cycle:d}", info)


class ConvergenceError(Exception):
    """Raised when the calculation did not converge."""


def ksdft(
    mol: gto.Mole,
    savefile: Path,
    xc_functional: str = r"PBE",
    init_guess: str = "minao",
    max_cycle: int = 60,
    diis_space: int = 8,
    diis_method: str = "CDIIS",
    grid_level: int = 3,
    prune_method: str | Callable | None = dft.nwchem_prune,
    density_fit_basis: str = "def2-universal-jfit",
    density_fit_threshold: int = 30,
    convergence_tolerance: float = 1e-9,
    extra_callback: Callable = None,
    use_perturbation: bool = False,
    perturbation_cfg: DictConfig | None = None,
) -> None:
    """Calculates the non-relativistic spin calculation. The theory of KSDFT is described in [M-OFDFT]_ in
    Section `A.1 <https://arxiv.org/html/2309.16578v2#A1.SS1>`_
    to Section `A.3 <https://arxiv.org/html/2309.16578v2#A1.SS3>`_
    and [STRUCTURES25]_, Section `S.1.2 <https://arxiv.org/html/2503.00443v2#Ax1.SS1.SSS2>`_. XC functional should be PBE.
    Basis set should be 6-31G(2df,p). DIIS is enabled by default. MINAO initialization is active by
    default.

    Args:
        mol: Molecule object
        savefile: Path to the savefile
        xc_functional: The xc functional. Default is PBE.
        init_guess: the initial guess method (default is MINAO)
        max_cycle: the maximal number of iterations
        diis_space: the maximal number of Fock matrices used in the DIIS method
        diis_method: Either CDIIS(default), EDIIS or ADIIS
        grid_level: The grid level to use for the integration grid used in the xc-functional.
        prune_method: The method to prune the integration grid. If None, no pruning is performed.
        density_fit_basis: The basis set to use for the density fitting.
        density_fit_threshold: The threshold for the number of atoms in the molecule to use density fitting.
        convergence_tolerance: The convergence tolerance after which the SCF iteration stops. An alternative value can
            be 1meV 0.0000367493, see Appendix C.2 in [M-OFDFT]_.
        extra_callback: Additional callback function to be called after the original callback is called each iteration.
        use_perturbation: If True, the Fock matrix is perturbed each iteration.
        perturbation_cfg: Configuration for the perturbation of the Fock matrix.
    Returns:
        None
    Raises:
        ConvergenceError: If the calculation did not converge.
    """
    start_time = time.time()

    if mol.spin != 0:
        raise ValueError("Only closed-shell molecules are supported.")

    # our diis averaging only works if DIIS is enabled from the start (which is the pyscf default)
    diis_start_cycle = 0

    mf = dft.RKS(mol, xc=xc_functional)

    if len(mol.atom_charges()) >= density_fit_threshold:
        mf.density_fit(density_fit_basis)
    mf.chkfile = savefile.as_posix()

    if isinstance(prune_method, str):
        if prune_method not in STRING_TO_PRUNE:
            raise NotImplementedError("This pruning method is currently not implemented")
        prune_method = STRING_TO_PRUNE[prune_method]
    if prune_method not in PRUNE_TO_STRING:
        prune_string = "unknown_function"
    else:
        prune_string = PRUNE_TO_STRING[prune_method]

    if diis_method == "CDIIS":
        if use_perturbation:

            def perturbation_function(fock_matrix, mf, cycle):
                return perturb_fock_matrix(
                    fock_matrix,
                    mf,
                    cycle,
                    perturbation_cfg.start_std,
                    perturbation_cfg.end_std,
                    perturbation_cfg.start_cycle,
                    perturbation_cfg.end_cycle,
                    perturbation_cfg.of_basis_set,
                    perturbation_cfg.std_sampling,
                    perturbation_cfg.reduction_factor,
                )

            mf.DIIS.extrapolate = lambda self, n_d=None: patched_extrapolate(
                self,
                n_d,
                use_perturbation,
                perturbation_function,
                perturbation_cfg.start_cycle,
                perturbation_cfg.end_cycle,
            )
        else:
            mf.DIIS.extrapolate = lambda self, n_d=None: patched_extrapolate(self, n_d)

    else:
        raise NotImplementedError(f"DIIS method {diis_method} not supported.")

    mf.grids.level = grid_level
    mf.grids.prune = prune_method

    if extra_callback is None:
        callback = _save_scf_iteration_callback
    else:

        def callback(*args, **kwargs):
            _save_scf_iteration_callback(*args, **kwargs)
            extra_callback(*args, **kwargs)

    mf.run(
        callback=callback,
        init_guess=init_guess,
        max_cycle=max_cycle,
        conv_tol=convergence_tolerance,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
    )

    assert mf.damp == 0
    assert mf.level_shift == 0

    if not mf.converged:
        raise ConvergenceError("The calculation did not converge.")

    lumo_idx = compute_lumo_idx(mf.mo_occ)

    time_end = time.time()

    # Keep orbitals through the LUMO.
    mo_coeff_sliced = mf.mo_coeff[..., : lumo_idx + 1]
    mo_occ_sliced = mf.mo_occ[..., : lumo_idx + 1]

    res = {
        "converged": mf.converged,
        "total_energy": mf.e_tot,
        "occupation_numbers_orbitals": mo_occ_sliced,
        "molecular_coeffs_orbitals": mo_coeff_sliced,
        "max_cycle": mf.max_cycle,
        "name_xc_functional": xc_functional,
        "init_guess": init_guess,
        "convergence_tolerance": convergence_tolerance,
        "diis_start_cycle": diis_start_cycle,
        "diis_space": diis_space,
        "diis_method": diis_method,
        "grid_level": mf.grids.level,
        "prune_method": prune_string,
        "calculation_time": time_end - start_time,
        "expected_S_squared": mf.spin_square()[0],
        "multiplicity": mol.spin + 1,
        "charge": mol.charge,
    }
    scf.chkfile.save(mf.chkfile, "Results", res)
    with h5py.File(mf.chkfile, "r+") as fh5:
        if "scf" in fh5:
            del fh5["scf"]
