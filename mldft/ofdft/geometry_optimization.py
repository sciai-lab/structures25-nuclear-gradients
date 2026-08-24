"""Module containing the geometry optimization engine for OFDFT employing the geomeTRIC package."""

import os
import tempfile
import time
import uuid
from collections.abc import Callable

import geometric
import geometric.molecule
import numpy as np
import torch
from loguru import logger
from pyscf import gto
from tqdm import tqdm

from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.density_optimization import density_optimization
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.initial_guess import get_initial_guess_estimation
from mldft.ofdft.nuclear_gradients import (
    DEFAULT_GRADIENTS,
    get_gradient_method,
)
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.utils.molecules import geometry_to_string
from mldft.utils.sample_generator import SampleGenerator

MAX_DENOP_FAILURES = 3


class GeometryOptimizationNotConvergedError(Exception):
    pass


class GeomOpt(geometric.engine.Engine):
    """Geometry optimization for OFDFT using geometric.

    Attributes:
        mol: The molecule to optimize.
        cycle: The current cycle of the optimization.
        e_last: The energy of the last optimization step.
        callback: A callback function to call after each optimization step.
        maxsteps: The maximum number of geometry optimization steps.
        energy_and_gradient_calculator: The function to calculate the energy and gradient.
    """

    def __init__(
        self,
        sample: OFData,
        func_factory: FunctionalFactory,
        sample_generator: SampleGenerator,
        optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
        gradients: str | Callable = DEFAULT_GRADIENTS,
        maxsteps: int = 100,
        geometry_opt_callback: None | Callable = None,
        initial_guess_from_last: str | None = None,
        ofdft_kwargs: dict | None = None,
        nuclear_gradient_device: torch.device | str = "auto",
    ) -> None:
        """Initialize the geometry optimization engine.

        Args:
            sample: The OFData sample to optimize.
            func_factory: The functional factory to use.
            sample_generator: The model configuration to use.
            optimizer: The optimizer to use for the OF-DFT single point calculations.
            gradients: The gradients to use (Numerical or Analytical). Can be a string or a callable.
            maxsteps: The maximum number of geometry optimization steps.
            geometry_opt_callback: A callback function to call after each optimization step.
            initial_guess_from_last: If set, the initial guess for the density is taken from the last
                geometry optimization step. Can be "simple" or "safe".
            ofdft_kwargs: Additional keyword arguments for the density optimization.
        """
        if ofdft_kwargs is None:
            ofdft_kwargs = {"callback": None}

        # if initial guess from last is being used the ofdft kwargs will be modified, it is
        # thus saver to deep copy them here so that the user does not need to wory about it
        ofdft_kwargs_local = ofdft_kwargs.copy()

        gradient_method = get_gradient_method(
            gradients, sample_generator, func_factory, device=nuclear_gradient_device
        )

        initial_guess_estimation = get_initial_guess_estimation(initial_guess_from_last)

        mol = sample.mol
        self.mol = mol
        self.sample = sample

        if mol.natm == 1:
            raise ValueError("Nothing to optimize in a single atom")

        molecule = geometric.molecule.Molecule()
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        # geometric molecule expects the coordinates in Angstrom
        molecule.xyzs = np.asarray([[mol.atom_coord(i, unit="Angstrom") for i in range(mol.natm)]])
        super().__init__(molecule)

        self.cycle = 0
        self.e_last = 0
        self.last_coeffs = None
        self.callback = geometry_opt_callback
        self.maxsteps = maxsteps
        self.consecutive_denop_failures = 0
        self.calc_new_return_time = None
        self._last_denop_time = 0.0
        self._last_gradient_time = 0.0
        self.pbar = tqdm(
            range(self.maxsteps),
            leave=False,
            dynamic_ncols=True,
            position=int(os.getenv("GEOMOPT_PID", 0)),
        )
        self.iter_pbar = iter(self.pbar)

        def energy_and_gradient_calculator(
            mol: gto.Mole, last_coeffs: None | torch.Tensor = None
        ) -> tuple[float, np.ndarray, torch.Tensor, bool]:
            """Calculate the energy and gradient for the new geometry.

            Args:
                mol: The molecule with the current geometry.

            Returns:
                A tuple containing the energy, gradient (shape 3 * natms), coeffs, and
                convergence status.
            """
            sample = sample_generator.get_sample_from_mol(mol)
            if initial_guess_estimation is not None and last_coeffs is not None:
                ofdft_kwargs_local["initialization"] = initial_guess_estimation(
                    sample, last_coeffs
                )

            denop_start = time.perf_counter()
            energy, coeffs, converged, _ = density_optimization(
                sample,
                sample.mol,
                optimizer,
                func_factory,
                disable_printing=True,
                **ofdft_kwargs_local,
            )
            denop_time = time.perf_counter() - denop_start

            if not converged:
                logger.warning("Single point calculation did not converge")

            self.sample = sample
            gradient_start = time.perf_counter()
            gradient = gradient_method(sample, coeffs)
            gradient_time = time.perf_counter() - gradient_start
            self._last_denop_time = denop_time
            self._last_gradient_time = gradient_time
            return (
                energy.total_energy,
                gradient.cpu().detach().numpy(),
                coeffs.clone().detach(),
                converged,
            )

        self.energy_and_gradient_calculator = energy_and_gradient_calculator

    def calc_new(self, coords: np.ndarray, dirname: str) -> dict:
        """Calculate the energy and gradient for the new geometry.

        Required by geometric to run the optimization.

        Args:
            coords: The new coordinates in Bohr shape (3 * natoms,).
            dirname: required by geometric, but not actively used by us.
        """
        if self.cycle >= self.maxsteps:
            raise GeometryOptimizationNotConvergedError(
                f"Geometry optimization did not converge in {self.maxsteps} steps"
            )
        mol = self.mol

        # Calculate time geometric spent processing previous step
        calc_new_call_time = time.perf_counter()
        if self.calc_new_return_time is not None:
            geometric_time = calc_new_call_time - self.calc_new_return_time
        else:
            geometric_time = 0.0  # First cycle has no geometric time yet

        self.cycle += 1
        logger.trace(f"Geometry optimization cycle: {self.cycle}")

        coords = coords.reshape(-1, 3)
        # But geometric returns the new coordinates in Bohr
        self.mol.set_geom_(coords, unit="Bohr")

        logger.trace("Current Geometry:\n" + geometry_to_string(mol))

        energy, gradient, last_coeffs, converged = self.energy_and_gradient_calculator(
            mol, last_coeffs=self.last_coeffs
        )
        denop_time = self._last_denop_time
        gradient_time = self._last_gradient_time

        nuclear_gradient_norm = np.linalg.norm(gradient)
        max_nuclear_gradient = np.abs(gradient).max()
        rms_nuclear_gradient = np.sqrt(np.power(gradient, 2).sum() / gradient.size)
        logger.debug(
            f"  Cycle: {self.cycle:>2}, energy step: {(energy - self.e_last) * 1e3:>11.3f} mHa, "
            f"nuclear gradient norm: {nuclear_gradient_norm*1e3:>9.3f} mHa/Bohr"
            f" max: {max_nuclear_gradient*1e3:>9.3f} mHa/Bohr"
            f" rms: {rms_nuclear_gradient*1e3:>9.3f}"
        )
        self.pbar.set_description(
            f"ΔE={(energy - self.e_last) * 1e3:>11.3f} mHa, nuc_grad_norm={nuclear_gradient_norm*1e3:>9.3f} mHa/Bohr max={max_nuclear_gradient*1e3:>5.3f} rms={rms_nuclear_gradient*1e3:>5.3f}"
        )
        next(self.iter_pbar)
        self.last_coeffs = last_coeffs
        self.e_last = energy

        if callable(self.callback):
            self.callback(locals())

        if not converged:
            self.consecutive_denop_failures += 1
            if self.consecutive_denop_failures >= MAX_DENOP_FAILURES:
                raise GeometryOptimizationNotConvergedError(
                    f"Density optimization did not converge {MAX_DENOP_FAILURES} times in a row at cycle {self.cycle}"
                )
        else:
            self.consecutive_denop_failures = 0

        self.calc_new_return_time = time.perf_counter()
        return {"energy": energy, "gradient": gradient}

    def run(self, **kwargs) -> tuple[gto.Mole, OFData]:
        """Run the geometry optimization.

        Returns:
            The optimized molecule.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # this should be made a pathlib object
            tmpf = os.path.join(tmpdir, str(uuid.uuid4()))
            geometric.optimize.run_optimizer(
                customengine=self, constraints=None, input=tmpf, **kwargs
            )
            self.pbar.close()

        return self.mol, self.sample
