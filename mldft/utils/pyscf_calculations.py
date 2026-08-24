"""Shared PySCF calculation and checkpoint helpers."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from pyscf import dft, gto, scf
from pyscf.lib import chkfile

from mldft.utils.grids import STRING_TO_PRUNE


def build_ks_calculation(mol: gto.Mole, ks_cfg: DictConfig) -> dft.rks.RKS:
    """Build an RKS calculator configured from a Hydra config.

    Args:
        mol: Molecule for which to build the calculator.
        ks_cfg: Configuration for the Kohn-Sham calculation.

    Returns:
        Configured RKS object.
    """
    ks = dft.RKS(mol, xc=ks_cfg.get("xc"))
    ks.init_guess = ks_cfg.get("initialization")
    ks.conv_tol = ks_cfg.get("convergence_tolerance")
    ks.grids.level = ks_cfg.get("grid_level")
    ks.grids.prune = STRING_TO_PRUNE[ks_cfg.get("prune_method")]
    ks.chkfile = None

    density_fit_threshold = ks_cfg.get("density_fit_threshold")
    if mol.natm >= density_fit_threshold:
        ks.density_fit(auxbasis=ks_cfg.get("density_fit_basis"))

    ks.verbose = 0
    return ks


def load_scf_with_optional_data(ks_file: Path, optional_keys: tuple[str, ...] = ()) -> Any:
    """Load PySCF SCF data and optional top-level checkpoint keys.

    Args:
        ks_file: Path to the KS checkpoint file.
        optional_keys: Extra chkfile keys to load if present.

    Returns:
        Tuple of molecule and SCF data, with optional keys appended to the data dictionary.
    """
    ks_mol, ks_data = scf.chkfile.load_scf(ks_file)
    if not isinstance(ks_data, dict):
        return ks_mol, ks_data

    for key in optional_keys:
        try:
            ks_data[key] = chkfile.load(ks_file, key)
        except Exception:
            # PySCF chkfiles do not expose a stable missing-key exception type.
            continue

    return ks_mol, ks_data
