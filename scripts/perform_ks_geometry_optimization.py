import argparse
import time
from pathlib import Path

from geometric.nifty import logger as geom_logger
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf.lib import chkfile
from tqdm.auto import tqdm

from mldft.utils.pyscf_calculations import build_ks_calculation


def perform_ks_geometry_optimization(
    xyz_path: Path, out_path: Path, ks_cfg: DictConfig, geometric_cfg: DictConfig
) -> None:
    """Perform KS geometry optimization on a set of XYZ files.

    Args:
        xyz_path: Path to the directory containing XYZ files.
        out_path: Path to the output directory for checkpoint files.
        ks_cfg: Configuration for the Kohn-Sham calculations.
        geometric_cfg: Configuration for the geometric optimization.
    """
    basis = ks_cfg.get("basis")
    files = sorted(xyz_path.glob("*.xyz"))
    optimization_kwargs_raw = OmegaConf.to_container(geometric_cfg, resolve=True)
    if not isinstance(optimization_kwargs_raw, dict):
        raise TypeError("Expected geometric configuration to resolve to a dictionary.")
    optimization_kwargs_template: dict[str, object] = optimization_kwargs_raw
    n_fails = 0
    for file in tqdm(files):
        outfile = out_path / f"{file.stem}.chk"
        if outfile.exists():
            logger.info(f"{outfile} already exists, skipping.")
            continue

        logger.info(f"Processing {file}...")
        # This will currently only work for neutral, singulet molecules.
        mol = gto.M(atom=str(file), basis=basis)
        ks = build_ks_calculation(mol, ks_cfg)
        initial_energy: float | None = None

        optimization_kwargs = dict(optimization_kwargs_template)
        user_callback = optimization_kwargs.pop("callback", None)

        def optimization_callback(local_vars: dict) -> None:
            nonlocal initial_energy
            if initial_energy is None:
                initial_energy = float(local_vars["energy"])
            if callable(user_callback):
                user_callback(local_vars)

        start_time = time.time()
        try:
            mol_eq = optimize(ks, callback=optimization_callback, **optimization_kwargs)
        except RuntimeError:
            logger.warning(f"{file} geometry optimization failed!")
            n_fails += 1
            continue

        end_time = time.time()
        opt_time = end_time - start_time

        start_time = time.time()
        ks_eq = build_ks_calculation(mol_eq, ks_cfg)
        ks_eq.chkfile = str(outfile)
        ks_eq.run()
        end_time = time.time()
        scf_time = end_time - start_time

        if initial_energy is None:
            raise RuntimeError(
                f"Failed to capture initial energy from geometry optimization for {file}."
            )

        chkfile.dump_chkfile_key(outfile, "initial_energy", initial_energy)
        chkfile.dump_chkfile_key(outfile, "opt_time", opt_time)
        chkfile.dump_chkfile_key(outfile, "scf_time", scf_time)
        logger.info(f"KS geometry optimization completed for {file}.")

    if n_fails > 0:
        raise RuntimeError(f"KS geometry optimization failed for {n_fails} files.")


if __name__ == "__main__":
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|" "<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Name of the directory that contains the xyz directory.",
    )
    parser.add_argument(
        "-k",
        "--ks_cfg",
        type=str,
        help="Name of the ks config file",
        default="default.yaml",
    )
    parser.add_argument(
        "-g",
        "--geometric_cfg",
        type=str,
        help="Name of the geometric config file.",
        default="loose.yaml",
    )
    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../configs/datagen/kohn_sham"):
        ks_cfg = compose(config_name=args.ks_cfg, overrides=["perturbation_cfg.of_basis_set=none"])
    with initialize(version_base="1.3", config_path="../configs/ofdft/geometric_config"):
        geometric_cfg = compose(config_name=args.geometric_cfg)

    xyz = Path(args.directory) / "xyz"
    assert xyz.exists(), f"Directory {xyz} does not exist."
    out = xyz.parent / "ks-opt"
    if not out.exists():
        out.mkdir()
    config_dir = xyz.parent / "config"
    if not config_dir.exists():
        config_dir.mkdir()
    OmegaConf.save(ks_cfg, config_dir / "kohn_sham.yaml")
    OmegaConf.save(ks_cfg, config_dir / "kohn_sham_resolved.yaml", resolve=True)
    OmegaConf.save(geometric_cfg, config_dir / "geometric.yaml")
    OmegaConf.save(geometric_cfg, config_dir / "geometric_resolved.yaml", resolve=True)

    geom_logger.setLevel(40)  # suppress geometric logger
    perform_ks_geometry_optimization(xyz, out, ks_cfg, geometric_cfg)
