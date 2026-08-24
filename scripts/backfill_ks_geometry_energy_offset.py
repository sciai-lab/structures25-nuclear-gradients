import argparse
import time
from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mldft.ofdft.geometry_energy_offset import (
    KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE,
    set_ks_geometry_energy_offset_result,
)
from mldft.utils.conversions import as_scalar_bool, as_scalar_float
from mldft.utils.pyscf_calculations import (
    build_ks_calculation,
    load_scf_with_optional_data,
)


def _resolve_ks_opt_dir(ks_path: Path) -> Path:
    """Resolve a dataset root or ks-opt path to the KS checkpoint directory."""
    if ks_path.name == "ks-opt":
        return ks_path
    return ks_path / "ks-opt"


def _load_ks_cfg(ks_cfg_path: Path | None) -> DictConfig:
    """Load the KS level-of-theory config."""
    if ks_cfg_path is None:
        ks_cfg_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "datagen"
            / "kohn_sham"
            / "default.yaml"
        )
    return OmegaConf.load(ks_cfg_path)


def _build_ml_geometry_mol(ks_mol, ml_mol):
    """Copy the KS molecule and replace only its coordinates with the ML geometry."""
    if ks_mol.natm != ml_mol.natm:
        raise ValueError(
            f"Molecule atom count mismatch: KS has {ks_mol.natm}, ML has {ml_mol.natm}."
        )

    ks_symbols = [ks_mol.atom_symbol(index) for index in range(ks_mol.natm)]
    ml_symbols = [ml_mol.atom_symbol(index) for index in range(ml_mol.natm)]
    if ks_symbols != ml_symbols:
        raise ValueError(
            f"Molecule atom symbols mismatch: KS has {ks_symbols}, ML has {ml_symbols}."
        )

    return ks_mol.set_geom_(
        ml_mol.atom_coords(unit="Bohr"),
        unit="Bohr",
        inplace=False,
    )


def _compute_ks_single_point_energy(mol, ks_cfg: DictConfig) -> tuple[float, bool, float]:
    """Run a configured KS single-point calculation."""
    ks = build_ks_calculation(mol, ks_cfg)
    start_time = time.perf_counter()
    energy = float(ks.kernel())
    runtime_s = time.perf_counter() - start_time
    return energy, bool(ks.converged), runtime_s


def backfill_ks_geometry_energy_offsets(
    output_dir: Path,
    ks_opt_dir: Path,
    ks_cfg: DictConfig,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Backfill KS geometry energy offsets for saved geometry optimization samples."""
    sample_dir = output_dir / "geometry_opt_sample"
    sample_files = sorted(sample_dir.glob("*.pt"))
    if not sample_files:
        raise FileNotFoundError(f"No sample files found in {sample_dir}")

    updated = 0
    skipped = 0
    failed = 0

    for sample_file in tqdm(sample_files, desc="Backfilling KS geometry offsets"):
        basename = sample_file.stem
        ks_file = ks_opt_dir / f"{basename}.chk"

        try:
            sample = torch.load(sample_file, weights_only=False, map_location=torch.device("cpu"))
            converged = as_scalar_bool(
                getattr(sample, "geomopt_converged", False), name="geomopt_converged"
            )
            if not converged:
                skipped += 1
                continue

            if hasattr(sample, KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE):
                already_available = as_scalar_bool(
                    getattr(sample, KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE),
                    name=KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE,
                )
            else:
                already_available = False
            if already_available and not overwrite:
                skipped += 1
                continue

            ks_mol, ks_data = load_scf_with_optional_data(ks_file)
            if not isinstance(ks_data, dict) or "e_tot" not in ks_data:
                raise ValueError(f"KS checkpoint does not contain scf/e_tot: {ks_file}")
            ks_energy_at_ks_geometry = as_scalar_float(ks_data["e_tot"], name="ks_data['e_tot']")

            ml_geom_mol = _build_ml_geometry_mol(ks_mol, sample.mol)
            ks_energy_at_ml_geometry, sp_converged, runtime_s = _compute_ks_single_point_energy(
                ml_geom_mol, ks_cfg
            )
            set_ks_geometry_energy_offset_result(
                sample,
                ks_energy_at_ml_geometry=ks_energy_at_ml_geometry,
                ks_energy_at_ml_geometry_converged=sp_converged,
                ks_energy_at_ks_geometry=ks_energy_at_ks_geometry,
                scf_time=runtime_s,
            )

            torch.save(sample, sample_file)
            updated += 1
        except Exception as exc:
            logger.warning(f"Failed for {sample_file.name}: {exc}")
            failed += 1

    return updated, skipped, failed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill KS geometry energy offsets E_GEO = E_KS(ML geometry) - "
            "E_KS(KS geometry) for existing geometry optimization samples."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Geometry optimization output directory containing geometry_opt_sample/.",
    )
    parser.add_argument(
        "ks_path",
        type=Path,
        help="Dataset root containing ks-opt/ or the ks-opt/ directory itself.",
    )
    parser.add_argument(
        "--ks_config",
        type=Path,
        default=None,
        help="Path to KS config. Defaults to configs/datagen/kohn_sham/default.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite already available KS geometry energy offsets.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate geometry plots and publication JSON after backfilling.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ks_opt_dir = _resolve_ks_opt_dir(Path(args.ks_path))
    if not ks_opt_dir.exists():
        raise FileNotFoundError(f"KS checkpoint directory does not exist: {ks_opt_dir}")

    updated, skipped, failed = backfill_ks_geometry_energy_offsets(
        output_dir=output_dir,
        ks_opt_dir=ks_opt_dir,
        ks_cfg=_load_ks_cfg(args.ks_config),
        overwrite=args.overwrite,
    )
    logger.info(
        "Completed KS geometry energy offset backfill. "
        f"updated={updated}, skipped={skipped}, failed={failed}"
    )

    if args.replot:
        from mldft.utils.plotting.plot_geometry_optimization import plot_data_from_path

        plot_data_from_path(output_dir, ks_opt_dir.parent)
        logger.info("Regenerated geometry plots and publication JSON.")

    if failed > 0:
        raise RuntimeError(f"Failed to backfill KS geometry energy offset for {failed} samples.")


if __name__ == "__main__":
    main()
