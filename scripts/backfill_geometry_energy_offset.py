import argparse
from pathlib import Path

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft.density_optimization import OFDFT_KWARGS
from mldft.ofdft.energies import Energies
from mldft.ofdft.geometry_energy_offset import (
    GEOMETRY_ENERGY_OFFSET_AVAILABLE,
    set_geometry_energy_offset_result,
)
from mldft.ofdft.optimizer import Optimizer
from mldft.ofdft.run_geometry_optimization import compute_ml_single_point_energy
from mldft.utils.conversions import as_scalar_bool, as_scalar_float
from mldft.utils.environ import (
    parse_run_path,
)
from mldft.utils.omegaconf_resolvers import to_no_basis_transforms_dataset_statistics
from mldft.utils.plotting.plot_geometry_optimization import plot_data_from_path
from mldft.utils.pyscf_calculations import load_scf_with_optional_data
from mldft.utils.sad_guesser import SADNormalizationMode
from mldft.utils.sample_generator import SampleGenerator


def _resolve_ks_opt_dir(ks_path: Path) -> Path:
    """Resolve the path to the KS optimization directory.

    Args:
        ks_path: Either the dataset root containing ``ks-opt`` or the ``ks-opt`` directory.

    Returns:
        Path to ``ks-opt``.
    """
    if ks_path.name == "ks-opt":
        return ks_path
    return ks_path / "ks-opt"


def _extract_of_energy(sample) -> float:
    """Extract the final ML energy from a geometry optimization sample.

    Args:
        sample: Geometry optimization sample.

    Returns:
        Final total ML energy in Hartree.
    """
    if hasattr(sample, "energy"):
        if isinstance(sample.energy, Energies):
            return as_scalar_float(sample.energy.total_energy, name="sample.energy.total_energy")
        else:
            return as_scalar_float(sample.energy, name="sample.energy")

    if hasattr(sample, "geomopt_energies") and len(sample.geomopt_energies) > 0:
        return as_scalar_float(sample.geomopt_energies[-1], name="sample.geomopt_energies[-1]")

    raise ValueError("Sample does not contain a final energy value.")


def _build_ofdft_kwargs_from_cfg(cfg: DictConfig, sample_generator: SampleGenerator) -> dict:
    """Build OFDFT kwargs from a geometry-optimization run config.

    Args:
        cfg: Geometry-optimization run config.

    Returns:
        Dictionary of kwargs accepted by ``density_optimization``.
    """
    denop_cfg = cfg.get("denop_settings")
    ofdft_kwargs = {key: value for key, value in denop_cfg.items() if key in OFDFT_KWARGS}
    ofdft_kwargs["disable_pbar"] = cfg.get("disable_pbar", True)

    if ofdft_kwargs["initialization"] in ["sad", "sad_transformed"]:
        dataset_statistics_path = sample_generator.model_config.data.dataset_statistics.path
        if ofdft_kwargs["initialization"] == "sad":
            dataset_statistics_path = to_no_basis_transforms_dataset_statistics(
                dataset_statistics_path,
                sample_generator.model_config.data.transforms.name,
            )
        sad_guess_kwargs = dict(
            dataset_statistics=DatasetStatistics(dataset_statistics_path),
            normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
            basis_info=sample_generator.basis_info,
            weigher_key="ground_state_only",
            spherical_average=True,
        )
    else:
        sad_guess_kwargs = {}

    ofdft_kwargs["sad_guess_kwargs"] = sad_guess_kwargs
    return ofdft_kwargs


def backfill_geometry_energy_offsets(
    output_dir: Path,
    ks_opt_dir: Path,
    sample_generator: SampleGenerator,
    functional_factory,
    optimizer: Optimizer,
    ofdft_kwargs: dict,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Backfill geometry energy offsets for saved geometry optimization samples.

    Args:
        output_dir: Geometry optimization output directory containing ``geometry_opt_sample``.
        ks_opt_dir: Directory containing ``*.chk`` files for KS minima.
        sample_generator: Sample generator configured for the model checkpoint.
        functional_factory: Functional factory aligned with ``sample_generator``.
        optimizer: Density optimizer for single-point evaluations.
        ofdft_kwargs: Arguments passed to the OFDFT single-point calculation.
        overwrite: Whether to overwrite already present offsets.

    Returns:
        Tuple of ``(updated, skipped, failed)``.
    """
    sample_dir = output_dir / "geometry_opt_sample"
    sample_files = sorted(sample_dir.glob("*.pt"))
    if not sample_files:
        raise FileNotFoundError(f"No sample files found in {sample_dir}")

    updated = 0
    skipped = 0
    failed = 0

    for sample_file in tqdm(sample_files, desc="Backfilling geometry offsets"):
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

            if hasattr(sample, GEOMETRY_ENERGY_OFFSET_AVAILABLE):
                already_available = as_scalar_bool(
                    getattr(sample, GEOMETRY_ENERGY_OFFSET_AVAILABLE),
                    name=GEOMETRY_ENERGY_OFFSET_AVAILABLE,
                )
            else:
                already_available = False
            if already_available and not overwrite:
                skipped += 1
                continue

            if not ks_file.exists():
                raise FileNotFoundError(f"Missing KS checkpoint: {ks_file}")

            ks_mol, _ = load_scf_with_optional_data(ks_file)
            ml_energy_at_ks_geometry, sp_converged, runtime_s = compute_ml_single_point_energy(
                mol=ks_mol,
                sample_generator=sample_generator,
                functional_factory=functional_factory,
                optimizer=optimizer,
                ofdft_kwargs=ofdft_kwargs,
            )
            ml_energy_at_ml_geometry = _extract_of_energy(sample)
            geometry_energy_offset = (
                ml_energy_at_ml_geometry - ml_energy_at_ks_geometry
                if sp_converged
                else float("nan")
            )

            set_geometry_energy_offset_result(
                sample,
                ml_energy_at_ks_geometry=ml_energy_at_ks_geometry,
                ml_energy_at_ks_geometry_converged=sp_converged,
                denop_time=runtime_s,
                geometry_energy_offset=geometry_energy_offset,
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
            "Backfill geometry energy offsets E_GEO = E_ML(A) - E_ML(B) for existing "
            "geometry optimization samples."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Geometry optimization output directory containing geometry_opt_sample/ and config.yaml.",
    )
    parser.add_argument(
        "ks_path",
        type=Path,
        help="Dataset root containing ks-opt/ or the ks-opt/ directory itself.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to geometry-optimization config.yaml. Defaults to <output_dir>/config.yaml.",
    )
    parser.add_argument(
        "--run_path",
        type=Path,
        default=None,
        help="Override model run path from config.",
    )
    parser.add_argument(
        "--ckpt_choice",
        type=str,
        default=None,
        help="Override checkpoint choice from config (e.g. last, best_energy).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for model inference (defaults to config device).",
    )
    parser.add_argument(
        "--transform_device",
        type=str,
        default=None,
        help="Override transform device (defaults to config transform_device).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite already available geometry energy offsets.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate geometry plots and publication JSON after backfilling.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = Path(args.config) if args.config else output_dir / ".hydra/config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    run_path_cfg = cfg.get("run_path")
    if args.run_path is not None:
        run_path = args.run_path
    elif run_path_cfg is not None:
        if isinstance(run_path_cfg, str):
            run_path = Path(run_path_cfg)
        else:
            run_path = run_path_cfg
    else:
        raise ValueError("Missing run_path. Set it in config.yaml or pass --run_path.")
    run_path = parse_run_path(run_path)
    ckpt_choice = args.ckpt_choice or cfg.get("ckpt_choice", "last")
    device = args.device or cfg.get("device", "cpu")
    transform_device = args.transform_device or cfg.get("transform_device", "cpu")
    ks_opt_dir = _resolve_ks_opt_dir(Path(args.ks_path))

    if not ks_opt_dir.exists():
        raise FileNotFoundError(f"KS checkpoint directory does not exist: {ks_opt_dir}")

    optimizer = instantiate(cfg.get("density_optimizer"))
    negative_integrated_density_penalty_weight = cfg.get("denop_settings").get(
        "negative_integrated_density_penalty_weight", 0.0
    )

    sample_generator = SampleGenerator.from_run_path(
        run_path,
        device=device,
        transform_device=transform_device,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        ckpt_choice=ckpt_choice,
    )
    functional_factory = sample_generator.get_functional_factory()

    ofdft_kwargs = _build_ofdft_kwargs_from_cfg(cfg, sample_generator)

    updated, skipped, failed = backfill_geometry_energy_offsets(
        output_dir=output_dir,
        ks_opt_dir=ks_opt_dir,
        sample_generator=sample_generator,
        functional_factory=functional_factory,
        optimizer=optimizer,
        ofdft_kwargs=ofdft_kwargs,
        overwrite=args.overwrite,
    )

    logger.info(f"Completed backfill. updated={updated}, skipped={skipped}, failed={failed}")

    if args.replot:
        plot_data_from_path(output_dir, ks_opt_dir.parent)
        logger.info("Regenerated geometry plots and publication JSON.")

    if failed > 0:
        raise RuntimeError(f"Failed to backfill geometry energy offset for {failed} samples.")


if __name__ == "__main__":
    main()
