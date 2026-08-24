"""Evaluate analytical ML forces at coefficients obtained from fresh OFDFT runs.

This task is the density-optimized counterpart of ``analytical_forces``.  It does
not load previously stored optimized coefficients.  Instead, each selected label
sample is density-optimized with ``density_optimization_with_label`` and the
resulting final OFDFT coefficients are used for the analytical nuclear-gradient
calculation.

The reference zarr files contain nuclear gradients generated from the label
reference density. ``report_forces`` only changes result metadata; signed vectors
are left as nuclear gradients because this task emits MAE/RMSE magnitudes only.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.convert_transforms import PrepareForDensityOptimization
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.density_optimization import density_optimization_with_label
from mldft.ofdft.evaluation.analytical_forces import (
    CSV_PER_MOLECULE,
    SUMMARY_YAML,
    NonConvergedOFDftError,
    _base_result_row,
    _cfg_get,
    _configure_logger,
    _denop_cfg_get,
    _evaluate_sample_rows,
    _failure_row,
    _gradient_sample,
    _make_model_gradient_state,
    _ofdft_diagnostic_metadata,
    _prepare_evaluation_dataset,
    _raise_if_no_successful_rows,
    _resolve_split_file_path,
    _save_resolved_cli_config,
    _to_numpy_vector,
    _write_evaluation_outputs,
    compute_force_error_statistics,
    load_reference_gradients,
    reference_path_for_label_path,
    resolve_reference_model_key,
    summarize_metric_rows,
)
from mldft.ofdft.functional_factory import requires_grid
from mldft.ofdft.nuclear_gradients import get_gradient_method
from mldft.utils import extras
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path
from mldft.utils.instantiators import instantiate_model
from mldft.utils.omegaconf_resolvers import to_no_basis_transforms_dataset_statistics
from mldft.utils.sad_guesser import SADNormalizationMode
from mldft.utils.sample_generator import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype

DEFAULT_GRADIENT_METHOD = "analytical_generator"
DEFAULT_KS_BASIS = "6-31G(2df,p)"
DEFAULT_MAX_XC_MEMORY = 4000
DEFAULT_CONVERGENCE_CRITERION = "last_iter"
DEFAULT_INITIALIZATION = "minao"


def _make_denop_transforms(
    model_config: Any,
    basis_info: Any,
    grid_level: int,
    grid_prune: str,
    negative_integrated_density_penalty_weight: float | None,
) -> Any:
    """Build dataset transforms required by density optimization."""
    transforms = instantiate(model_config.data.transforms)
    add_grid = requires_grid(
        model_config.data.target_key,
        negative_integrated_density_penalty_weight,
    )
    transforms.pre_transforms = [
        transform
        for transform in transforms.pre_transforms
        if not isinstance(transform, PrepareForDensityOptimization)
    ]
    transforms.pre_transforms.insert(
        0,
        PrepareForDensityOptimization(basis_info, add_grid, grid_level, grid_prune),
    )
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False
    return transforms


@dataclass(slots=True)
class OFDFTForceEvaluationContext:
    """Container for models, factories, optimizer settings, and gradient callables."""

    sample_generator: SampleGenerator
    total_gradient_fn: Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]
    model_gradient_fn: Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]
    denop_func_factory: Any
    proj_minao_module: Any
    optimizer: Any
    dataset_statistics_path: Path | str | None
    ks_basis: str
    initialization: str
    normalize_initial_guess: bool
    max_xc_memory: int
    convergence_criterion: str
    sad_guess_kwargs: Mapping[str, Any] | None = None


def evaluate_gradient_sample_from_ofdft(
    sample: Any,
    label_path: Path | str,
    context: OFDFTForceEvaluationContext,
    nuclear_gradients_dirname: str = "nuclear_gradients",
    reference_total_key: str = "nuc_grad_tot",
    reference_model_key: str = "nuc_grad_kin_plus_xc",
    report_forces: bool = True,
) -> dict[str, Any]:
    """Run OFDFT for one sample and evaluate analytical forces at final coefficients."""
    label_path = Path(label_path)
    reference_path = reference_path_for_label_path(label_path, nuclear_gradients_dirname)
    ref_total, ref_model = load_reference_gradients(
        reference_path,
        total_key=reference_total_key,
        model_key=reference_model_key,
    )
    denop_result = run_ofdft_for_sample(sample, context)
    coeffs = denop_result["coeffs"]
    gradient_sample = _gradient_sample(sample, context.sample_generator)

    pred_total = _to_numpy_vector(context.total_gradient_fn(gradient_sample, coeffs))
    pred_model = _to_numpy_vector(context.model_gradient_fn(gradient_sample, coeffs))

    row = _base_result_row(
        sample,
        label_path,
        reference_path,
        reference_total_key,
        reference_model_key,
        report_forces,
    )
    row["evaluation_coefficients"] = "ofdft_optimized"
    row.update(_ofdft_diagnostic_metadata(denop_result.get("metric_dict", {})))
    row.update(compute_force_error_statistics(pred_total, ref_total, row["n_atom"], "total"))
    row.update(compute_force_error_statistics(pred_model, ref_model, row["n_atom"], "model"))
    return row


def run_ofdft_for_sample(
    sample: Any,
    context: OFDFTForceEvaluationContext,
) -> dict[str, Any]:
    """Run density optimization and return final AO-basis coefficients plus diagnostics."""
    callback = ConvergenceCallback()
    result = density_optimization_with_label(
        sample,
        mol=sample.mol,
        optimizer=context.optimizer,
        func_factory=context.denop_func_factory,
        callback=callback,
        max_xc_memory=context.max_xc_memory,
        normalize_initial_guess=context.normalize_initial_guess,
        ks_basis=context.ks_basis,
        proj_minao_module=context.proj_minao_module,
        initialization=context.initialization,
        sad_guess_kwargs=dict(context.sad_guess_kwargs or {}),
        convergence_criterion=context.convergence_criterion,
    )
    metric_dict, callback, energies_label, _energy_functional = result
    if callback.stopping_index is None:
        raise RuntimeError("Density optimization did not report a stopping index.")
    if (
        metric_dict.get("gs_gradient_norm") is None
        or metric_dict.get("gs_gradient_norm") > context.optimizer.convergence_tolerance
    ):
        raise NonConvergedOFDftError("Density optimization did not converge.", metric_dict)
    coeffs = callback.coeffs[callback.stopping_index]
    metric_dict["converged"] = True
    return {"coeffs": coeffs, "metric_dict": metric_dict}


def _sad_guess_kwargs(
    initialization: str,
    dataset_statistics_path: Path | str | None,
    sample_generator: SampleGenerator,
) -> dict[str, Any]:
    """Build SAD guess keyword arguments when the OFDFT initialization requires them."""
    if initialization not in {"sad", "sad_transformed"}:
        return {}
    if dataset_statistics_path is None:
        raise RuntimeError("SAD initialization requires dataset statistics.")
    return {
        "dataset_statistics": DatasetStatistics(dataset_statistics_path),
        "normalization_mode": SADNormalizationMode.PER_ATOM_WEIGHTED,
        "basis_info": sample_generator.basis_info,
        "weigher_key": "ground_state_only",
        "spherical_average": True,
    }


def _make_ofdft_context(
    ckpt_path: Path | str,
    guess_ckpt_path: Path | str | None,
    model_config: Any,
    optimizer: Any,
    device: torch.device,
    model_dtype: torch.dtype,
    deterministic: bool,
    batch_size: int,
    batch_timeout: float,
    xc_functional: str,
    negative_integrated_density_penalty_weight: float,
    gradient_method: str,
    dataset_statistics_path: Path | str | None,
    ks_basis: str,
    initialization: str,
    normalize_initial_guess: bool,
    max_xc_memory: int,
    convergence_criterion: str,
) -> OFDFTForceEvaluationContext:
    """Instantiate models, factories, sample generator, and analytical gradient callables."""
    state = _make_model_gradient_state(
        ckpt_path,
        model_config,
        device,
        model_dtype,
        deterministic,
        batch_size,
        batch_timeout,
        xc_functional,
        negative_integrated_density_penalty_weight,
    )
    if guess_ckpt_path is None:
        proj_minao_module = state.lightning_module
    else:
        proj_minao_module = instantiate_model(
            Path(guess_ckpt_path),
            str(device),
            model_dtype,
            deterministic,
        )
    total_gradient_fn = get_gradient_method(
        gradient_method,
        state.sample_generator,
        state.total_factory,
        device=str(device),
    )
    model_gradient_fn = get_gradient_method(
        gradient_method,
        state.sample_generator,
        state.model_factory,
        device=str(device),
        include_hellmann=False,
    )
    return OFDFTForceEvaluationContext(
        sample_generator=state.sample_generator,
        total_gradient_fn=total_gradient_fn,
        model_gradient_fn=model_gradient_fn,
        denop_func_factory=state.total_factory,
        proj_minao_module=proj_minao_module,
        optimizer=optimizer,
        dataset_statistics_path=dataset_statistics_path,
        ks_basis=ks_basis,
        initialization=initialization,
        normalize_initial_guess=normalize_initial_guess,
        max_xc_memory=max_xc_memory,
        convergence_criterion=convergence_criterion,
        sad_guess_kwargs=_sad_guess_kwargs(
            initialization,
            dataset_statistics_path,
            state.sample_generator,
        ),
    )


def _dataset_statistics_path(model_config: Any, initialization: str) -> Path | str | None:
    """Resolve dataset-statistics path, including the SAD no-transform variant."""
    if initialization not in {"sad", "sad_transformed"}:
        return None
    dataset_statistics_path = model_config.data.dataset_statistics.path
    if initialization == "sad":
        return to_no_basis_transforms_dataset_statistics(
            dataset_statistics_path,
            model_config.data.transforms.name,
        )
    return dataset_statistics_path


def _evaluate_dataset_rows_from_ofdft(
    dataset: Any,
    paths: Sequence[Path],
    context: OFDFTForceEvaluationContext,
    nuclear_gradients_dirname: str,
    reference_total_key: str,
    reference_model_key: str,
    report_forces: bool,
    fail_fast: bool,
    torch_threads_per_worker: int,
    device: str,
) -> list[dict[str, Any]]:
    """Run OFDFT and analytical force evaluation for all selected dataset rows."""

    def evaluate(label_path: Path, sample: Any) -> Mapping[str, Any]:
        return evaluate_gradient_sample_from_ofdft(
            sample=sample,
            label_path=label_path,
            context=context,
            nuclear_gradients_dirname=nuclear_gradients_dirname,
            reference_total_key=reference_total_key,
            reference_model_key=reference_model_key,
            report_forces=report_forces,
        )

    def failure_row(label_path: Path, error: Exception) -> Mapping[str, Any]:
        return _failure_row(
            label_path,
            error,
            reference_total_key,
            reference_model_key,
            nuclear_gradients_dirname,
            report_forces,
        )

    def log_success(sample_rows: list[Mapping[str, Any]]) -> None:
        row = sample_rows[0]
        logger.info(
            f"{row['mol_id']:<12} {row['sum_formula']:<16} "
            f"iters={row['ofdft_iterations']:<5} "
            f"total_MAE={row['total_mae_ha_per_bohr']:.4e} Ha/Bohr "
            f"model_MAE={row['model_mae_ha_per_bohr']:.4e} Ha/Bohr"
        )

    return _evaluate_sample_rows(
        dataset,
        paths,
        "ofdft force eval",
        evaluate,
        failure_row,
        log_success,
        fail_fast,
        torch_threads_per_worker,
        device,
    )


@set_default_torch_dtype(OFDFT_DTYPE)
def run_gradient_evaluation_from_ofdft(
    ckpt_path: Path | str,
    model_config: DictConfig,
    optimizer: Any,
    split_file_path: Path | str,
    split: str = "val",
    guess_ckpt_path: Path | str | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    batch_timeout: float = 0.05,
    model_dtype: torch.dtype = OFDFT_DTYPE,
    deterministic: bool = True,
    xc_functional: str = "PBE",
    ks_basis: str = DEFAULT_KS_BASIS,
    initialization: str = DEFAULT_INITIALIZATION,
    normalize_initial_guess: bool = True,
    max_xc_memory: int = DEFAULT_MAX_XC_MEMORY,
    negative_integrated_density_penalty_weight: float = 0.0,
    grid_level: int = 3,
    grid_prune: str = "nwchem_prune",
    convergence_criterion: str = DEFAULT_CONVERGENCE_CRITERION,
    n_molecules: int = 1,
    molecule_choice: str | list[int] = "first_n",
    molecule_ids: Sequence[int] | None = None,
    seed: int | None = 0,
    nuclear_gradients_dirname: str = "nuclear_gradients",
    reference_total_key: str = "nuc_grad_tot",
    reference_model_key: str | None = "auto",
    gradient_method: str = DEFAULT_GRADIENT_METHOD,
    report_forces: bool = True,
    save_dir: Path | str | None = None,
    fail_fast: bool = False,
    torch_threads_per_worker: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run fresh OFDFT optimizations and evaluate analytical forces at final coefficients."""
    device = torch.device(device)
    split_file_path = Path(split_file_path)
    save_dir = Path(save_dir or Path.cwd())
    save_dir.mkdir(parents=True, exist_ok=True)

    def transforms_factory(basis_info: Any) -> Any:
        return _make_denop_transforms(
            model_config=model_config,
            basis_info=basis_info,
            grid_level=grid_level,
            grid_prune=grid_prune,
            negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        )

    dataset, paths = _prepare_evaluation_dataset(
        model_config,
        split_file_path,
        split,
        n_molecules,
        molecule_choice,
        molecule_ids,
        seed,
        transforms_factory,
    )
    initialization = str(initialization)
    context = _make_ofdft_context(
        ckpt_path=ckpt_path,
        guess_ckpt_path=guess_ckpt_path,
        model_config=model_config,
        optimizer=optimizer,
        device=device,
        model_dtype=model_dtype,
        deterministic=deterministic,
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        xc_functional=xc_functional,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        gradient_method=gradient_method,
        dataset_statistics_path=_dataset_statistics_path(model_config, initialization),
        ks_basis=ks_basis,
        initialization=initialization,
        normalize_initial_guess=normalize_initial_guess,
        max_xc_memory=max_xc_memory,
        convergence_criterion=convergence_criterion,
    )
    resolved_reference_model_key = resolve_reference_model_key(model_config, reference_model_key)
    rows = _evaluate_dataset_rows_from_ofdft(
        dataset=dataset,
        paths=paths,
        context=context,
        nuclear_gradients_dirname=nuclear_gradients_dirname,
        reference_total_key=reference_total_key,
        reference_model_key=resolved_reference_model_key,
        report_forces=report_forces,
        fail_fast=fail_fast,
        torch_threads_per_worker=torch_threads_per_worker,
        device=str(device),
    )
    _raise_if_no_successful_rows(rows)
    summary = summarize_metric_rows(rows)
    summary["evaluation_coefficients"] = "ofdft_optimized"
    _write_evaluation_outputs(save_dir, CSV_PER_MOLECULE, SUMMARY_YAML, rows, summary)
    logger.success(f"OFDFT analytical-force evaluation saved to {save_dir}")
    return rows, summary


def _guess_checkpoint_path(cfg: DictConfig) -> Path | None:
    """Resolve the optional separate checkpoint used for projected MINAO guesses."""
    if cfg.get("guess_path") is None:
        return None
    return run_to_checkpoint_path(cfg.guess_path, cfg.get("guess_ckpt_choice", "last"))


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/ofdft",
    config_name="analytical_forces_at_gs.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for OFDFT-coefficient analytical force evaluation."""
    load_dotenv()
    extras(cfg)

    log_file = Path(str(cfg.get("log_file"))) if cfg.get("log_file") is not None else None
    _configure_logger(log_file)

    run_path = parse_run_path(cfg.run_path)
    model_config = OmegaConf.load(run_path / "hparams.yaml")
    resolved_model_config = OmegaConf.load(run_path / "hparams_resolved.yaml")
    split_file_path = _resolve_split_file_path(
        cfg.get("split_file_path"), model_config, resolved_model_config
    )
    ckpt_path = run_to_checkpoint_path(run_path, cfg.ckpt_choice)
    guess_ckpt_path = _guess_checkpoint_path(cfg)
    optimizer = instantiate(cfg.get("density_optimizer"))

    with open_dict(cfg):
        cfg.run_path = str(run_path)
        cfg.ckpt_path = str(ckpt_path)
        cfg.guess_ckpt_path = str(guess_ckpt_path)
        cfg.split_file_path = str(split_file_path)
    _save_resolved_cli_config(cfg, cfg.get("config_file"), log_file)

    run_gradient_evaluation_from_ofdft(
        ckpt_path=ckpt_path,
        model_config=model_config,
        optimizer=optimizer,
        split_file_path=split_file_path,
        split=_cfg_get(cfg, "split", "val"),
        guess_ckpt_path=guess_ckpt_path,
        device=_cfg_get(cfg, "device", "cpu"),
        batch_size=_cfg_get(cfg, "batch_size", 1),
        batch_timeout=_cfg_get(cfg, "batch_timeout", 0.05),
        model_dtype=str_to_torch_float_dtype(_cfg_get(cfg, "model_dtype", OFDFT_DTYPE)),
        deterministic=_cfg_get(cfg, "deterministic", True),
        xc_functional=_denop_cfg_get(cfg, "xc_functional", "PBE"),
        ks_basis=_denop_cfg_get(cfg, "ks_basis", DEFAULT_KS_BASIS),
        initialization=_denop_cfg_get(cfg, "initialization", DEFAULT_INITIALIZATION),
        normalize_initial_guess=_denop_cfg_get(cfg, "normalize_initial_guess", True),
        max_xc_memory=_denop_cfg_get(cfg, "max_xc_memory", DEFAULT_MAX_XC_MEMORY),
        negative_integrated_density_penalty_weight=_denop_cfg_get(
            cfg,
            "negative_integrated_density_penalty_weight",
            0.0,
        ),
        grid_level=_denop_cfg_get(cfg, "grid_level", 3),
        grid_prune=_denop_cfg_get(cfg, "grid_prune", "nwchem_prune"),
        convergence_criterion=_denop_cfg_get(
            cfg,
            "convergence_criterion",
            DEFAULT_CONVERGENCE_CRITERION,
        ),
        n_molecules=_cfg_get(cfg, "n_molecules", 1),
        molecule_choice=_cfg_get(cfg, "molecule_choice", "first_n"),
        molecule_ids=_cfg_get(cfg, "molecule_ids", []),
        seed=_cfg_get(cfg, "seed", 0),
        nuclear_gradients_dirname=_cfg_get(cfg, "nuclear_gradients_dirname", "nuclear_gradients"),
        reference_total_key=_cfg_get(cfg, "reference_total_key", "nuc_grad_tot"),
        reference_model_key=_cfg_get(cfg, "reference_model_key", "auto"),
        gradient_method=_cfg_get(cfg, "gradient_method", DEFAULT_GRADIENT_METHOD),
        report_forces=_cfg_get(cfg, "report_forces", True),
        save_dir=log_file.parent if log_file is not None else Path.cwd(),
        fail_fast=_cfg_get(cfg, "fail_fast", False),
        torch_threads_per_worker=_cfg_get(cfg, "torch_threads_per_worker", 1),
    )


if __name__ == "__main__":
    main()
