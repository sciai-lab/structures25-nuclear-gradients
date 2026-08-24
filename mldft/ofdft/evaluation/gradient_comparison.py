"""Compare analytical, fixed-coefficient numerical, and fully numerical gradients."""

import os
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
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.evaluation.analytical_forces import (
    _base_result_row,
    _cfg_get,
    _configure_logger,
    _denop_cfg_get,
    _evaluate_sample_rows,
    _failure_row,
    _gradient_sample,
    _ofdft_diagnostic_metadata,
    _prepare_evaluation_dataset,
    _raise_if_no_successful_rows,
    _resolve_split_file_path,
    _sample_n_atoms,
    _save_resolved_cli_config,
    _to_numpy_vector,
    _write_evaluation_outputs,
    compute_force_error_statistics,
    reference_path_for_label_path,
    summarize_metric_rows,
)
from mldft.ofdft.evaluation.analytical_forces_at_gs import (
    DEFAULT_CONVERGENCE_CRITERION,
    DEFAULT_INITIALIZATION,
    DEFAULT_KS_BASIS,
    DEFAULT_MAX_XC_MEMORY,
    _dataset_statistics_path,
    _guess_checkpoint_path,
    _make_denop_transforms,
    _make_ofdft_context,
    run_ofdft_for_sample,
)
from mldft.ofdft.nuclear_gradients import get_gradient_method
from mldft.utils import extras
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path
from mldft.utils.utils import set_default_torch_dtype

CSV_PER_MOLECULE = "gradient_comparison_per_molecule.csv"
SUMMARY_YAML = "gradient_comparison_summary.yaml"
NUCLEAR_GRADIENTS_DIRNAME = "gradient_comparison"
GRADIENT_VECTORS_DIRNAME = "gradient_vectors"
DEFAULT_METHODS = ("analytical_generator", "numerical", "fully_numerical")
DEFAULT_REFERENCE_METHOD = "fully_numerical"


def make_gradient_methods(
    context: Any,
    methods: Sequence[str],
    device: torch.device,
    epsilon: float,
    strict: bool = False,
) -> dict[str, Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]]:
    """Instantiate gradient callables for all requested comparison methods."""
    gradient_methods: dict[str, Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]] = {}
    for method in methods:
        kwargs: dict[str, Any] = {}
        if method == "fully_numerical":
            kwargs = {
                "optimizer": context.optimizer,
                "epsilon": epsilon,
                "max_xc_memory": context.max_xc_memory,
                "normalize_initial_guess": context.normalize_initial_guess,
                "ks_basis": context.ks_basis,
                "proj_minao_module": context.proj_minao_module,
                "sad_guess_kwargs": dict(context.sad_guess_kwargs or {}),
                "strict": strict,
            }
        gradient_methods[method] = get_gradient_method(
            method,
            context.sample_generator,
            context.denop_func_factory,
            device=str(device),
            **kwargs,
        )
    return gradient_methods


def compute_gradient_comparison_error_statistics(
    predicted: np.ndarray | torch.Tensor | Sequence[float],
    reference: np.ndarray | torch.Tensor | Sequence[float],
    n_atoms: int,
    prefix: str,
) -> dict[str, float | int]:
    """Compute component, per-Cartesian, and per-atom MAE comparison metrics."""
    metrics = compute_force_error_statistics(predicted, reference, n_atoms, prefix)
    predicted_arr = _to_numpy_vector(predicted)
    reference_arr = _to_numpy_vector(reference)
    error_by_atom = (predicted_arr - reference_arr).reshape(int(n_atoms), 3)
    metrics[f"{prefix}_per_cartesian_mae_ha_per_bohr"] = float(np.mean(np.abs(error_by_atom)))
    metrics[f"{prefix}_per_atom_mae_ha_per_bohr"] = float(
        np.mean(np.linalg.norm(error_by_atom, axis=1))
    )
    return metrics


def gradient_vector_path_for_label_path(
    label_path: Path | str, gradient_save_dir: Path | str
) -> Path:
    """Return the compressed gradient-vector output path for one label file."""
    label_path = Path(label_path)
    gradient_save_dir = Path(gradient_save_dir)
    name = label_path.name
    for suffix in (".zarr.zip", ".zip", ".zarr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return gradient_save_dir / f"{name}.npz"


def save_gradient_vectors(
    path: Path | str,
    predictions: Mapping[str, np.ndarray],
) -> Path:
    """Save one NumPy array per gradient method to a compressed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in predictions.items()}
    np.savez_compressed(path, **arrays)
    return path


def evaluate_gradient_comparison_sample(
    sample: Any,
    label_path: Path | str,
    context: Any,
    gradient_methods: Mapping[str, Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]],
    reference_method: str = DEFAULT_REFERENCE_METHOD,
    report_forces: bool = True,
    gradient_save_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Run OFDFT once for a sample and compare all configured gradient methods."""
    if reference_method not in gradient_methods:
        raise ValueError(f"Reference method {reference_method!r} is not configured.")

    label_path = Path(label_path)
    ofdft_result = run_ofdft_for_sample(sample, context)
    coeffs = ofdft_result["coeffs"]
    gradient_sample = _gradient_sample(sample, context.sample_generator)

    predictions = {
        method: _to_numpy_vector(gradient_fn(gradient_sample, coeffs))
        for method, gradient_fn in gradient_methods.items()
    }
    gradient_vector_path = None
    if gradient_save_dir is not None:
        gradient_vector_path = save_gradient_vectors(
            gradient_vector_path_for_label_path(label_path, gradient_save_dir),
            predictions,
        )
    reference = predictions[reference_method]
    n_atoms = _sample_n_atoms(sample)
    reference_path = reference_path_for_label_path(label_path, NUCLEAR_GRADIENTS_DIRNAME)
    diagnostic_metadata = _ofdft_diagnostic_metadata(ofdft_result.get("metric_dict", {}))

    pairwise_statistics: dict[tuple[str, str], dict[str, float | int]] = {}
    method_names = list(predictions)
    for first_index, first_method in enumerate(method_names):
        for second_method in method_names[first_index + 1 :]:
            pairwise_statistics[(first_method, second_method)] = (
                compute_gradient_comparison_error_statistics(
                    predictions[first_method],
                    predictions[second_method],
                    n_atoms,
                    "pair",
                )
            )

    rows = []
    for method, predicted in predictions.items():
        row = _base_result_row(
            sample,
            label_path,
            reference_path,
            reference_method,
            reference_method,
            report_forces,
        )
        row.update(
            {
                "method": method,
                "reference_method": reference_method,
                "evaluation_coefficients": "ofdft_optimized",
                "gradient_norm_ha_per_bohr": float(np.linalg.norm(predicted)),
                "gradient_path": (
                    str(gradient_vector_path) if gradient_vector_path is not None else ""
                ),
                "gradient_key": method if gradient_vector_path is not None else "",
            }
        )
        row.update(diagnostic_metadata)
        for comparison_method in predictions:
            if comparison_method == method:
                continue
            pair_key = (method, comparison_method)
            if pair_key not in pairwise_statistics:
                pair_key = (comparison_method, method)
            stats = pairwise_statistics[pair_key]
            row.update(
                {
                    key.replace("pair_", f"vs_{comparison_method}_", 1): value
                    for key, value in stats.items()
                }
            )
        row.update(
            compute_gradient_comparison_error_statistics(
                predicted,
                reference,
                n_atoms,
                "vs_reference",
            )
        )
        rows.append(row)
    return rows


def _evaluate_dataset_rows_from_gradient_comparison(
    dataset: Any,
    paths: Sequence[Path],
    context: Any,
    gradient_methods: Mapping[str, Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray]],
    reference_method: str,
    report_forces: bool,
    fail_fast: bool,
    torch_threads_per_worker: int,
    device: str,
    gradient_save_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate selected rows using shared recoverable-failure handling."""

    def evaluate(label_path: Path, sample: Any) -> Sequence[Mapping[str, Any]]:
        return evaluate_gradient_comparison_sample(
            sample=sample,
            label_path=label_path,
            context=context,
            gradient_methods=gradient_methods,
            reference_method=reference_method,
            report_forces=report_forces,
            gradient_save_dir=gradient_save_dir,
        )

    def failure_row(label_path: Path, error: Exception) -> Mapping[str, Any]:
        return _failure_row(
            label_path,
            error,
            reference_method,
            reference_method,
            NUCLEAR_GRADIENTS_DIRNAME,
            report_forces,
        )

    def log_success(sample_rows: list[Mapping[str, Any]]) -> None:
        method_summaries = ", ".join(
            f"{row['method']}={row['vs_reference_mae_ha_per_bohr']:.4e} Ha/Bohr"
            for row in sample_rows
        )
        logger.info(f"{sample_rows[0]['mol_id']:<12} {method_summaries}")

    return _evaluate_sample_rows(
        dataset,
        paths,
        "gradient comparison",
        evaluate,
        failure_row,
        log_success,
        fail_fast,
        torch_threads_per_worker,
        device,
    )


@set_default_torch_dtype(OFDFT_DTYPE)
def run_gradient_comparison(
    ckpt_path: Path | str,
    model_config: DictConfig,
    optimizer: Any,
    split_file_path: Path | str,
    methods: Sequence[str] = DEFAULT_METHODS,
    reference_method: str = DEFAULT_REFERENCE_METHOD,
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
    epsilon: float = 1e-4,
    fully_numerical_strict: bool = False,
    report_forces: bool = True,
    save_dir: Path | str | None = None,
    save_gradients: bool = False,
    fail_fast: bool = False,
    torch_threads_per_worker: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run OFDFT and compare configured nuclear-gradient methods."""
    methods = tuple(methods)
    if reference_method not in methods:
        raise ValueError(f"Reference method {reference_method!r} must be in methods.")

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
        gradient_method=methods[0],
        dataset_statistics_path=_dataset_statistics_path(model_config, initialization),
        ks_basis=ks_basis,
        initialization=initialization,
        normalize_initial_guess=normalize_initial_guess,
        max_xc_memory=max_xc_memory,
        convergence_criterion=convergence_criterion,
    )
    gradient_methods = make_gradient_methods(
        context,
        methods=methods,
        device=device,
        epsilon=epsilon,
        strict=fully_numerical_strict,
    )
    gradient_save_dir = save_dir / GRADIENT_VECTORS_DIRNAME if save_gradients else None
    rows = _evaluate_dataset_rows_from_gradient_comparison(
        dataset=dataset,
        paths=paths,
        context=context,
        gradient_methods=gradient_methods,
        reference_method=reference_method,
        report_forces=report_forces,
        fail_fast=fail_fast,
        torch_threads_per_worker=torch_threads_per_worker,
        device=str(device),
        gradient_save_dir=gradient_save_dir,
    )
    _raise_if_no_successful_rows(rows)
    summary = summarize_metric_rows(rows, metrics=False)
    summary["evaluation_coefficients"] = "ofdft_optimized"
    summary["reference_method"] = reference_method
    summary["methods"] = list(methods)
    _write_evaluation_outputs(save_dir, CSV_PER_MOLECULE, SUMMARY_YAML, rows, summary)
    logger.success(f"Gradient comparison saved to {save_dir}")
    return rows, summary


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/ofdft",
    config_name="gradient_comparison.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for OFDFT gradient-method comparison."""
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

    fully_numerical_cfg = cfg.get("fully_numerical", {}) or {}
    run_gradient_comparison(
        ckpt_path=ckpt_path,
        model_config=model_config,
        optimizer=optimizer,
        split_file_path=split_file_path,
        methods=_cfg_get(cfg, "methods", list(DEFAULT_METHODS)),
        reference_method=_cfg_get(cfg, "reference_method", DEFAULT_REFERENCE_METHOD),
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
        epsilon=_cfg_get(cfg, "epsilon", 1e-4),
        fully_numerical_strict=bool(fully_numerical_cfg.get("strict", False)),
        report_forces=_cfg_get(cfg, "report_forces", True),
        save_dir=log_file.parent if log_file is not None else Path.cwd(),
        save_gradients=_cfg_get(cfg, "save_gradients", False),
        fail_fast=_cfg_get(cfg, "fail_fast", False),
        torch_threads_per_worker=_cfg_get(cfg, "torch_threads_per_worker", 1),
    )


if __name__ == "__main__":
    main()
