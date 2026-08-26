"""Evaluate analytical ML forces against saved nuclear-gradient reference data.

The reference zarr files contain nuclear gradients. ``report_forces`` only changes
result metadata; signed vectors are left as nuclear gradients because this task
emits MAE/RMSE magnitudes only.
"""

import csv
import os
import pickle
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Keep thread-pool defaults consistent with run_density_optimization.py.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import hydra
import numpy as np
import torch
import zarr
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from rootutils import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.dataset import HDF5ShardOFDataset, OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import Representation
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.nuclear_gradients import get_gradient_method
from mldft.ofdft.run_density_optimization import (
    configure_dataset_indices,
    find_molecule_path_by_id,
    get_paths_from_split,
    resolve_torch_threads_per_worker,
)
from mldft.utils import extras
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import (
    get_mldft_data_path,
    parse_run_path,
    run_to_checkpoint_path,
)
from mldft.utils.instantiators import instantiate_model
from mldft.utils.molecules import build_molecule_np
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sample_generator import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype

CSV_PER_MOLECULE = "force_gradient_eval_per_molecule.csv"
SUMMARY_YAML = "force_gradient_eval_summary.yaml"
METRIC_PREFIXES = ("total", "model")
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
HARTREE_TO_MEV = 27211.386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_UNIT_FACTORS = {
    "ha_per_bohr": 1.0,
    "kcal_per_mol_angstrom": HARTREE_TO_KCAL_PER_MOL / BOHR_TO_ANGSTROM,
    "mev_per_angstrom": HARTREE_TO_MEV / BOHR_TO_ANGSTROM,
}
UNIT_LABEL_BY_SUFFIX = {
    "ha_per_bohr": "Ha/Bohr",
    "kcal_per_mol_angstrom": "kcal/(mol A)",
    "mev_per_angstrom": "meV/A",
}
REFERENCE_KEY_BASE_BY_TARGET = {
    "kin": "nuc_grad_kin",
    "kin_plus_xc": "nuc_grad_kin_plus_xc",
    "kin_minus_apbe": "nuc_grad_kin_minus_apbe",
    "kinapbe": "nuc_grad_kinapbe",
    "tot": "nuc_grad_tot",
}
REFERENCE_MODEL_KEY_BY_TARGET = {
    **REFERENCE_KEY_BASE_BY_TARGET,
    **{f"grad_{key}": value for key, value in REFERENCE_KEY_BASE_BY_TARGET.items()},
}
GROUND_STATE_ENERGY_KEYS = (
    "e_electron",
    "e_ext",
    "e_hartree",
    "e_kin",
    "e_kin_plus_xc",
    "e_kin_minus_apbe",
    "e_kinapbe",
    "e_xc",
    "e_tot",
)
CSV_BASE_FIELDS = (
    "mol_id",
    "sum_formula",
    "n_atom",
    "n_heavy_atom",
    "quantity",
    "skipped",
    "skip_reason",
    "reference_total_key",
    "reference_model_key",
    "label_path",
    "reference_path",
)
DATASET_KWARGS_TO_DROP = (
    "hdf5_max_open_shards",
    "label_path",
    "kohn_sham_path",
    "ground_state_coeffs",
    "load_ground_state_gradients",
    "restricted",
)


class NonConvergedOFDftError(ValueError):
    """Raised when density optimization did not converge for a sample."""

    def __init__(self, message: str, metric_dict: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.metric_dict = dict(metric_dict or {})


RECOVERABLE_SAMPLE_ERRORS = (OSError, KeyError, ValueError, NonConvergedOFDftError)
OFDFT_DIAGNOSTIC_SUMMARY_KEYS = (
    "ofdft_iterations",
    "ofdft_energy_error_mha",
    "ofdft_abs_energy_error_mha",
    "ofdft_density_l2_norm",
    "ofdft_gradient_norm",
)


def reference_path_for_label_path(
    label_path: Path | str,
    nuclear_gradients_dirname: str = "nuclear_gradients",
) -> Path:
    """Return the reference-gradient zarr path corresponding to a label path."""
    label_path = Path(label_path)
    parent = label_path.parent.parent if label_path.parent.name == "labels" else label_path.parent
    return parent / nuclear_gradients_dirname / label_path.name


def load_reference_gradients(
    reference_path: Path | str,
    total_key: str = "nuc_grad_tot",
    model_key: str = "nuc_grad_kin_plus_xc",
) -> tuple[np.ndarray, np.ndarray]:
    """Load total and model-target reference nuclear gradients from a zarr zip file."""
    reference_path = Path(reference_path)
    with zarr.ZipStore(str(reference_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        missing = [key for key in (total_key, model_key) if key not in root.keys()]
        if missing:
            raise KeyError(
                f"Reference file {reference_path} is missing keys {missing}. "
                f"Available keys: {sorted(root.keys())}"
            )
        return _zarr_array(root, total_key), _zarr_array(root, model_key)


def _zarr_array(root: Any, key: str) -> np.ndarray:
    """Return one zarr dataset as a flattened float64 NumPy vector."""
    return np.asarray(root[key][...], dtype=np.float64).reshape(-1)


def resolve_reference_model_key(
    model_config: Any,
    reference_model_key: str | None = "auto",
) -> str:
    """Resolve the model-only reference key from the training target key."""
    if reference_model_key not in (None, "auto"):
        return str(reference_model_key)

    target_key = str(model_config.data.target_key)
    if target_key not in REFERENCE_MODEL_KEY_BY_TARGET:
        raise ValueError(
            f"Cannot infer reference model-gradient key for target_key={target_key!r}. "
            f"Known targets: {sorted(REFERENCE_MODEL_KEY_BY_TARGET)}"
        )
    return REFERENCE_MODEL_KEY_BY_TARGET[target_key]


def compute_force_error_statistics(
    predicted: np.ndarray | torch.Tensor | Sequence[float],
    reference: np.ndarray | torch.Tensor | Sequence[float],
    n_atoms: int,
    prefix: str,
) -> dict[str, float | int]:
    """Compute per-sample MAE/RMSE in Ha/Bohr, kcal/(mol A), and meV/A."""
    predicted_arr = _to_numpy_vector(predicted)
    reference_arr = _to_numpy_vector(reference)
    _validate_gradient_vectors(predicted_arr, reference_arr, n_atoms)

    error = predicted_arr - reference_arr
    mae_ha_per_bohr = float(np.mean(np.abs(error)))
    rmse_ha_per_bohr = float(np.sqrt(np.mean(error**2)))
    metrics: dict[str, float | int] = {f"{prefix}_n_components": int(error.size)}
    for unit_suffix, factor in FORCE_UNIT_FACTORS.items():
        metrics[f"{prefix}_mae_{unit_suffix}"] = mae_ha_per_bohr * factor
        metrics[f"{prefix}_rmse_{unit_suffix}"] = rmse_ha_per_bohr * factor
    return metrics


def _to_numpy_vector(value: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    """Convert tensors, arrays, or sequences to flattened float64 NumPy vectors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _validate_gradient_vectors(predicted: np.ndarray, reference: np.ndarray, n_atoms: int) -> None:
    """Validate that predicted and reference gradients are finite 3N vectors."""
    if predicted.shape != reference.shape:
        raise ValueError(
            "Predicted and reference gradients must have the same shape. "
            f"Got predicted={predicted.shape}, reference={reference.shape}."
        )
    expected_size = 3 * int(n_atoms)
    if predicted.size != expected_size:
        raise ValueError(
            f"Gradient vector must have shape (3 * n_atoms,), i.e. ({expected_size},). "
            f"Got {predicted.shape}."
        )
    if not np.isfinite(predicted).all():
        raise ValueError("Predicted gradient contains NaN or Inf values.")
    if not np.isfinite(reference).all():
        raise ValueError("Reference gradient contains NaN or Inf values.")


def evaluate_gradient_sample(
    sample: Any,
    label_path: Path | str,
    total_gradient_fn: Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray],
    model_gradient_fn: Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray],
    sample_generator: Any,
    nuclear_gradients_dirname: str = "nuclear_gradients",
    reference_total_key: str = "nuc_grad_tot",
    reference_model_key: str = "nuc_grad_kin_plus_xc",
    report_forces: bool = True,
) -> dict[str, Any]:
    """Evaluate one sample and return a flat metrics row."""
    label_path = Path(label_path)
    reference_path = reference_path_for_label_path(label_path, nuclear_gradients_dirname)
    ref_total, ref_model = load_reference_gradients(
        reference_path,
        total_key=reference_total_key,
        model_key=reference_model_key,
    )
    # if sample is properly loaded from a label file it is guaranteed to have this attribute
    coeffs = sample.ground_state_coeffs
    gradient_sample = _gradient_sample(sample, sample_generator)

    pred_total = _to_numpy_vector(total_gradient_fn(gradient_sample, coeffs))
    pred_model = _to_numpy_vector(model_gradient_fn(gradient_sample, coeffs))

    n_atoms = _sample_n_atoms(sample)
    row = _base_result_row(
        sample,
        label_path,
        reference_path,
        reference_total_key,
        reference_model_key,
        report_forces,
    )
    row.update(compute_force_error_statistics(pred_total, ref_total, n_atoms, "total"))
    row.update(compute_force_error_statistics(pred_model, ref_model, n_atoms, "model"))
    return row


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _gradient_sample(sample: Any, sample_generator: Any) -> Any:
    """Regenerate the gradient-evaluation sample from the label geometry."""
    mol = build_molecule_np(
        _to_numpy_array(sample.atomic_numbers),
        _to_numpy_array(sample.pos),
    )
    return sample_generator.get_sample_from_mol(mol)


def _base_result_row(
    sample: Any,
    label_path: Path,
    reference_path: Path,
    reference_total_key: str,
    reference_model_key: str,
    report_forces: bool,
) -> dict[str, Any]:
    """Create the common metadata columns for a successful per-sample row."""
    return {
        "mol_id": getattr(sample, "mol_id", label_path.stem.split(".")[0]),
        "sum_formula": _sample_formula(sample),
        "n_atom": _sample_n_atoms(sample),
        "n_heavy_atom": _sample_n_heavy_atoms(sample),
        "label_path": str(label_path),
        "reference_path": str(reference_path),
        "reference_total_key": reference_total_key,
        "reference_model_key": reference_model_key,
        "quantity": "force" if report_forces else "nuclear_gradient",
        "skipped": False,
        "skip_reason": "",
    }


def _sample_formula(sample: Any) -> str:
    """Return the molecular sum formula if the sample contains a PySCF molecule."""
    if not hasattr(sample, "mol"):
        return ""
    try:
        return mole_to_sum_formula(sample.mol, True)
    except Exception:
        return ""


def _sample_n_atoms(sample: Any) -> int:
    """Return the number of atoms from either sample metadata or its molecule."""
    return int(sample.n_atom) if hasattr(sample, "n_atom") else int(sample.mol.natm)


def _sample_n_heavy_atoms(sample: Any) -> int | None:
    """Return the number of non-hydrogen atoms when atomic numbers are available."""
    if not hasattr(sample, "atomic_numbers"):
        return None
    return int(np.sum(_to_numpy_vector(sample.atomic_numbers) > 1))


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]], metrics: bool = True
) -> dict[str, Any]:
    """Summarize successful rows with full-dataset MAE/RMSE in all reported units."""
    successful_rows = [row for row in rows if not bool(row.get("skipped", False))]
    summary: dict[str, Any] = {
        "n_total": len(rows),
        "n_success": len(successful_rows),
        "n_skipped": len(rows) - len(successful_rows),
        "units": dict(UNIT_LABEL_BY_SUFFIX),
    }
    if metrics:
        summary.update(_weighted_dataset_metrics(successful_rows))
    summary.update(_diagnostic_column_statistics(successful_rows))
    return summary


def _diagnostic_column_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize numeric OFDFT diagnostic columns when present."""
    statistics: dict[str, float] = {}
    for key in OFDFT_DIAGNOSTIC_SUMMARY_KEYS:
        values = [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None and np.isfinite(float(row[key]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        statistics[f"{key}_mean"] = float(np.mean(array))
        statistics[f"{key}_min"] = float(np.min(array))
        statistics[f"{key}_max"] = float(np.max(array))
    return statistics


def _weighted_dataset_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate per-sample unit-converted MAE/RMSE into component-weighted metrics."""
    metrics: dict[str, float] = {}
    for prefix in METRIC_PREFIXES:
        n_key = f"{prefix}_n_components"
        usable_rows = [row for row in rows if n_key in row]
        if not usable_rows:
            continue

        weights = np.asarray([float(row[n_key]) for row in usable_rows], dtype=np.float64)
        total_weight = float(np.sum(weights))
        if total_weight == 0.0:
            continue

        for unit_suffix in FORCE_UNIT_FACTORS:
            mae_key = f"{prefix}_mae_{unit_suffix}"
            rmse_key = f"{prefix}_rmse_{unit_suffix}"
            mae = np.asarray([float(row[mae_key]) for row in usable_rows], dtype=np.float64)
            rmse = np.asarray([float(row[rmse_key]) for row in usable_rows], dtype=np.float64)
            metrics[mae_key] = float(np.sum(mae * weights) / total_weight)
            metrics[rmse_key] = float(np.sqrt(np.sum((rmse**2) * weights) / total_weight))
    return metrics


def write_metrics_csv(csv_path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write per-molecule rows to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("")
        return

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return stable CSV fieldnames with metadata columns first."""
    fieldnames = [field for field in CSV_BASE_FIELDS if any(field in row for row in rows)]
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def write_summary_yaml(summary_path: Path | str, summary: Mapping[str, Any]) -> None:
    """Write summary metrics to yaml."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(dict(summary)), f=summary_path, resolve=True)


def _write_evaluation_outputs(
    save_dir: Path,
    csv_name: str,
    summary_name: str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write per-molecule CSV and summary YAML outputs."""
    write_metrics_csv(save_dir / csv_name, rows)
    write_summary_yaml(save_dir / summary_name, summary)


def _configure_logger(log_file: Path | str | None) -> None:
    """Configure loguru for tqdm-friendly console output and optional file logging."""
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(lambda msg: tqdm.write(msg, end=""), format=logger_format, colorize=True)
    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, level="TRACE", rotation="10 MB", backtrace=True, diagnose=True)
    logger.info(f'Logging to "{log_file}"')


def _resolve_split_file_path(
    split_file_path: Path | str | None,
    model_config: Any,
    resolved_model_config: Any,
) -> Path:
    """Resolve the split pickle path from CLI override or training config defaults."""
    if split_file_path is None:
        split_file_path = _default_split_file_path(model_config, resolved_model_config)

    split_file_path = Path(str(split_file_path))
    return (
        split_file_path
        if split_file_path.is_absolute()
        else get_mldft_data_path() / split_file_path
    )


def _default_split_file_path(model_config: Any, resolved_model_config: Any) -> Any:
    """Return the split file path stored in the training configuration."""
    split_file = model_config.data.datamodule.split_file
    if hasattr(split_file, "output_file"):
        return resolved_model_config.data.datamodule.split_file.output_file
    return split_file


def _instantiate_basis_info(model_config: Any) -> Any:
    """Instantiate the basis information object from the model configuration."""
    return instantiate(model_config.data.basis_info)


def _prepare_dataset_kwargs(model_config: Any, split_file_path: Path) -> dict[str, Any]:
    """Build dataset keyword arguments for loading ground-state label samples."""
    dataset_kwargs = instantiate(model_config.data.datamodule.dataset_kwargs)
    dataset_kwargs.pop("hdf5_manifest_path", None)
    dataset_kwargs.pop("denop_hdf5_manifest_path", None)
    dataset_kwargs["hdf5_manifest_path"] = str(_denop_manifest_path(model_config, split_file_path))
    dataset_kwargs["limit_scf_iterations"] = -1
    dataset_kwargs["additional_keys_at_ground_state"] = _ground_state_energy_representations()
    for key in DATASET_KWARGS_TO_DROP:
        dataset_kwargs.pop(key, None)
    return dataset_kwargs


def _denop_manifest_path(model_config: Any, split_file_path: Path) -> Path:
    """Return the expected density-optimization HDF5 manifest path."""
    return split_file_path.parent / f"labels_denop_h5_{model_config.data.target_key}/manifest.json"


def _ground_state_energy_representations() -> dict[str, Representation]:
    """Return extra ground-state energy keys requested from the dataset."""
    return {f"of_labels/energies/{key}": Representation.SCALAR for key in GROUND_STATE_ENERGY_KEYS}


def _select_paths(
    split_file_path: Path,
    split: str,
    n_molecules: int,
    molecule_choice: str | list[int] | ListConfig,
    molecule_ids: Sequence[int] | None,
    seed: int | None,
) -> tuple[list[Path], list[int]]:
    """Select label paths and SCF-iteration indices from a split pickle."""
    with split_file_path.open("rb") as handle:
        split_dict = pickle.load(handle)

    if molecule_ids:
        return _select_paths_by_molecule_id(split_dict, split, molecule_ids)
    return _select_paths_by_split(split_dict, split, n_molecules, molecule_choice, seed)


def _select_paths_by_molecule_id(
    split_dict: Mapping[str, Any],
    split: str,
    molecule_ids: Sequence[int],
) -> tuple[list[Path], list[int]]:
    """Select paths for explicit molecule IDs, searching all standard splits."""
    paths: list[Path] = []
    iterations: list[int] = []
    data_dir = get_mldft_data_path()
    for molecule_id in molecule_ids:
        result = find_molecule_path_by_id(int(molecule_id), split_dict, data_dir, split)
        if result is None:
            raise ValueError(f"Molecule {molecule_id} not found in train/val/test splits.")
        paths.append(result[0])
        iterations.append(result[1])
    return paths, iterations


def _select_paths_by_split(
    split_dict: Mapping[str, Any],
    split: str,
    n_molecules: int,
    molecule_choice: str | list[int] | ListConfig,
    seed: int | None,
) -> tuple[list[Path], list[int]]:
    """Select paths by split, molecule-choice strategy, and optional seed."""
    split_paths, split_iterations = get_paths_from_split(split_dict, split, get_mldft_data_path())
    indices = configure_dataset_indices(len(split_paths), int(n_molecules), molecule_choice, seed)
    return [split_paths[int(index)] for index in indices], [
        split_iterations[int(index)] for index in indices
    ]


def _build_dataset(
    paths: Sequence[Path],
    num_scf_iterations_per_path: Sequence[int],
    basis_info: Any,
    dataset_kwargs: Mapping[str, Any],
    transforms: Any | None = None,
) -> Any:
    """Create either an HDF5-shard dataset or plain zarr-backed OF dataset."""
    dataset_kwargs = dict(dataset_kwargs)
    hdf5_manifest_path = dataset_kwargs.pop("hdf5_manifest_path", None)
    if hdf5_manifest_path is not None and Path(str(hdf5_manifest_path)).exists():
        source_keys = [path.relative_to(get_mldft_data_path()).as_posix() for path in paths]
        return HDF5ShardOFDataset(
            source_keys=source_keys,
            num_scf_iterations_per_path=num_scf_iterations_per_path,
            basis_info=basis_info,
            transforms=transforms,
            hdf5_manifest_path=Path(str(hdf5_manifest_path)),
            **dataset_kwargs,
        )
    return OFDataset(
        paths=paths,
        num_scf_iterations_per_path=num_scf_iterations_per_path,
        basis_info=basis_info,
        transforms=transforms,
        **dataset_kwargs,
    )


def _prepare_evaluation_dataset(
    model_config: Any,
    split_file_path: Path,
    split: str,
    n_molecules: int,
    molecule_choice: str | list[int] | ListConfig,
    molecule_ids: Sequence[int] | None,
    seed: int | None,
    transforms_factory: Callable[[Any], Any] | None = None,
) -> tuple[Any, list[Path]]:
    """Build the selected evaluation dataset and return it with label paths."""
    basis_info = _instantiate_basis_info(model_config)
    dataset_kwargs = _prepare_dataset_kwargs(model_config, split_file_path)
    paths, scf_iterations = _select_paths(
        split_file_path,
        split,
        n_molecules,
        molecule_choice,
        molecule_ids,
        seed,
    )
    transforms = transforms_factory(basis_info) if transforms_factory is not None else None
    dataset = _build_dataset(paths, scf_iterations, basis_info, dataset_kwargs, transforms)
    return dataset, paths


def _iter_samples_with_paths(
    dataset: Any,
    paths: Sequence[Path],
    torch_threads_per_worker: int,
    device: str,
):
    """Yield single samples together with their source label paths."""
    torch.set_num_threads(resolve_torch_threads_per_worker(torch_threads_per_worker))
    dataloader = OFLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        list_keys=[],
    )
    for label_path, batch in zip_longest(paths, dataloader, fillvalue=None):
        if label_path is None:
            raise ValueError("Dataloader yielded more samples than there are label paths.")
        if batch is None:
            raise ValueError("Dataloader yielded fewer samples than there are label paths.")
        sample = batch.to_data_list()[0] if hasattr(batch, "to_data_list") else batch
        yield Path(label_path), sample.to(device) if hasattr(sample, "to") else sample


@dataclass(slots=True)
class ModelGradientState:
    """Reusable model state for force-gradient evaluation tasks."""

    lightning_module: Any
    sample_generator: SampleGenerator
    total_factory: FunctionalFactory
    model_factory: FunctionalFactory


def _make_model_gradient_state(
    ckpt_path: Path | str,
    model_config: Any,
    device: torch.device,
    model_dtype: torch.dtype,
    deterministic: bool,
    batch_size: int,
    batch_timeout: float,
    xc_functional: str,
    negative_integrated_density_penalty_weight: float,
) -> ModelGradientState:
    """Instantiate the checkpoint, sample generator, and energy factories."""
    lightning_module = instantiate_model(
        Path(ckpt_path),
        str(device),
        model_dtype,
        deterministic,
        batch_size,
        batch_timeout,
    )
    sample_generator = SampleGenerator(model_config=model_config, model=lightning_module)
    total_factory = FunctionalFactory.from_module(
        lightning_module,
        xc_functional,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
    )
    return ModelGradientState(
        lightning_module=lightning_module,
        sample_generator=sample_generator,
        total_factory=total_factory,
        model_factory=FunctionalFactory(lightning_module),
    )


def _make_gradient_functions(
    ckpt_path: Path | str,
    model_config: Any,
    device: torch.device,
    model_dtype: torch.dtype,
    deterministic: bool,
    batch_size: int,
    batch_timeout: float,
    xc_functional: str,
    negative_integrated_density_penalty_weight: float,
    gradient_method: str,
) -> tuple[Any, Callable, Callable]:
    """Instantiate the checkpoint and construct total/model-only gradient callables."""
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
    total_gradient_fn = get_gradient_method(
        gradient_method,
        state.sample_generator,
        state.total_factory,
        device=str(device),
    )
    if (
        gradient_method.lower() == "analytical_generator"
        or gradient_method.lower() == "analytical"
        or gradient_method.lower() == "analytic"
    ):
        # numerical gradients either have them included anyway because of the physical terms present
        # or not, they can not be activated or deactivated
        kwargs = {
            "include_hellmann": False,
        }
    else:
        kwargs = {}
    model_gradient_fn = get_gradient_method(
        gradient_method, state.sample_generator, state.model_factory, device=str(device), **kwargs
    )
    return state.sample_generator, total_gradient_fn, model_gradient_fn


def _failure_row(
    label_path: Path,
    error: Exception,
    reference_total_key: str,
    reference_model_key: str,
    nuclear_gradients_dirname: str,
    report_forces: bool,
) -> dict[str, Any]:
    """Return a skipped-row record for a failed molecule evaluation."""
    row = {
        "mol_id": label_path.stem.split(".")[0],
        "sum_formula": "",
        "n_atom": None,
        "n_heavy_atom": None,
        "quantity": "force" if report_forces else "nuclear_gradient",
        "skipped": True,
        "skip_reason": str(error),
        "exception_type": type(error).__name__,
        "reference_total_key": reference_total_key,
        "reference_model_key": reference_model_key,
        "label_path": str(label_path),
        "reference_path": str(
            reference_path_for_label_path(label_path, nuclear_gradients_dirname)
        ),
    }
    row.update(_failure_diagnostic_metadata(error))
    return row


def _failure_diagnostic_metadata(error: Exception) -> dict[str, Any]:
    """Return diagnostic columns carried by recoverable evaluation errors."""
    metric_dict = getattr(error, "metric_dict", None)
    if not isinstance(metric_dict, Mapping):
        return {}
    return _ofdft_diagnostic_metadata(metric_dict)


def _ofdft_diagnostic_metadata(metric_dict: Mapping[str, Any]) -> dict[str, Any]:
    stopping_index = int(metric_dict.get("stopping_index", -1))
    metadata: dict[str, Any] = {
        "ofdft_stopping_index": stopping_index,
        "ofdft_iterations": stopping_index + 1,
        "ofdft_converged": bool(metric_dict.get("converged", False)),
        "ofdft_energy_error_mha": None,
        "ofdft_abs_energy_error_mha": None,
        "ofdft_density_l2_norm": None,
        "ofdft_gradient_norm": None,
    }
    for source_key, target_key, factor in (
        ("gs_signed_energy_error", "ofdft_energy_error_mha", 1000.0),
        ("gs_abs_energy_error", "ofdft_abs_energy_error_mha", 1000.0),
        ("gs_l2_norm", "ofdft_density_l2_norm", 1.0),
        ("gs_gradient_norm", "ofdft_gradient_norm", 1.0),
    ):
        if source_key in metric_dict:
            metadata[target_key] = float(metric_dict[source_key]) * factor
    return metadata


@set_default_torch_dtype(OFDFT_DTYPE)
def run_gradient_evaluation(
    ckpt_path: Path | str,
    model_config: DictConfig,
    split_file_path: Path | str,
    split: str = "val",
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    batch_timeout: float = 0.05,
    model_dtype: torch.dtype = OFDFT_DTYPE,
    deterministic: bool = True,
    xc_functional: str = "PBE",
    negative_integrated_density_penalty_weight: float = 0.0,
    n_molecules: int = 1,
    molecule_choice: str | list[int] | ListConfig = "first_n",
    molecule_ids: Sequence[int] | None = None,
    seed: int | None = 0,
    nuclear_gradients_dirname: str = "nuclear_gradients",
    reference_total_key: str = "nuc_grad_tot",
    reference_model_key: str | None = "auto",
    gradient_method: str = "analytical_generator",
    report_forces: bool = True,
    save_dir: Path | str | None = None,
    fail_fast: bool = False,
    torch_threads_per_worker: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the analytical nuclear-gradient evaluation over a split selection."""
    device = torch.device(device)
    split_file_path = Path(split_file_path)
    save_dir = Path(save_dir or Path.cwd())
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset, paths = _prepare_evaluation_dataset(
        model_config,
        split_file_path,
        split,
        n_molecules,
        molecule_choice,
        molecule_ids,
        seed,
    )
    sample_generator, total_gradient_fn, model_gradient_fn = _make_gradient_functions(
        ckpt_path,
        model_config,
        device,
        model_dtype,
        deterministic,
        batch_size,
        batch_timeout,
        xc_functional,
        negative_integrated_density_penalty_weight,
        gradient_method,
    )
    resolved_reference_model_key = resolve_reference_model_key(model_config, reference_model_key)

    rows = _evaluate_dataset_rows(
        dataset,
        paths,
        sample_generator,
        total_gradient_fn,
        model_gradient_fn,
        nuclear_gradients_dirname,
        reference_total_key,
        resolved_reference_model_key,
        report_forces,
        fail_fast,
        torch_threads_per_worker,
        str(device),
    )
    _raise_if_no_successful_rows(rows)
    summary = summarize_metric_rows(rows)
    _write_evaluation_outputs(save_dir, CSV_PER_MOLECULE, SUMMARY_YAML, rows, summary)
    logger.success(f"Gradient evaluation saved to {save_dir}")
    return rows, summary


def _evaluate_sample_rows(
    dataset: Any,
    paths: Sequence[Path],
    desc: str,
    sample_evaluator: Callable[[Path, Any], Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    failure_row_factory: Callable[[Path, Exception], Mapping[str, Any]],
    log_success: Callable[[list[Mapping[str, Any]]], None],
    fail_fast: bool,
    torch_threads_per_worker: int,
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate samples with shared recoverable-error handling."""
    rows: list[dict[str, Any]] = []
    iterator = _iter_samples_with_paths(dataset, paths, torch_threads_per_worker, device)
    for label_path, sample in tqdm(iterator, total=len(paths), desc=desc):
        try:
            result = sample_evaluator(label_path, sample)
            sample_rows = [result] if isinstance(result, Mapping) else list(result)
            rows.extend(dict(row) for row in sample_rows)
            log_success(sample_rows)
        except RECOVERABLE_SAMPLE_ERRORS as error:
            if fail_fast:
                raise
            logger.warning(f"Skipping {label_path} due to recoverable error: {error}")
            rows.append(dict(failure_row_factory(label_path, error)))
    return rows


def _evaluate_dataset_rows(
    dataset: Any,
    paths: Sequence[Path],
    sample_generator: Any,
    total_gradient_fn: Callable,
    model_gradient_fn: Callable,
    nuclear_gradients_dirname: str,
    reference_total_key: str,
    reference_model_key: str,
    report_forces: bool,
    fail_fast: bool,
    torch_threads_per_worker: int,
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate analytical-gradient rows using shared row handling."""

    def evaluate(label_path: Path, sample: Any) -> Mapping[str, Any]:
        return evaluate_gradient_sample(
            sample=sample,
            sample_generator=sample_generator,
            label_path=label_path,
            total_gradient_fn=total_gradient_fn,
            model_gradient_fn=model_gradient_fn,
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
            f"total_MAE={row['total_mae_ha_per_bohr']:.4e} Ha/Bohr "
            f"model_MAE={row['model_mae_ha_per_bohr']:.4e} Ha/Bohr"
        )

    return _evaluate_sample_rows(
        dataset,
        paths,
        "gradient eval",
        evaluate,
        failure_row,
        log_success,
        fail_fast,
        torch_threads_per_worker,
        device,
    )


def _raise_if_no_successful_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail a run that produced no successful per-sample evaluations."""
    if not any(not bool(row.get("skipped", False)) for row in rows):
        raise RuntimeError("No successful force evaluations were produced.")


def _cfg_get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Read a top-level Hydra config value with a default."""
    return cfg.get(key, default)


def _denop_cfg_get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Read a value from the top level, falling back to denop_settings."""
    if cfg.get(key) is not None:
        return cfg.get(key)
    return cfg.get("denop_settings", {}).get(key, default)


def _save_resolved_cli_config(cfg: DictConfig, config_file: Any, log_file: Path | None) -> None:
    """Persist the resolved CLI config to configured output locations."""
    if config_file is not None:
        OmegaConf.save(cfg, Path(str(config_file)), resolve=True)
    if log_file is not None:
        OmegaConf.save(cfg, log_file.parent / "hparams_resolved.yaml", resolve=True)


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/ofdft/",
    config_name="analytical_forces.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
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

    with open_dict(cfg):
        cfg.run_path = str(run_path)
        cfg.ckpt_path = str(ckpt_path)
        cfg.split_file_path = str(split_file_path)
    _save_resolved_cli_config(cfg, cfg.get("config_file"), log_file)

    run_gradient_evaluation(
        ckpt_path=ckpt_path,
        model_config=model_config,
        split_file_path=split_file_path,
        split=_cfg_get(cfg, "split", "val"),
        device=_cfg_get(cfg, "device", "cpu"),
        batch_size=_cfg_get(cfg, "batch_size", 1),
        batch_timeout=_cfg_get(cfg, "batch_timeout", 0.05),
        model_dtype=str_to_torch_float_dtype(_cfg_get(cfg, "model_dtype", OFDFT_DTYPE)),
        deterministic=_cfg_get(cfg, "deterministic", True),
        xc_functional=_denop_cfg_get(cfg, "xc_functional", "PBE"),
        negative_integrated_density_penalty_weight=_denop_cfg_get(
            cfg,
            "negative_integrated_density_penalty_weight",
            0.0,
        ),
        n_molecules=_cfg_get(cfg, "n_molecules", 1),
        molecule_choice=_cfg_get(cfg, "molecule_choice", "first_n"),
        molecule_ids=_cfg_get(cfg, "molecule_ids", []),
        seed=_cfg_get(cfg, "seed", 0),
        nuclear_gradients_dirname=_cfg_get(cfg, "nuclear_gradients_dirname", "nuclear_gradients"),
        reference_total_key=_cfg_get(cfg, "reference_total_key", "nuc_grad_tot"),
        reference_model_key=_cfg_get(cfg, "reference_model_key", "auto"),
        gradient_method=_cfg_get(cfg, "gradient_method", "analytical_generator"),
        report_forces=_cfg_get(cfg, "report_forces", True),
        save_dir=log_file.parent if log_file is not None else Path.cwd(),
        fail_fast=_cfg_get(cfg, "fail_fast", False),
        torch_threads_per_worker=_cfg_get(cfg, "torch_threads_per_worker", 1),
    )


if __name__ == "__main__":
    main()
