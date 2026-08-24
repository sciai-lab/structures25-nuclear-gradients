import os

# Limit BLAS/OpenMP thread pools unless the user already configured them.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import time
from pathlib import Path
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from geometric.nifty import logger as geom_logger
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from pyscf import gto
from tqdm import tqdm

from mldft.ml.data.components.basis_transforms import ApplyBasisTransformation
from mldft.ml.data.components.of_data import Representation
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.density_optimization import OFDFT_KWARGS, density_optimization
from mldft.ofdft.energies import Energies
from mldft.ofdft.geometry_energy_offset import (
    set_geometry_energy_offset_defaults,
    set_geometry_energy_offset_result,
)
from mldft.ofdft.geometry_optimization import (
    GeometryOptimizationNotConvergedError,
    GeomOpt,
)
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.utils import extras
from mldft.utils.conversions import pyscf_to_rdkit
from mldft.utils.environ import CheckpointChoice
from mldft.utils.omegaconf_resolvers import to_no_basis_transforms_dataset_statistics
from mldft.utils.plotting.plot_geometry_optimization import (
    plot_single_geometry_optimization,
    plot_summary_geometry_optimization,
)
from mldft.utils.pyscf_calculations import load_scf_with_optional_data
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sad_guesser import SADNormalizationMode
from mldft.utils.sample_generator import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype


class GeometryCallback:
    """Callback to save energies and trajectories during geometry optimization."""

    def __init__(self):
        """Callback to save energies and trajectories during geometry optimization."""
        self.energies = []
        self.ground_state_coeffs_list = []
        self.nuclear_gradient_norms = []
        self.denop_times = []
        self.gradient_times = []
        self.geometric_times = []
        self.coords = []

    def callback(self, local_vars: dict):
        """Callback function to save energies and trajectories during geometry optimization."""
        self.energies.append(local_vars["energy"])
        self.ground_state_coeffs_list.append(local_vars["last_coeffs"].cpu().numpy())
        self.nuclear_gradient_norms.append(local_vars["nuclear_gradient_norm"])
        self.denop_times.append(local_vars["denop_time"])
        self.gradient_times.append(local_vars["gradient_time"])
        self.geometric_times.append(local_vars["geometric_time"])
        self.coords.append(local_vars["coords"])


def _to_total_energy(value: float | Energies) -> float:
    """Convert a scalar or Energies object to a total energy in Hartree."""
    if isinstance(value, Energies):
        return float(value.total_energy)
    return float(value)


def _molecule_id_from_geometry_file(file: Path) -> int | None:
    """Extract the dataset molecule ID encoded in a geometry filename."""
    stem = file.stem
    if stem.isdigit():
        return int(stem)

    _, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return int(suffix)

    return None


def _select_geometry_molecule_files(
    files: list[Path],
    molecule_ids: list[int] | None,
    start_idx: int,
    n_molecules: int | None,
) -> list[tuple[int, Path]]:
    """Select geometry files by dataset molecule ID or contiguous position."""
    if molecule_ids is not None and len(molecule_ids) > 0:
        if any(molecule_id < 0 for molecule_id in molecule_ids):
            raise ValueError("molecule_ids must be non-negative.")

        files_by_molecule_id = {}
        duplicated_ids = set()
        for file in files:
            molecule_id = _molecule_id_from_geometry_file(file)
            if molecule_id is None:
                continue
            if molecule_id in files_by_molecule_id:
                duplicated_ids.add(molecule_id)
            files_by_molecule_id[molecule_id] = file

        if duplicated_ids:
            raise ValueError(
                f"Ambiguous molecule ids in geometry xyz filenames: {sorted(duplicated_ids)}."
            )

        missing_ids = [
            molecule_id for molecule_id in molecule_ids if molecule_id not in files_by_molecule_id
        ]
        if missing_ids:
            available_ids = sorted(files_by_molecule_id)
            raise ValueError(
                f"Molecule ids {missing_ids} not found in geometry xyz filenames. "
                f"Available molecule ids include {available_ids[:10]}."
            )

        return [(molecule_id, files_by_molecule_id[molecule_id]) for molecule_id in molecule_ids]

    if start_idx < 0:
        raise ValueError(f"start_idx must be non-negative, but got {start_idx}.")
    if start_idx >= len(files):
        raise ValueError(
            f"start_idx ({start_idx}) must be smaller than the number of molecules "
            f"({len(files)})."
        )
    if n_molecules is not None and n_molecules < 1:
        raise ValueError(f"n_molecules must be at least 1, but got {n_molecules}.")

    if n_molecules is None:
        end_idx = len(files)
    else:
        end_idx = min(start_idx + n_molecules, len(files))
        if start_idx + n_molecules > len(files):
            logger.info(
                f"Requested {n_molecules} molecules from start_idx={start_idx}, but only "
                f"{len(files) - start_idx} are available."
            )

    return [
        (molecule_id, file)
        for molecule_id, file in enumerate(files[start_idx:end_idx], start=start_idx)
    ]


def compute_ml_single_point_energy(
    mol: gto.Mole,
    sample_generator: SampleGenerator,
    functional_factory,
    optimizer: Optimizer,
    ofdft_kwargs: dict,
) -> tuple[float, bool, float]:
    """Compute one ML OFDFT single-point energy for the provided geometry.

    Args:
        mol: Geometry at which to evaluate the ML OFDFT functional.
        sample_generator: Sample generator for model-aligned sample creation.
        functional_factory: Functional factory used to construct the OFDFT functional.
        optimizer: Density optimizer.
        ofdft_kwargs: Keyword arguments for :func:`density_optimization`.

    Returns:
        Tuple ``(energy_ha, converged, runtime_s)``.
    """
    sample = sample_generator.get_sample_from_mol(mol)
    start_time = time.time()
    energies, _, converged, _ = density_optimization(
        sample,
        sample.mol,
        optimizer,
        functional_factory,
        disable_printing=True,
        **ofdft_kwargs,
    )
    runtime_s = time.time() - start_time
    return _to_total_energy(energies), bool(converged), runtime_s


def geometry_opt_worker(
    run_path: Path | str,
    negative_integrated_density_penalty_weight: float,
    ckpt_choice: CheckpointChoice,
    ofdft_kwargs: dict,
    optimizer: Optimizer,
    gradients: str,
    maxsteps: int,
    initial_guess_from_last: Literal["simple", "safe"] | None,
    geometric_config: dict,
    save_dir: Path,
    ks_path: Path,
    geometric_loglevel: int,
    device: str | torch.device,
    transform_device: str | torch.device,
    main_log_level: int,
    molecule_log_level: int,
    num_threads: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    nuclear_gradient_device: str | torch.device,
    compute_geometry_energy_offset: bool = False,
    raise_error: bool = False,
) -> None:
    """Worker process for geometry optimization.

    Args:
        run_path: Path to the run directory.
        negative_integrated_density_penalty_weight: Weight for the negative integrated density penalty.
        ckpt_choice: Checkpoint choice for the model.
        ofdft_kwargs: Additional arguments for OFDFT.
        optimizer: Optimizer for OFDFT calculation.
        gradients: Type of gradients to use.
        maxsteps: Maximum number of steps for optimization.
        initial_guess_from_last: How to initialize the density from the last ground state.
        geometric_config: Additional configuration for geometry optimization.
        save_dir: Directory to save the results.
        ks_path: Path to the KS results for comparison.
        geometric_loglevel: Logging level for geometric optimizer.
        device: Device to use for the worker.
        transform_device: Device for transformation.
        num_threads: Number of threads to use for the worker.
        task_queue: Queue for tasks to process.
        result_queue: Queue for results to return.
        nuclear_gradient_device: Device to use for nuclear gradient calculations.
        compute_geometry_energy_offset: Whether to evaluate the ML functional at the
            KS-optimized geometry to compute ``E_ML(A) - E_ML(B)``.
        raise_error: Whether to raise errors or catch and log them.
    """

    # prepare worker and device
    geom_logger.setLevel(geometric_loglevel)

    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
        level=main_log_level,
    )
    logger.trace(f"Worker on {device} started.")
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(device)

    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(num_threads)
    if isinstance(device, torch.device):
        device_number = device.index if device.index is not None else 0
    elif isinstance(device, str) and device.startswith("cuda"):
        device_number = int(device.split(":")[1])
    else:
        device_number = 0  # default to 0 for CPU or unexpected values
    os.environ["GEOMOPT_PID"] = str(device_number)

    # load model and statistics
    sample_generator = SampleGenerator.from_run_path(
        run_path,
        device=device,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        ckpt_choice=ckpt_choice,
        transform_device=transform_device,
    )
    functional_factory = sample_generator.get_functional_factory()

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

    # run actual calculations

    while True:
        task = task_queue.get()
        if task is None:
            logger.trace("Termination signal received")
            break  # Exit signal

        task_idx, molecule_id, file = task

        try:
            result = geometry_opt_calculation(
                file,
                molecule_id,
                sample_generator,
                functional_factory,
                optimizer,
                gradients,
                maxsteps,
                initial_guess_from_last,
                ofdft_kwargs,
                geometric_config,
                save_dir,
                ks_path,
                molecule_log_level,
                nuclear_gradient_device,
                compute_geometry_energy_offset,
            )
            result_queue.put((task_idx, result))
        except Exception as e:
            if raise_error:
                raise e
            else:
                logger.opt(exception=True).debug(f"{device} encountered Error")
                result_queue.put((task_idx, None))


def geometry_opt_calculation(
    file: Path,
    idx: int,
    sample_generator: SampleGenerator,
    functional_factory,
    optimizer: Optimizer,
    gradients: str,
    maxsteps: int,
    initial_guess_from_last: str | None,
    ofdft_kwargs: dict,
    geometric_config: dict,
    save_dir: Path,
    ks_path: Path,
    molecule_log_level: int,
    nuclear_gradient_device: str | torch.device = "auto",
    compute_geometry_energy_offset: bool = False,
):
    """Perform geometry optimization for a single molecule.

    Args:
        file: Path to the molecule file.
        idx: Index of the molecule.
        sample_generator: Model configuration.
        functional_factory: Functional factory for OFDFT.
        optimizer: Optimizer for OFDFT calculation.
        gradients: Type of gradients to use.
        maxsteps: Maximum number of steps for optimization.
        initial_guess_from_last: How to initialize the density from the last ground state.
        ofdft_kwargs: Additional arguments for OFDFT.
        geometric_config: Additional configuration for geometry optimization.
        save_dir: Directory to save the results.
        ks_path: Path to the KS results for comparison.
        molecule_log_level: Logging level for molecule-specific logs.
        nuclear_gradient_device: Device to use for nuclear gradient calculations.
        compute_geometry_energy_offset: Whether to evaluate the ML functional at the
            KS-optimized geometry and store ``E_ML(A) - E_ML(B)``.

    Returns:
        features: Features extracted from the geometry optimization also comparing to KS.
    """

    basename = file.stem

    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    molecule_sink = logger.add(
        save_dir / "geometry_opt_log" / f"{basename}.log",
        level=molecule_log_level,
        format=logger_format,
    )
    mol = gto.M(atom=str(file))
    sample = sample_generator.get_sample_from_mol(mol)
    device = sample.coeffs.device

    geometry_callback = GeometryCallback()
    geom_opt = GeomOpt(
        sample,
        functional_factory,
        sample_generator,
        optimizer=optimizer,
        gradients=gradients,
        maxsteps=maxsteps,
        geometry_opt_callback=geometry_callback.callback,
        initial_guess_from_last=initial_guess_from_last,
        ofdft_kwargs=ofdft_kwargs,
        nuclear_gradient_device=nuclear_gradient_device,
    )

    logger.info(f"{device}: Optimizing molecule {idx+1}: {mole_to_sum_formula(mol, True)}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # ensure correct timing
    start_time = time.time()
    try:
        opt_mol, sample = geom_opt.run(**geometric_config)
        sample.add_item("geomopt_converged", True, representation=Representation.NONE)
        logger.success(f"Optimization converged for {mole_to_sum_formula(mol, True)}.")
    except GeometryOptimizationNotConvergedError:
        sample = geom_opt.sample
        sample.add_item("geomopt_converged", False, representation=Representation.NONE)
        logger.warning(f"Optimization did not converge for {mole_to_sum_formula(mol, True)}.")
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # ensure correct timing
    end_time = time.time()

    cycle = geom_opt.cycle

    logger.trace(f"Final Geometry:\n{sample.mol.tostring('xyz')}")

    # Delete large matrices
    sample.delete_item("overlap_matrix")
    sample.delete_item("coulomb_matrix")
    sample.delete_item("nuclear_attraction_vector")
    if hasattr(sample, "ao"):
        sample.delete_item("ao")
    # Transform back to untransformed basis
    sample = ApplyBasisTransformation()(
        sample,
        sample.transformation_matrix,
        sample.inv_transformation_matrix,
        invert=True,
    )
    sample.delete_item("transformation_matrix")
    sample.delete_item("inv_transformation_matrix")
    sample = sample.to("cpu")

    sample.add_item("energy", geom_opt.e_last, representation=Representation.NONE)
    sample.add_item("geomopt_time", end_time - start_time, representation=Representation.NONE)
    sample.add_item("geomopt_cycle", cycle, representation=Representation.NONE)
    sample.add_item(
        "geomopt_energies",
        geometry_callback.energies,
        representation=Representation.NONE,
    )
    coords = np.stack(geometry_callback.coords, axis=0)
    coords = torch.as_tensor(coords, device=sample.coeffs.device, dtype=OFDFT_DTYPE)
    sample.add_item("geomopt_trajectory", coords, representation=Representation.NONE)
    ground_state_coeffs = np.stack(geometry_callback.ground_state_coeffs_list, axis=0)
    ground_state_coeffs = torch.as_tensor(
        ground_state_coeffs, device=sample.coeffs.device, dtype=torch.float64
    )
    # representation should be something like "vector_list", but none is fine for now
    sample.add_item(
        "ground_state_coeffs_trajectory", ground_state_coeffs, representation=Representation.NONE
    )
    nuclear_gradient_norms = torch.as_tensor(
        geometry_callback.nuclear_gradient_norms,
        device=sample.coeffs.device,
        dtype=torch.float64,
    )
    sample.add_item(
        "nuclear_gradient_norm_trajectory",
        nuclear_gradient_norms,
        representation=Representation.NONE,
    )
    denop_times = torch.as_tensor(
        geometry_callback.denop_times, device=sample.coeffs.device, dtype=torch.float64
    )
    sample.add_item("geomopt_denop_times", denop_times, representation=Representation.NONE)
    gradient_times = torch.as_tensor(
        geometry_callback.gradient_times,
        device=sample.coeffs.device,
        dtype=torch.float64,
    )
    sample.add_item("geomopt_gradient_times", gradient_times, representation=Representation.NONE)
    geometric_times = torch.as_tensor(
        geometry_callback.geometric_times,
        device=sample.coeffs.device,
        dtype=torch.float64,
    )
    sample.add_item("geomopt_geometric_times", geometric_times, representation=Representation.NONE)

    sample = sample.to("cpu")

    # compare to ks
    ks_mol, ks_data = load_scf_with_optional_data(
        ks_path / f"{basename}.chk",
        optional_keys=("initial_energy",),
    )

    set_geometry_energy_offset_defaults(sample)

    if compute_geometry_energy_offset and bool(sample.geomopt_converged):
        try:
            ml_energy_at_ks_geometry, sp_converged, offset_runtime = (
                compute_ml_single_point_energy(
                    mol=ks_mol,
                    sample_generator=sample_generator,
                    functional_factory=functional_factory,
                    optimizer=optimizer,
                    ofdft_kwargs=ofdft_kwargs,
                )
            )
            ml_energy_at_ml_geometry = _to_total_energy(sample.energy)
            geometry_energy_offset = ml_energy_at_ml_geometry - ml_energy_at_ks_geometry

            set_geometry_energy_offset_result(
                sample,
                ml_energy_at_ks_geometry=ml_energy_at_ks_geometry,
                ml_energy_at_ks_geometry_converged=sp_converged,
                denop_time=offset_runtime,
                geometry_energy_offset=geometry_energy_offset if sp_converged else np.nan,
            )
            if not sp_converged:
                logger.warning(
                    "ML single-point on KS geometry did not converge for "
                    f"{mole_to_sum_formula(ks_mol, True)}. "
                    "Geometry energy offset is marked unavailable."
                )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to compute ML single-point energy at KS geometry for "
                f"{mole_to_sum_formula(ks_mol, True)}. Geometry energy offset remains unavailable."
            )

    save_file = save_dir / "geometry_opt_sample" / f"{basename}.pt"
    logger.trace(f"Saving sample to {save_file}")
    torch.save(sample, save_file)

    plot_file = save_dir / "geometry_opt_plot" / f"{basename}.pdf"

    plot_result = plot_single_geometry_optimization(
        sample, ks_mol, ks_data, name=basename, return_features=True
    )
    if isinstance(plot_result, tuple):
        fig, features = plot_result
    else:
        fig = plot_result
        features = {}

    logger.trace(f"{device}: Saving plot to {plot_file}")
    fig.savefig(plot_file, bbox_inches="tight", dpi=300)
    logger.trace(f"{device}: Saving plot finished")

    plt.close(fig)

    # remove the molecule logger
    logger.remove(molecule_sink)

    ks_rdkit_mol = pyscf_to_rdkit(ks_mol, sanitize=True)

    features["ks_rdkit_mol"] = ks_rdkit_mol
    return features


@set_default_torch_dtype(torch.float64)
def run_geometry_process(
    run_path: Path | str,
    data_path: Path | str,
    log_file: Path | str,
    ckpt_choice: CheckpointChoice = "last",
    n_molecules: int | None = 1,
    molecule_ids: list[int] | None = None,
    start_idx: int = 0,
    device: str | torch.device = "cuda",
    transform_device: str | torch.device = "cpu",
    optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
    negative_integrated_density_penalty_weight: float = 0.0,
    maxsteps: int = 50,
    gradients: str = "Analytical",
    geometric_config: dict | None = None,
    geometric_loglevel: int = 30,
    initial_guess_from_last: Literal["simple", "safe"] | None = None,
    ofdft_kwargs: dict | None = None,
    main_log_level: int = 20,
    molecule_log_level: int = 25,
    num_devices: int = 1,
    num_threads_per_process: int = 8,
    nuclear_gradient_device: str | torch.device = "auto",
    compute_geometry_energy_offset: bool = False,
) -> None:
    """Run the geometry optimization and compare to saved ks results.

    Args:
        run_path: Path to the run directory.
        data_path: Path to the data directory.
        log_file: Path to the log file.
        ckpt_choice: Choice of checkpoint to load.
        n_molecules: Number of molecules to process.
        molecule_ids: If set, only these molecule IDs are processed. IDs are resolved from
            the geometry filename stems, e.g. ``0003452.xyz`` or
            ``omol-geom_03106275.xyz``.
        start_idx: Start index for processing molecules when molecule_ids is unset.
        device: Device to use.
        transform_device: Device for transformation.
        optimizer: Optimizer for ofdft calculation.
        negative_integrated_density_penalty_weight: Weight for penalty.
        maxsteps: Maximum number of steps for optimization.
        gradients: Type of gradients.
        initial_guess_from_last:: How to initialize the density from the last
            ground state. Can be "simple" or "safe".
            initialized with a new guess.
        geometric_config: Additional configuration for geometry optimization.
        geometric_loglevel: logging level for geometric optimizer.
        ofdft_kwargs: Additional arguments for OFDFT.
        num_processes_per_device: Number of processes per device.
        num_devices: Number of devices.
        num_workers: Number of workers.
        num_threads_per_process: Number of threads per process.
        nuclear_gradient_device: Device to use for nuclear gradient calculations.
        compute_geometry_energy_offset: Whether to run an extra ML single-point at
            each KS minimum geometry and store ``E_ML(A) - E_ML(B)``.
    """
    mp.set_start_method("spawn", force=True)

    if geometric_config is None:
        geometric_config = {}
    if ofdft_kwargs is None:
        ofdft_kwargs = {}

    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
        level=main_log_level,
    )
    if device == "cpu":
        assert num_devices == 1, "Only one cpu device is supported."
    elif device == "cuda":
        assert (
            num_devices <= torch.cuda.device_count()
        ), f"Configured {num_devices} cuda devices but only {torch.cuda.device_count()} are available."

    if isinstance(run_path, str):
        run_path = Path(run_path)
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if isinstance(log_file, str):
        log_file = Path(log_file)
    save_dir = log_file.parent
    save_dir.mkdir(exist_ok=True)  # for saving the denop trajectories of the samples
    (save_dir / "geometry_opt_sample").mkdir(exist_ok=True)
    (save_dir / "geometry_opt_plot").mkdir(exist_ok=True)
    (save_dir / "geometry_opt_log").mkdir(exist_ok=True)

    ks_path = data_path / "ks-opt"

    files = sorted((data_path / "xyz").glob("*.xyz"))
    task_inputs = _select_geometry_molecule_files(
        files,
        molecule_ids=molecule_ids,
        start_idx=start_idx,
        n_molecules=n_molecules,
    )
    if molecule_ids is not None and len(molecule_ids) > 0:
        logger.info(f"Running geometry optimization only for molecule ids: {molecule_ids}")

    if len(task_inputs) == 0:
        raise ValueError("No molecules selected for geometry optimization.")

    worker_devices = [f"cuda:{i}" if device == "cuda" else "cpu" for i in range(num_devices)]

    logger.trace("Starting Multiprocessing")
    # Queues

    task_queues = [mp.Queue() for _ in range(num_devices)]
    result_queue = mp.Queue()

    workers = []
    for i in range(num_devices):
        p = mp.Process(
            target=geometry_opt_worker,
            args=(
                run_path,
                negative_integrated_density_penalty_weight,
                ckpt_choice,
                ofdft_kwargs,
                optimizer,
                gradients,
                maxsteps,
                initial_guess_from_last,
                geometric_config,
                save_dir,
                ks_path,
                geometric_loglevel,
                worker_devices[i],
                transform_device,
                main_log_level,
                molecule_log_level,
                num_threads_per_process,
                task_queues[i],
                result_queue,
                nuclear_gradient_device,
                compute_geometry_energy_offset,
            ),
        )
        p.start()
        workers.append(p)
        # still an assertion error here
    logger.trace(workers)

    for task_idx, (molecule_id, file) in enumerate(task_inputs):
        worker_id = task_idx % num_devices
        logger.trace(
            f"Calculating molecule {molecule_id} ({task_idx + 1}/{len(task_inputs)}) "
            f"on worker: {worker_id}"
        )
        logger.trace(f"File: {file}")
        task_queues[worker_id].put((task_idx, molecule_id, file))

    # Signal workers to exit
    for q in task_queues:
        q.put(None)

    # Collect results
    results = [None] * len(task_inputs)
    for _ in range(len(task_inputs)):
        task_idx, result = result_queue.get()
        results[task_idx] = result

    # Wait for workers to finish
    for p in workers:
        p.join()

    logger.info("Starting summary plot")
    plot_file = save_dir / "geometry_opt_summary.pdf"
    plot_summary_geometry_optimization(results, filename=plot_file)
    logger.info(f"Plot saved to {plot_file}")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/ofdft",
    config_name="geometry_optimization.yaml",
)
def main(cfg: DictConfig):
    """Main function to use hydra main.

    Enables to also run the meth:`run_denop_processes` in code.
    """
    load_dotenv()

    extras(cfg)
    denop_cfg = cfg.get("denop_settings")

    geometric_config = cfg.get("geometric_config")

    ofdft_kwargs = {key: value for key, value in denop_cfg.items() if key in OFDFT_KWARGS}
    ofdft_kwargs["disable_pbar"] = cfg.get("disable_pbar")

    optimizer = instantiate(cfg.get("density_optimizer"))

    negative_integrated_density_penalty_weight = denop_cfg.get(
        "negative_integrated_density_penalty_weight", 0.0
    )

    run_geometry_process(
        run_path=cfg.get("run_path"),
        data_path=cfg.get("data_path"),
        log_file=cfg.get("log_file"),
        ckpt_choice=cfg.get("ckpt_choice"),
        n_molecules=cfg.get("n_molecules"),
        molecule_ids=cfg.get("molecule_ids"),
        start_idx=cfg.get("start_idx", 0),
        device=cfg.get("device"),
        transform_device=cfg.get("transform_device"),
        optimizer=optimizer,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        maxsteps=cfg.get("maxsteps"),
        gradients=cfg.get("gradients"),
        geometric_config=geometric_config,
        geometric_loglevel=cfg.get("geometric_loglevel"),
        main_log_level=cfg.get("main_log_level"),
        molecule_log_level=cfg.get("molecule_log_level"),
        ofdft_kwargs=ofdft_kwargs,
        initial_guess_from_last=cfg.get("initial_guess_from_last"),
        num_devices=cfg.get("num_devices"),
        num_threads_per_process=cfg.get("num_threads_per_process"),
        nuclear_gradient_device=cfg.get("nuclear_gradient_device"),
        compute_geometry_energy_offset=cfg.get("compute_geometry_energy_offset", False),
    )


if __name__ == "__main__":
    main()
