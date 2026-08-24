"""Run script for density optimization using a trained model."""

# this should be in the scripts directory, no?
# it is also overloaded
import os

# Limit BLAS/OpenMP thread pools unless the user already configured them.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import csv
import pickle
import queue
import signal
import threading
import time
from functools import partial
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Callable, List, Tuple
from urllib.parse import quote

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pyscf import gto
from rootutils import rootutils
from tqdm import tqdm

import mldft.utils.omegaconf_resolvers  # noqa
from mldft.ml.data.components.basis_transforms import (
    ApplyBasisTransformation,
    transform_tensor_with_sample,
)
from mldft.ml.data.components.convert_transforms import PrepareForDensityOptimization
from mldft.ml.data.components.dataset import HDF5ShardOFDataset, OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import BasisInfo, Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.density_optimization import (
    density_optimization,
    density_optimization_with_label,
)
from mldft.ofdft.energies import Energies
from mldft.ofdft.functional_factory import FunctionalFactory, requires_grid
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.utils import extras
from mldft.utils.continue_denop import (
    get_existing_sample_paths,
    get_missing_paths_and_scf_iters,
    merge_denop_configs,
)
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import (
    get_mldft_data_path,
    parse_run_path,
    run_to_checkpoint_path,
)
from mldft.utils.instantiators import instantiate_model
from mldft.utils.omegaconf_resolvers import to_no_basis_transforms_dataset_statistics
from mldft.utils.pdf_utils import HierarchicalPlotPDF
from mldft.utils.plotting.plot_density_optimization import (
    plot_density_optimization,
)
from mldft.utils.plotting.summary_density_optimization import (
    density_optimization_summary_pdf_plot,
    get_runwise_density_optimization_data,
    save_density_optimization_metrics,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sad_guesser import SADNormalizationMode
from mldft.utils.sample import add_density_optimization_trajectories_to_sample
from mldft.utils.sample_generator import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def construct_label_path(data_dir: Path, dataset: str, label_path: str) -> Path:
    """Constructs the full path to a label file."""
    return data_dir / dataset / "labels" / label_path


def get_paths_from_split(
    split_dict: dict, split_name: str, data_dir: Path
) -> Tuple[List[Path], List[int]]:
    """Extracts paths and SCF iterations from a specific split (train/val/test)."""
    split_files = split_dict[split_name]
    paths = [
        construct_label_path(data_dir, dataset, label_path)
        for dataset, label_path, _ in split_files
    ]
    scf_iterations = [scf_iterations for _, _, scf_iterations in split_files]
    return paths, scf_iterations


def find_molecule_path_by_id(
    molecule_id: int, split_dict: dict, data_dir: Path, user_split: str
) -> Tuple[Path, int] | None:
    """Searches for a molecule by its ID across all splits (train, val, test).

    Looks for the molecule ID in the filename stem (e.g. "31415" in "00031415.zarr.zip") and
    returns the path and SCF iterations if found.

    For performance reasons, we first look into the user-specified split, then into the other
    splits in the order train, val, test.
    """
    split_order = [user_split]
    for split in ["train", "val", "test"]:
        if split not in split_order:
            split_order.append(split)

    for split_name in split_order:
        split_files = split_dict[split_name]
        for dataset, label_path, scf_iterations in split_files:
            path = construct_label_path(data_dir, dataset, label_path)
            filename = path.name.removesuffix(".zarr.zip").rsplit("_", 1)[-1]
            if filename.isdigit() and int(filename) == molecule_id:
                return path, scf_iterations

    return None


def resolve_torch_threads_per_worker(torch_threads_per_worker: int | None) -> int:
    """Resolve Torch intra-op threads for a denop worker thread."""
    return max(1, int(torch_threads_per_worker or 1))


def configure_dataset_indices(
    dataset_size: int,
    n_molecules: int,
    molecule_choice: str | list,
    seed: int | None,
    start_idx: int = 0,
) -> np.ndarray:
    """Configure the indices of the dataset to optimize.

    If n_molecules==-1, all molecules in the dataset are used.
    """
    if n_molecules == -1:
        n_molecules = dataset_size
    if n_molecules > dataset_size:
        logger.warning(
            f"Requested {n_molecules} molecules, but only {dataset_size} are available. "
            f"Setting n_molecules to {dataset_size}."
        )
        n_molecules = dataset_size
    if molecule_choice == "first_n":
        dataset_indices = np.arange(n_molecules)[start_idx:]
    elif molecule_choice == "random":
        if start_idx != 0:
            logger.warning(
                f"Random molecule choice with start_idx {start_idx}. Ignoring start_idx."
            )
        dataset_indices = np.random.choice(dataset_size, n_molecules, replace=False)
    elif molecule_choice == "seeded_random":
        rng = np.random.default_rng(seed)
        # When calling this with different n_molecules, the order stays the same
        dataset_indices = rng.choice(dataset_size, n_molecules, replace=False)[start_idx:]
    elif isinstance(molecule_choice, (list, ListConfig)):
        if len(molecule_choice) < n_molecules:
            logger.warning(
                f"Requested {n_molecules} molecules, but only {len(molecule_choice)} are "
                f"available. Setting n_molecules to {len(molecule_choice)}."
            )
            n_molecules = len(molecule_choice)
        elif len(molecule_choice) > n_molecules:
            logger.warning(
                f"Requested {n_molecules} molecules, but {len(molecule_choice)} are available. "
                f"Using the first {n_molecules} molecules."
            )
        dataset_indices = np.array(molecule_choice[start_idx:n_molecules])
    elif isinstance(molecule_choice, str) and molecule_choice.startswith("always_"):
        index = int(molecule_choice.split("_")[1])
        dataset_indices = np.array([index] * n_molecules)
    else:
        raise ValueError(
            f"Unknown molecule choice {molecule_choice}. "
            f"Must be 'first_n', 'random', 'seeded_random', or a list of indices."
        )
    return dataset_indices


def set_torch_defaults_worker(id: int, num_threads: int, device: torch.device | str):
    """Set the torch defaults for a dataloader worker."""
    torch.set_default_dtype(OFDFT_DTYPE)
    torch.set_default_device(device)
    torch.set_num_threads(resolve_torch_threads_per_worker(num_threads))


def denop_thread(
    thread_idx: int,
    process_idx: int,
    num_threads: int,
    dataset: OFDataset,
    basis_info: BasisInfo,
    lightning_module: MLDFTLitModule,
    proj_minao_module: MLDFTLitModule,
    device: str | torch.device,
    optimizer: Optimizer,
    xc_functional: str,
    ks_basis: str,
    max_xc_memory: int,
    negative_integrated_density_penalty_weight: float,
    initialization: str,
    normalize_initial_guess: bool,
    dataset_statistics_path: Path | str,
    convergence_criterion: str,
    csv_lock: LockType,
    plot_queue,
    plot_every_n: int,
    save_dir: Path,
    fail_fast: bool,
    save_coeff_interval: int,
    compute_final_hessian: bool,
    termination_event: EventType,
    torch_threads_per_worker: int,
):
    """Worker thread for density optimization."""
    torch_threads_per_worker = resolve_torch_threads_per_worker(torch_threads_per_worker)
    torch.set_num_threads(torch_threads_per_worker)
    torch.set_default_device(device)
    global_idx = thread_idx + process_idx * num_threads
    delay_per_thread = 1.0  # seconds
    time.sleep(delay_per_thread * global_idx)  # wait a bit to let all threads start
    logger.info(f"Thread initialized with ID: {global_idx}")
    dataset_length = len(dataset)
    dataloader = OFLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # separate processes can cause issues with multithreading
        pin_memory=False,
        prefetch_factor=None,
        list_keys=[],
        worker_init_fn=partial(
            set_torch_defaults_worker, num_threads=torch_threads_per_worker, device=device
        ),
    )
    if initialization in ["sad", "sad_transformed"]:
        sad_guess_kwargs = dict(
            dataset_statistics=DatasetStatistics(dataset_statistics_path),
            normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
            basis_info=basis_info,
            weigher_key="ground_state_only",
            spherical_average=True,
        )
    else:
        sad_guess_kwargs = {}

    func_factory = FunctionalFactory.from_module(
        lightning_module,
        xc_functional,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
    )
    start_time = time.time()
    for i, sample in enumerate(dataloader):
        if termination_event.is_set():
            logger.info("Termination requested, exiting thread loop.")
            break
        try:
            sample = sample.to_data_list()[0]
            sample.to(device)
            logger.trace(
                f"Optimizing {mole_to_sum_formula(sample.mol, True):<16}"
                f"{sample.mol_id:<15} on GPU {process_idx}, thread {thread_idx} "
                f"({i + 1}/{dataset_length})"
            )
            callback = ConvergenceCallback()
            result = density_optimization_with_label(
                sample,
                mol=sample.mol,
                optimizer=optimizer,
                func_factory=func_factory,
                callback=callback,
                max_xc_memory=max_xc_memory,
                normalize_initial_guess=normalize_initial_guess,
                ks_basis=ks_basis,
                proj_minao_module=proj_minao_module,
                initialization=initialization,
                sad_guess_kwargs=sad_guess_kwargs,
                convergence_criterion=convergence_criterion,
                position=global_idx,
            )
            metric_dict, callback, energies_label, energy_functional = result
            end_time = time.time()
            time_per_mol = end_time - start_time
            sample.add_item("time", time_per_mol, representation=Representation.SCALAR)
            logger.info(
                f"{sample.mol_id:<15} {mole_to_sum_formula(sample.mol, use_subscript=True):<15}"
                f"ΔE={metric_dict['gs_signed_energy_error'] * 1000:6.2g} mHa, "
                f"|Δρ|={metric_dict['gs_l2_norm']:6.2g}, "
                f"|g|={metric_dict['gs_gradient_norm']:7.2e}, "
                f"iters={metric_dict['stopping_index'] + 1:<4}"
            )

            if compute_final_hessian:
                # compute the hessian of the functional at the stopping index
                logger.info("Computing the Hessian of the energy functional at the final index")
                hess_sample = sample.clone()
                hessian = energy_functional(hess_sample, compute_hessian=True)[2]
                eigenvalues, _ = torch.linalg.eigh(hessian)
                logger.info(
                    f"{'Hessian':<11}{mole_to_sum_formula(sample.mol, True):<16} number of "
                    f"negative eigenvalues: {(eigenvalues < -1e-6).sum()}/{len(eigenvalues)}"
                )
                sample.add_item(
                    "final_hessian_eigenvalues",
                    eigenvalues,
                    representation=Representation.SCALAR,
                )
                untransformed_hessian = transform_tensor_with_sample(
                    sample, hessian, Representation.BILINEAR_FORM, invert=True
                )
                eigenvalues, _ = torch.linalg.eigh(untransformed_hessian)
                sample.add_item(
                    "final_untransformed_hessian_eigenvalues",
                    eigenvalues,
                    representation=Representation.SCALAR,
                )

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
            sample = add_density_optimization_trajectories_to_sample(
                sample, callback, energies_label, save_coeff_interval
            )
            sample.add_item(
                "ks_basis",
                ks_basis,
                representation=Representation.NONE,
            )

            # Write CSV data directly
            csv_path = save_dir / "reduced_denop_data.csv"
            stopping_index = sample.stopping_index

            sample_trajectory_path_for_molecule = (
                save_dir / "sample_trajectories" / f"sample_{sample.mol_id}.pt"
            )

            saved_data = {
                "mol_id": sample.mol_id,
                "sum_formula": mole_to_sum_formula(sample.mol, True),
                "ground_state_energy_total": sample.ground_state_energy_total,
                "final_energy_total": sample.trajectory_energy_total[stopping_index].item(),
                "final_density_l2_norm": sample.trajectory_l2_norm[stopping_index].item(),
                "final_gradient_norm": sample.trajectory_gradient_norm[stopping_index].item(),
                "total_energy_error_per_atom": torch.abs(
                    sample.trajectory_energy_total[stopping_index]
                    - sample.ground_state_energy_total
                ).item()
                / sample.n_atom,
                "density_l2_error_per_electron": sample.trajectory_l2_norm[stopping_index].item()
                / sample.n_electron,
                "run_time": sample.time,
                "iterations": stopping_index,
                "path_to_sample_tra": sample_trajectory_path_for_molecule,
            }

            with csv_lock:
                file_exists = csv_path.exists()
                with open(csv_path, mode="a", newline="") as f:
                    fieldnames = saved_data.keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(saved_data)

            torch.save(sample, sample_trajectory_path_for_molecule)
            if plot_queue is not None and i % plot_every_n == 0:
                plot_path = save_dir / "plot_trajectories" / f"sample_{sample.mol_id}.pt"
                sample_full_trajectory = add_density_optimization_trajectories_to_sample(
                    sample, callback, energies_label, save_coeff_interval=1
                )
                torch.save(sample_full_trajectory, plot_path)
                plot_queue.put(plot_path)

            start_time = time.time()
        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt in thread. Exiting.")
            termination_event.set()
            return
        except Exception as e:
            logger.exception(f"Error in thread {thread_idx} during density optimization: {e}")
            if fail_fast:
                raise e
            else:
                logger.warning(
                    f"Skipping molecule {sample.mol_id} due to error. "
                    "Set fail_fast=True to raise immediately."
                )
                continue

    return


def denop_process(
    process_idx: int,
    num_threads: int,
    ckpt_path: Path,
    guess_ckpt_path: Path,
    paths,
    num_scf_iterations_per_path,
    device: str,
    model_dtype,
    deterministic: bool,
    batch_size: int,
    batch_timeout: float,
    basis_info: BasisInfo,
    transforms,
    dataset_kwargs: dict,
    optimizer: Optimizer,
    xc_functional: str,
    ks_basis: str,
    max_xc_memory: int,
    negative_integrated_density_penalty_weight: float,
    initialization: str,
    normalize_initial_guess: bool,
    dataset_statistics_path: Path | str,
    convergence_criterion: str,
    csv_lock: LockType,
    plot_queue: mp.Queue,
    plot_every_n: int,
    save_dir: Path,
    fail_fast: bool,
    save_coeff_interval: int,
    compute_final_hessian: bool = False,
    termination_event: EventType | None = None,
    torch_threads_per_worker: int = 1,
):
    """Density optimization that starts multiple threads for batched denop."""
    torch_threads_per_worker = resolve_torch_threads_per_worker(torch_threads_per_worker)
    torch.set_default_device(device)
    torch.set_default_dtype(OFDFT_DTYPE)
    torch.set_num_threads(torch_threads_per_worker)

    # README: this is a 'hack' to avoid the lazy-loading error: RuntimeError: lazy wrapper should be called at most once
    # https://github.com/pytorch/pytorch/issues/90613
    # basic reason: linalg library is big and is thus lazily loaded when needed
    # this must only happen once per process, but threading will do so in each thread as soon as some linalg
    # function is called ==> RuntimeError b/c a race condition occurs on the initialization singleton
    # this is a (hopefully) temporary workaround until the lazy-loading logic of CUDA is fully integrated into torch and the
    # CUDA version that don´t have this yet are dropped. For now, it needs to stay whenever we use threading.Threads though.
    torch.inverse(torch.ones((1, 1), device=device))

    if termination_event is None:
        termination_event = mp.Event()
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )

    # Relay termination signals to all worker threads so they exit promptly.
    def _handle_signal(signum, _frame):
        logger.warning(f"Process {process_idx} received signal {signum}. Initiating shutdown.")
        termination_event.set()
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    # load models once for shared use across threads
    lightning_module = instantiate_model(
        ckpt_path, device, model_dtype, deterministic, batch_size, batch_timeout
    )
    if guess_ckpt_path is not None:
        proj_minao_module = instantiate_model(guess_ckpt_path, device, model_dtype, deterministic)
    else:
        proj_minao_module = lightning_module

    threads = []
    paths = np.array_split(paths, num_threads)
    num_scf_iterations_per_path = np.array_split(num_scf_iterations_per_path, num_threads)

    for thread_idx in range(num_threads):
        if len(paths[thread_idx]) == 0:
            continue
        local_dataset_kwargs = dataset_kwargs.copy()
        hdf5_manifest_path = local_dataset_kwargs.pop("hdf5_manifest_path", None)
        if hdf5_manifest_path is not None:
            manifest_path = get_mldft_data_path() / Path(str(hdf5_manifest_path))
            if manifest_path.exists():
                source_keys = [
                    path.relative_to(get_mldft_data_path()).as_posix()
                    for path in paths[thread_idx]
                ]
                dataset = HDF5ShardOFDataset(
                    source_keys=source_keys,
                    num_scf_iterations_per_path=num_scf_iterations_per_path[thread_idx],
                    basis_info=basis_info,
                    transforms=transforms,
                    hdf5_manifest_path=manifest_path,
                    **local_dataset_kwargs,
                )
            else:
                dataset = OFDataset(
                    paths=paths[thread_idx],
                    num_scf_iterations_per_path=num_scf_iterations_per_path[thread_idx],
                    basis_info=basis_info,
                    transforms=transforms,
                    **local_dataset_kwargs,
                )
        else:
            dataset = OFDataset(
                paths=paths[thread_idx],
                num_scf_iterations_per_path=num_scf_iterations_per_path[thread_idx],
                basis_info=basis_info,
                transforms=transforms,
                **local_dataset_kwargs,
            )
        kwargs = {
            "thread_idx": thread_idx,
            "process_idx": process_idx,
            "num_threads": num_threads,
            "dataset": dataset,
            "basis_info": basis_info,
            "lightning_module": lightning_module,
            "proj_minao_module": proj_minao_module,
            "device": device,
            "optimizer": optimizer,
            "xc_functional": xc_functional,
            "ks_basis": ks_basis,
            "max_xc_memory": max_xc_memory,
            "negative_integrated_density_penalty_weight": negative_integrated_density_penalty_weight,
            "initialization": initialization,
            "normalize_initial_guess": normalize_initial_guess,
            "dataset_statistics_path": dataset_statistics_path,
            "convergence_criterion": convergence_criterion,
            "csv_lock": csv_lock,
            "plot_queue": plot_queue,
            "plot_every_n": plot_every_n,
            "save_dir": save_dir,
            "fail_fast": fail_fast,
            "save_coeff_interval": save_coeff_interval,
            "compute_final_hessian": compute_final_hessian,
            "termination_event": termination_event,
            "torch_threads_per_worker": torch_threads_per_worker,
        }
        if num_threads > 1:
            t = threading.Thread(
                target=denop_thread,
                kwargs=kwargs,
            )
            t.daemon = True
            t.start()
            threads.append(t)
        else:
            denop_thread(**kwargs)
            return

    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
                if termination_event.is_set():
                    break
    except KeyboardInterrupt:
        termination_event.set()
        logger.warning("KeyboardInterrupt during thread join in process. Propagating.")
        raise


def plotting_worker(
    plot_queue: mp.Queue,
    save_dir: Path,
    basis_info: BasisInfo,
    plotting_functions: List[Callable],
    save_individual_plots: bool = False,
    num_threads: int = 8,
    fail_fast: bool = False,
    termination_event: EventType | None = None,
):
    """Worker process for handling plotting."""
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    with HierarchicalPlotPDF(
        out_pdf_path=save_dir / "density_optimization.pdf",
        individual_plot_directory=(
            save_dir / "individual_results" if save_individual_plots else None
        ),
    ) as pdf:
        while True:
            if termination_event is not None and termination_event.is_set():
                logger.info("Termination event set, stopping plotting worker.")
                break
            plot_path = None
            try:
                if termination_event is not None:
                    plot_path = plot_queue.get(timeout=1.0)
                else:
                    plot_path = plot_queue.get()
                if plot_path is None:  # sentinel value
                    break

                # Load plotting data
                sample = torch.load(plot_path, map_location="cpu", weights_only=False)
                coeff_dim = sample.ground_state_coeffs.shape[0]
                sample.add_item(
                    "transformation_matrix",
                    torch.eye(coeff_dim, dtype=OFDFT_DTYPE),
                    representation=Representation.VECTOR,
                )
                sample.add_item(
                    "inv_transformation_matrix",
                    torch.eye(coeff_dim, dtype=OFDFT_DTYPE),
                    representation=Representation.VECTOR,
                )
                # Create plots
                basis_l = basis_info.l_per_basis_func[sample.basis_function_ind]
                final_hessian_eigenvalues = None
                if hasattr(sample, "final_hessian_eigenvalues"):
                    final_hessian_eigenvalues = sample.final_hessian_eigenvalues
                plot_density_optimization(
                    sample,
                    sample.ground_state_coeffs,
                    plots=plotting_functions,
                    stopping_index=sample.stopping_index,
                    basis_l=basis_l,
                    final_hessian=final_hessian_eigenvalues,
                    add_mol_view_link=True,
                )
                pdf.savefig(f"{sample.mol_id}")
                plt.close()
                # Clean up temporary file
                os.unlink(plot_path)

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                if plot_path is not None:
                    try:
                        os.unlink(plot_path)
                    except OSError:
                        pass
                break
            except Exception as e:
                logger.exception(f"Error in plotting worker: {e}")
                if plot_path is not None:
                    try:
                        os.unlink(plot_path)
                    except OSError:
                        pass
                if fail_fast:
                    raise e


def evaluate_density_optimization(
    save_dir: Path,
    compute_grid_metrics: bool = True,
    compare_to_ks: bool = False,
    metrics_grid_level: int = 3,
    metrics_grid_prune: str = "nwchem_prune",
    swarm_plot_subsample: float = 1.0,
    plot_summary: bool = True,
):
    run_data_dict = get_runwise_density_optimization_data(
        sample_dir=save_dir / "sample_trajectories",
        compute_grid_metrics=compute_grid_metrics,
        compare_to_ks=compare_to_ks,
        metrics_grid_level=metrics_grid_level,
        metrics_grid_prune=metrics_grid_prune,
    )
    save_density_optimization_metrics(
        output_path=save_dir / "density_optimization_metrics.yaml",
        run_data_dict=run_data_dict,
    )
    if plot_summary:
        logger.info("Saving density optimization summary plot.")
        density_optimization_summary_pdf_plot(
            out_pdf_path=save_dir / "density_optimization_summary.pdf",
            run_data_dict=run_data_dict,
            subsample=swarm_plot_subsample,
        )


@set_default_torch_dtype(OFDFT_DTYPE)
def run_denop_processes(
    ckpt_path: Path,
    model_config: DictConfig,
    optimizer: Optimizer,
    split_file_path: str,
    split: str = "val",
    guess_ckpt_path: Path | str | None = None,
    device: torch.device | str = "cpu",
    num_devices: int = 1,
    num_threads: int = 8,
    torch_threads_per_worker: int = 1,
    batch_size: int = 0,
    batch_timeout: float = 0.05,
    model_dtype: torch.dtype = OFDFT_DTYPE,
    xc_functional: str = "PBE",
    ks_basis: str = "6-31G(2df,p)",
    initialization: str = "minao",
    n_molecules: int = 1,
    molecule_choice: str | list[int] = "first_n",
    molecule_ids: list[int] | None = None,
    seed: int = 0,
    log_file: Path | None = None,
    save_individual_plots: bool = False,
    plot_every_n: int = 10,
    plot_summary: bool = True,
    swarm_plot_subsample: float = 1.0,
    normalize_initial_guess: bool = True,
    grid_level: int = 2,
    grid_prune: str = "nwchem_prune",
    max_xc_memory: int = 4000,
    compute_grid_metrics: bool = True,
    compare_to_ks: bool = False,
    metrics_grid_level: int = 3,
    metrics_grid_prune: str = "nwchem_prune",
    plot_l1_norm: bool = True,
    negative_integrated_density_penalty_weight: float = 0.0,
    convergence_criterion: str = "last_iter",
    fail_fast: bool = False,
    deterministic: bool = True,
    save_coeff_interval: int = 100,
    compute_final_hessian: bool = False,
    continue_denop: bool = False,
    denop_dir: Path | str = None,
    plotting_functions: List[Callable] = None,
):
    """Script to run ofdft using a model checkpoint on multiple molecules.

    Note: Right now this only supports density optimizations using a checkpoint.

    Args:
        ckpt_path: The path to the model checkpoint.
        model_config: The model configuration.
        optimizer: The optimizer to use, e.g., :py:class:`~mldft.ofdft.optimizer.GradientDescent`.
        split_file_path: The path to the split file to use.
        split: The split to use, i.e. "train", "val" or "test".
        guess_ckpt_path: The path to the guess checkpoint. If None, then the same model is used
            for the proj_minao guess.
        device: The device to run on.
        num_devices: The number of devices to run on.
        num_threads: The number of denop worker threads per device to use. Should be at
            least `batch_size`.
        torch_threads_per_worker: The number of Torch intra-op threads to use inside each
            denop worker thread.
        batch_size: The batch size for the model inference.
        batch_timeout: The timeout for the batch inference in seconds.
        model_dtype: The dtype of the model.
        xc_functional: The XC functional to use. Ignored if it is part of the model prediction.
        initialization: The initial guess of the density.
        n_molecules: The number of molecules to optimize.
        molecule_choice: The choice of molecules to optimize. Options are "first_n", "random",
            "seeded_random", or a list of indices.
        molecule_ids: If set, only these molecule ids are optimized, and n_molecules and
            molecule_choice are ignored. Note that the molecule_ids are only unique for the given
            dataset, i.e. the same id can occur in different datasets. To make this unique, the
            dataset name would have to be included in a molecule id as well, making it a string.
            For now, this is only usable within a single dataset.
        seed: The random seed to use if molecule_choice is "seeded_random".
        log_file: The path to the log file. If None, then no log file is created.
        save_individual_plots: Whether to keep individual plots for each molecule.
        plot_every_n: How often to plot during the density optimization. If 0, no intermediate
            plots are created.
        plot_summary: Whether to plot the density optimization summary.
        swarm_plot_subsample: The subsample factor for the swarm plots.
        normalize_initial_guess: Whether to normalize the initial guess to the correct number of
            electrons.
        grid_level: The grid level of the integration grid, if required by the functional.
        grid_prune: The pruning method of the integration grid, if required by the functional.
        max_xc_memory: The maximum memory in MB to use for the XC functional evaluation.
        compute_grid_metrics: Whether to plot the L1 norm of the density error, for which the integration
            grid is required.
        metrics_grid_level: The grid level of the integration grid for the L1 norm.
        metrics_grid_prune: The pruning method for the integration grid for the L1 norm.
        negative_integrated_density_penalty_weight: The weight of the integrated negative density
            penalty.
        convergence_criterion: The convergence criterion for the density optimization.
        fail_fast: Whether to raise an error immediately if a molecule fails. If False, errors are
            logged, but the script continues. Useful to set to True for debugging.
        deterministic: The top-level ``deterministic`` attribute of the model will be set to this
            value. Models that need deterministic edge-frame construction, such as EquiformerV2
            and eSEN, should route that flag internally.
        save_coeff_interval: How often to save the coefficients during the density optimization.
        compute_final_hessian: Whether to compute the final hessian of the energy functional.
        continue_denop: Whether to continue a given density optimization.
        denop_dir: OFDFT run directory which should be continued if ``continue_denop=True``.
        plotting_functions: List of plotting functions to use to create the plots.
    """
    mp.set_start_method("spawn", force=True)

    # I think these are expensive to use in the end.
    transforms = instantiate(model_config.data.transforms)
    basis_info = instantiate(model_config.data.basis_info)
    add_grid = requires_grid(
        model_config.data.target_key, negative_integrated_density_penalty_weight
    )
    transforms.pre_transforms.insert(
        0, PrepareForDensityOptimization(basis_info, add_grid, grid_level, grid_prune)
    )
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False
    split_file_path = Path(split_file_path)
    if not split_file_path.is_absolute():
        split_file_path = get_mldft_data_path() / split_file_path

    dataset_kwargs = instantiate(model_config.data.datamodule.dataset_kwargs)
    dataset_kwargs.pop("hdf5_manifest_path", None)
    dataset_kwargs.pop("denop_hdf5_manifest_path", None)
    denop_manifest_path = (
        split_file_path.parent / f"labels_denop_h5_{model_config.data.target_key}/manifest.json"
    )
    dataset_kwargs["hdf5_manifest_path"] = str(denop_manifest_path)
    if denop_manifest_path.exists():
        logger.info(f"Using denop HDF5 shards: {denop_manifest_path}")
    else:
        logger.info(
            f"Denop HDF5 manifest not found at {denop_manifest_path}. "
            "Falling back to zarr labels for density optimization."
        )
    dataset_statistics_path = model_config.data.dataset_statistics.path
    if initialization == "sad":
        dataset_statistics_path = to_no_basis_transforms_dataset_statistics(
            dataset_statistics_path, model_config.data.transforms.name
        )
    if device == "cuda":
        assert (
            num_devices <= torch.cuda.device_count()
        ), f"Configured {num_devices} devices but only {torch.cuda.device_count()} GPUs available."

    save_dir = log_file.parent
    save_dir.mkdir(exist_ok=True)  # for saving the denop trajectories of the samples
    (save_dir / "sample_trajectories").mkdir(exist_ok=True)
    (save_dir / "plot_trajectories").mkdir(exist_ok=True)
    torch.save(basis_info, save_dir / "sample_trajectories" / "basis_info.pt")

    csv_lock = mp.Lock()
    # start_dynamical_denop(save_dir)
    encoded_path = quote(str(save_dir), safe="")
    logger.info(f"View the denop table at http://localhost:60501/run/{encoded_path}")

    data_dir = get_mldft_data_path()
    with open(split_file_path, "rb") as f:
        split_dict = pickle.load(f)

    dataset_kwargs.update(
        {
            "limit_scf_iterations": -1,
            "additional_keys_at_ground_state": {
                "of_labels/energies/e_electron": Representation.SCALAR,
                "of_labels/energies/e_ext": Representation.SCALAR,
                "of_labels/energies/e_hartree": Representation.SCALAR,
                "of_labels/energies/e_kin": Representation.SCALAR,
                "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
                "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
                "of_labels/energies/e_kinapbe": Representation.SCALAR,
                "of_labels/energies/e_xc": Representation.SCALAR,
                "of_labels/energies/e_tot": Representation.SCALAR,
            },
        }
    )

    # Remove kwargs unsupported by OFDataset when we fall back from HDF5 to zarr.
    for _key in (
        "hdf5_max_open_shards",
        "label_path",
        "kohn_sham_path",
        "ground_state_coeffs",
    ):
        dataset_kwargs.pop(_key, None)

    if plotting_functions is None:
        plotting_functions = []
    plot_process = None
    plot_queue = None
    termination_event = mp.Event()
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, _frame):
        logger.warning(f"Main process received signal {signum}. Initiating shutdown.")
        termination_event.set()
        raise KeyboardInterrupt()

    # Ensure Ctrl+C/SIGTERM bubble down to child processes and threads via the shared event.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if plot_every_n > 0:
        plot_queue = mp.Queue()
        plot_process = mp.Process(
            target=plotting_worker,
            args=(
                plot_queue,
                save_dir,
                basis_info,
                plotting_functions,
                save_individual_plots,
                num_threads,
                fail_fast,
                termination_event,
            ),
        )
        plot_process.daemon = True
        plot_process.start()

    if molecule_ids is not None and len(molecule_ids) > 0:
        all_paths = []
        all_num_scf_iterations_per_path = []

        for molecule_id in molecule_ids:
            result = find_molecule_path_by_id(molecule_id, split_dict, data_dir, split)
            if result is None:
                raise ValueError(
                    f"Molecule {molecule_id} not found in any split (train, val, test)"
                )
            all_paths.append(result[0])
            all_num_scf_iterations_per_path.append(result[1])

        logger.info(f"Running density optimization ONLY for molecules {molecule_ids}")
    else:
        val_paths, val_iterations = get_paths_from_split(split_dict, split, data_dir)
        dataset_indices = configure_dataset_indices(
            len(val_paths),
            n_molecules,
            molecule_choice,
            seed,
        )
        all_paths = [val_paths[j] for j in dataset_indices]
        all_num_scf_iterations_per_path = [val_iterations[j] for j in dataset_indices]

    if continue_denop:
        logger.info(f"Continuing density optimization of directory {str(denop_dir)}")
        existing_paths = get_existing_sample_paths(denop_dir)
        paths, num_scf_iterations_per_path = get_missing_paths_and_scf_iters(
            all_paths, all_num_scf_iterations_per_path, existing_paths
        )
        if len(paths) == 0:
            logger.info(
                f"All queried samples are already present in the directory {str(denop_dir)}. "
                "If you want to redo the density optimization, set 'continue_denop=false'."
            )
            if plot_process is not None:
                plot_queue.put(None)
                plot_process.join()
            return
        logger.info(f"Found {len(paths)} missing molecules, that will be run.")
    else:
        paths = all_paths
        num_scf_iterations_per_path = all_num_scf_iterations_per_path

    processes = []
    num_labels_to_process = len(paths)
    paths = np.array_split(paths, num_devices)
    num_scf_iterations_per_path = np.array_split(num_scf_iterations_per_path, num_devices)

    # this part might be better with a pool instead of ta loop
    # also it might be better to construct as little as possible on the
    # processes themselves?
    try:
        for process_idx in range(num_devices):
            process_device = "cpu" if device == "cpu" else f"cuda:{process_idx}"
            denop_kwargs = {
                "process_idx": process_idx,
                "num_threads": num_threads,
                "torch_threads_per_worker": torch_threads_per_worker,
                "ckpt_path": ckpt_path,
                "guess_ckpt_path": guess_ckpt_path,
                "paths": paths[process_idx],
                "num_scf_iterations_per_path": num_scf_iterations_per_path[process_idx],
                "device": process_device,
                "model_dtype": model_dtype,
                "deterministic": deterministic,
                "batch_size": batch_size,
                "batch_timeout": batch_timeout,
                "basis_info": basis_info,
                "transforms": transforms,
                "dataset_kwargs": dataset_kwargs,
                "optimizer": optimizer,
                "xc_functional": xc_functional,
                "ks_basis": ks_basis,
                "max_xc_memory": max_xc_memory,
                "negative_integrated_density_penalty_weight": negative_integrated_density_penalty_weight,
                "initialization": initialization,
                "normalize_initial_guess": normalize_initial_guess,
                "dataset_statistics_path": dataset_statistics_path,
                "convergence_criterion": convergence_criterion,
                "csv_lock": csv_lock,
                "plot_queue": plot_queue,
                "plot_every_n": plot_every_n,
                "save_dir": save_dir,
                "fail_fast": fail_fast,
                "save_coeff_interval": save_coeff_interval,
                "compute_final_hessian": compute_final_hessian,
                "termination_event": termination_event,
            }

            if num_devices == 1:
                denop_process(**denop_kwargs)
            else:
                p = mp.Process(target=denop_process, kwargs=denop_kwargs)
                p.daemon = True
                p.start()
                processes.append(p)

        if num_devices > 1:
            for p in processes:
                p.join()

        # how much time does this need?
        logger.success("Density optimization completed successfully. Evaluating metrics.")
        torch.set_default_device("cpu")
        evaluate_density_optimization(
            save_dir,
            compute_grid_metrics=compute_grid_metrics,
            compare_to_ks=compare_to_ks,
            metrics_grid_level=metrics_grid_level,
            metrics_grid_prune=metrics_grid_prune,
            swarm_plot_subsample=swarm_plot_subsample,
            plot_summary=plot_summary,
        )
        # finalize plotting process
        if plot_process is not None:
            logger.info("Finalizing plot process.")
            plot_queue.put(None)
            plot_process.join()

        logger.info(f"Density optimization saved to {save_dir}")

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Terminating child processes.")
        for p in processes:
            if p.is_alive():
                p.terminate()
        if plot_process is not None:
            if plot_process.is_alive():
                plot_process.terminate()
        termination_event.set()
        raise
    finally:
        if (plot_trajectories_dir := save_dir / "plot_trajectories").exists():
            try:
                plot_trajectories_dir.rmdir()
                logger.debug(f"Removed temporary directory {plot_trajectories_dir}")
            except OSError:
                logger.warning(f"{plot_trajectories_dir} is not empty; keeping directory.")
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


@hydra.main(
    version_base="1.3", config_path="../../configs/ofdft", config_name="density_optimization.yaml"
)
def main(cfg: DictConfig):
    """Main function to use hydra main.

    Enables to also run the meth:`run_denop_processes` in code.
    """
    load_dotenv()
    extras(cfg)

    log_file = cfg.get("log_file")
    if cfg.continue_denop:
        denop_dir = Path(str(cfg.denop_dir))
        log_file = denop_dir / "ofdft.log"  # append to the existing log file
    elif log_file is not None:
        log_file = Path(log_file)
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    if log_file is not None:
        log_file = Path(log_file)
        logger.add(
            log_file, level="TRACE", rotation="10 MB", enqueue=True, backtrace=True, diagnose=True
        )
        logger.info(f'Logging to "{log_file}"')

    run_path = parse_run_path(cfg.run_path)
    cfg.run_path = str(run_path)
    denop_cfg = cfg.get("denop_settings")
    optimizer = instantiate(cfg.get("density_optimizer"))
    model_config_path = run_path / "hparams.yaml"
    model_config_path_resolved = run_path / "hparams_resolved.yaml"
    model_config = OmegaConf.load(model_config_path)
    model_config_resolved = OmegaConf.load(model_config_path_resolved)

    split_file_path = cfg.get("split_file_path")
    if split_file_path is None:
        if hasattr(model_config.data.datamodule.split_file, "output_file"):
            # need to use the resolved config here, as the output_file contains an interpolation
            # of paths.output_dir which here points to the denop path, not the original model path
            split_file_path = model_config_resolved.data.datamodule.split_file.output_file
        else:
            split_file_path = model_config.data.datamodule.split_file
    cfg.split_file_path = str(split_file_path)

    ckpt_path = run_to_checkpoint_path(run_path, cfg.ckpt_choice)
    if cfg.get("guess_path") is not None:
        guess_ckpt_path = run_to_checkpoint_path(cfg.guess_path, cfg.guess_ckpt_choice)
    else:
        guess_ckpt_path = None
    with open_dict(cfg):
        cfg.ckpt_path = str(ckpt_path)
        cfg.guess_ckpt_path = str(guess_ckpt_path)

    if cfg.get("continue_denop"):
        cfg = merge_denop_configs(cfg)

    OmegaConf.save(cfg, log_file.parent / "hparams_resolved.yaml", resolve=True)

    run_denop_processes(
        ckpt_path=ckpt_path,
        model_config=model_config,
        optimizer=optimizer,
        split_file_path=split_file_path,
        split=cfg.get("split"),
        guess_ckpt_path=guess_ckpt_path,
        device=cfg.get("device"),
        num_devices=cfg.get("num_devices"),
        num_threads=cfg.get("num_threads"),
        torch_threads_per_worker=cfg.get("torch_threads_per_worker"),
        batch_size=cfg.get("batch_size"),
        batch_timeout=cfg.get("batch_timeout"),
        model_dtype=str_to_torch_float_dtype(cfg.get("model_dtype", OFDFT_DTYPE)),
        xc_functional=denop_cfg.get("xc_functional"),
        ks_basis=denop_cfg.get("ks_basis"),
        initialization=denop_cfg.get("initialization"),
        n_molecules=cfg.get("n_molecules"),
        molecule_choice=cfg.get("molecule_choice"),
        molecule_ids=cfg.get("molecule_ids"),
        seed=cfg.get("seed"),
        log_file=log_file,
        plot_every_n=cfg.get("plot_every_n"),
        plot_summary=cfg.get("plot_summary"),
        save_individual_plots=cfg.get("save_individual_plots"),
        swarm_plot_subsample=cfg.get("swarm_plot_subsample"),
        normalize_initial_guess=denop_cfg.get("normalize_initial_guess"),
        grid_level=denop_cfg.get("grid_level"),
        grid_prune=denop_cfg.get("grid_prune"),
        max_xc_memory=denop_cfg.get("max_xc_memory"),
        compute_grid_metrics=cfg.get("compute_grid_metrics"),
        compare_to_ks=cfg.get("compare_to_ks"),
        metrics_grid_level=cfg.get("metrics_grid_level"),
        metrics_grid_prune=cfg.get("metrics_grid_prune"),
        negative_integrated_density_penalty_weight=denop_cfg.get(
            "negative_integrated_density_penalty_weight"
        ),
        convergence_criterion=denop_cfg.get("convergence_criterion"),
        fail_fast=cfg.get("fail_fast"),
        deterministic=cfg.get("deterministic"),
        save_coeff_interval=cfg.get("save_coeff_interval"),
        compute_final_hessian=cfg.get("compute_final_hessian"),
        continue_denop=cfg.get("continue_denop"),
        denop_dir=cfg.get("denop_dir"),
        plotting_functions=(
            [instantiate(func) for func in cfg.get("subplot_list")]
            if cfg.get("subplot_list") is not None
            else []
        ),
    )


if __name__ == "__main__":
    main()


def run_singlepoint_ofdft(
    mol: gto.Mole,
    sample_generator: SampleGenerator,
    func_factory: FunctionalFactory,
    optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
    initial_guess_str: str = "minao",
    callback: ConvergenceCallback | None = None,
    ofdft_kwargs=None,
) -> Tuple[Energies, torch.Tensor, bool]:
    """Run a single-point OFDFT calculation for the given molecule.

    Args:
        mol: The molecule.
        sample_generator: The sample generator.
        func_factory: The functional factory.
        optimizer: The optimizer.
        initial_guess_str: The initial guess.
        callback: The callback.

    Returns:
        The final energies, coefficients and whether the calculation converged.
    """
    if ofdft_kwargs is None:
        ofdft_kwargs = dict()
    sample = sample_generator.get_sample_from_mol(mol)
    final_energies, final_coeffs, converged, _ = density_optimization(
        sample,
        sample.mol,
        optimizer,
        func_factory,
        callback,
        initialization=initial_guess_str,
        **ofdft_kwargs,
    )
    return final_energies, final_coeffs.detach().cpu(), converged
