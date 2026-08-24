"""Computing Kohn-Sham data for a dataset.

It can be run as a script using the mldft_ks command. This file implements the functions for single
or multiprocessing a whole dataset or a subset of it. The configuration can be found in
configs/datagen/config.yaml. Possible options are:

* n_molecules: Number of molecules to compute, -1 for all molecules.
* start_idx: Index of the first molecule to compute (not using the actual index but indices from 0 to num_molecules).
* num_processes: Number of processes to use for multiprocessing.
* num_threads_per_process: Number of threads to use per process.
* verify_files: Whether to verify the files after computation.
* kohn_sham:

  * basis: Basis set to use for the molecule.
  * xc: Exchange correlation functional.
  * initialization: The initial guess method.

Example usage in a terminal:

.. code-block::

    mldft_ks n_molecules=10
    mldft_ks dataset=qm9 verify_files=true
    mldft_ks dataset=qmugs_first_bin kohn_sham.basis=sto-3g kohn_sham.xc=lda
    mldft_ks dataset=omol
"""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import hydra
import numpy as np
import rootutils
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pyscf import gto
from tqdm import tqdm

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.datagen.methods.ksdft_calculation import ksdft
from mldft.utils.filequeue import FileQueue
from mldft.utils.molecules import build_molecule_np
from mldft.utils.multiprocess import (
    configure_max_memory_per_process,
    configure_processes_and_threads,
    resolve_max_memory_per_process,
    resolve_processes_and_threads,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.utils import extras

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def check_config(cfg: DictConfig, path: Path) -> None:
    """Check if the configuration is the same as the one in the file if it exists."""
    if path.exists():
        logger.info("Config file already exists, checking if its the same.")
        old_cfg = OmegaConf.load(path)
        differences = []
        for key in old_cfg.keys():
            assert key in cfg, f"Key {key} not in new config."
            if not cfg[key] == old_cfg[key]:
                differences.append((key, cfg[key], old_cfg[key]))
        if len(differences) > 0:
            raise AssertionError(f"Differences found in config: {differences}")
        new_keys = set(cfg.keys()).difference(old_cfg.keys())
        if len(new_keys) > 0:
            logger.warning(f"New keys found in config: {new_keys}")


def save_config(cfg: DictConfig, path: Path) -> None:
    """Save the configuration to a yaml file.

    Args:
        cfg: Hydra configuration object.
        path: Path to the output file.
    """
    with open(path, "w") as f:
        logger.info(f"Saving config file to {path}.")
        f.write(OmegaConf.to_yaml(cfg))
    try:
        path.chmod(0o770)
    except PermissionError:
        logger.warning(f"Permission denied: could not change mode for {path}.")
    except Exception as e:
        logger.warning(f"An error occurred while changing mode for {path}: {e}")


def load_mol_or_iter(
    i: int,
    dataset: DataGenDataset,
    basis: str,
) -> gto.Mole | Iterable:
    charges, position, charge, multiplicity = dataset.load_molecular_properties(i)
    # Calculate spin from multiplicity: multiplicity = 2S + 1, so spin = multiplicity - 1
    spin = (multiplicity - 1) if multiplicity is not None else 0
    if spin < 0:
        raise ValueError(f"Invalid multiplicity {multiplicity} for molecule index {i}.")
    mol_or_iter = build_molecule_np(
        charges, position, basis=basis, unit="Angstrom", spin=spin, mol_charge=charge
    )
    mol_or_iter.verbose = 2
    return mol_or_iter


def run_ksdft_and_handle_exceptions(
    molecule: gto.Mole,
    xc: str,
    init_guess: str,
    grid_level: int,
    prune_method: Callable | None | str,
    density_fit_basis: str | None,
    density_fit_threshold: int | None,
    convergence_tolerance: float | None,
    output_file: Path,
    use_perturbation: bool = False,
    perturbation_cfg: DictConfig | None = None,
):
    """Run the Kohn-Sham iteration and handle exceptions as parallel process.

    Args:
        molecule: PySCF molecule object.
        xc: Exchange correlation functional.
        init_guess: The initial guess method.
        grid_level: The grid level to use for the integration grid used in the xc-functional.
        prune_method: The method to prune the grid.
        density_fit_basis: The basis set to use for the density fitting.
        density_fit_threshold: The threshold of number of atoms to enable density fitting.
        convergence_tolerance: The convergence tolerance for the Kohn-Sham iteration.
        output_file: Path to the output file.
        use_perturbation: Whether to use perturbation in the effective potential.
        perturbation_cfg: Settings for the perturbation.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            ksdft(
                molecule,
                tmp_path,
                xc_functional=xc,
                init_guess=init_guess,
                grid_level=grid_level,
                prune_method=prune_method,
                density_fit_basis=density_fit_basis,
                density_fit_threshold=density_fit_threshold,
                convergence_tolerance=convergence_tolerance,
                use_perturbation=use_perturbation,
                perturbation_cfg=perturbation_cfg,
            )
            tmp_file.close()  # Close the temporary file before renaming
            shutil.move(tmp_path, output_file)
    except KeyboardInterrupt:
        logger.error(
            f"Keyboard interrupt, stopping computation. Removing current file {tmp_path}."
        )
        if tmp_path.exists():
            tmp_path.unlink()
        if output_file.exists():
            output_file.unlink()
        raise
    except Exception as e:
        # Handle exceptions, log the error, and delete the temporary file
        logger.exception(f"An error occurred: {e}")
        if tmp_path.exists():
            tmp_path.unlink()


def run_kohn_sham_geometry(
    dataset: DataGenDataset,
    idx: int,
    output_dir: Path,
    filename: str,
    basis: str,
    xc: str,
    init_guess: str,
    grid_level: int,
    prune_method: Callable | None | str,
    density_fit_basis: str | None,
    density_fit_threshold: int | None,
    convergence_tolerance: float | None,
    use_perturbation: bool = False,
    perturbation_cfg: DictConfig | None = None,
):
    """Run Kohn-Sham DFT calculation for a given molecule index in the dataset."""
    mol_or_iter = load_mol_or_iter(idx, dataset, basis)
    mol_iterable = isinstance(mol_or_iter, Iterable)
    molecules = enumerate(mol_or_iter, start=1) if mol_iterable else [(1, mol_or_iter)]
    for sample_id, mol in molecules:
        if mol_iterable:
            output_file = output_dir / f"{filename}_{idx:012}.{sample_id:012}.chk"
        else:
            output_file = output_dir / f"{filename}_{idx:012}.chk"
        logger.info(
            f"Computing molecule {idx} {mole_to_sum_formula(mol, True)} with "
            f"{len(mol.atom_charges())} atoms."
        )
        run_ksdft_and_handle_exceptions(
            mol,
            xc,
            init_guess,
            grid_level,
            prune_method,
            density_fit_basis,
            density_fit_threshold,
            convergence_tolerance,
            output_file,
            use_perturbation,
            perturbation_cfg,
        )


def _count_task_files(directory: Path, suffix: str = ".task") -> int:
    return sum(
        1 for entry in directory.iterdir() if entry.is_file() and entry.name.endswith(suffix)
    )


def _progress_monitor(fq: FileQueue, pbar: tqdm, update_interval: int = 3600):
    """Monitor FileQueue progress and update tqdm bar."""
    while True:
        st = time.perf_counter()
        completed = _count_task_files(fq.done) + _count_task_files(fq.failed)
        duration = (time.perf_counter() - st) * 1000
        logger.trace(f"Counted completed tasks in {duration:.3f} ms")
        pbar.n = completed
        pbar.refresh()
        if pbar.total is not None and completed >= pbar.total:
            break
        time.sleep(update_interval)


def setup_progress_monitor_and_pbar(
    fq: FileQueue, cfg: DictConfig
) -> tuple[tqdm, threading.Thread]:
    """To get a progress bar for the total filequeue progress, we check the filequeue status
    periodically.

    Since this can be slow for large filequeues, we run it in a separate thread.
    """
    initial_completed = _count_task_files(fq.done) + _count_task_files(fq.failed)
    total_tasks = initial_completed + _count_task_files(fq.todo) + _count_task_files(fq.running)
    pbar = tqdm(
        total=total_tasks, initial=initial_completed, desc="Data Generation", dynamic_ncols=True
    )
    # Start progress monitor thread
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(fq, pbar, cfg.progress_update_interval),
        daemon=True,
    )
    monitor_thread.start()
    return pbar, monitor_thread


def _ks_worker(
    dataset: DataGenDataset,
    cfg: DictConfig,
    filequeue_kwargs: dict[str, Any],
):
    """Worker function for processing Kohn-Sham geometry tasks from a FileQueue.

    This function repeatedly leases tasks from the FileQueue associated with the
    dataset's Kohn-Sham data directory, and runs Kohn-Sham calculations
    for each leased task.
    Args:
        dataset: DataGenDataset instance containing molecular data and output directory.
        cfg: Hydra/OmegaConf configuration object with Kohn-Sham calculation parameters.
        filequeue_kwargs: Additional keyword arguments for FileQueue initialization.
    """

    fq = FileQueue(dataset.kohn_sham_data_dir, **filequeue_kwargs)
    while (lease := fq.lease()) is not None:
        with lease:
            idx = int(lease.task.stem)
            run_kohn_sham_geometry(
                dataset=dataset,
                idx=idx,
                output_dir=dataset.kohn_sham_data_dir,
                filename=dataset.filename,
                basis=cfg.kohn_sham.basis,
                xc=cfg.kohn_sham.xc,
                init_guess=cfg.kohn_sham.initialization,
                grid_level=cfg.kohn_sham.grid_level,
                prune_method=cfg.kohn_sham.prune_method,
                density_fit_basis=cfg.kohn_sham.density_fit_basis,
                density_fit_threshold=cfg.kohn_sham.density_fit_threshold,
                convergence_tolerance=cfg.kohn_sham.convergence_tolerance,
                use_perturbation=cfg.kohn_sham.use_perturbation,
                perturbation_cfg=cfg.kohn_sham.perturbation_cfg,
            )


def compute_kohn_sham_dataset(cfg: DictConfig) -> None:
    """Run the Kohn-Sham computations on a subset of the dataset.

    Args:
        cfg: Hydra configuration object.

    Raises:
        AssertionError: If more molecules are requested than available or if n_molecules is smaller than -1.
    """
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level=cfg.get("log_level", "INFO"),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    cfg.num_processes, cfg.num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    cfg.max_memory_per_process = configure_max_memory_per_process(
        cfg.get("max_memory_per_process")
    )
    # Load Dataset
    dataset_settings = OmegaConf.to_container(cfg.dataset, resolve=True)
    if isinstance(dataset_settings, dict):
        dataset_settings["num_processes"] = cfg.num_processes
    dataset = hydra.utils.instantiate(dataset_settings)
    # Start configuring the computation
    logger.info(f"Computing on Dataset: {dataset.name}")
    n_molecules = cfg.n_molecules
    start_idx = cfg.start_idx
    n_available_molecules = dataset.num_molecules
    if cfg.verify_files:
        dataset.verify_files()
    assert start_idx <= n_available_molecules, "More molecules requested than available."
    assert n_molecules >= -1, "n_molecules must be -1 or larger."
    # Get the indices that haven't been computed yet, starting at start_idx
    ids_this_run = dataset.get_ids_todo_ks(
        start_idx,
        max_num_molecules=n_molecules,
        selected_ids=cfg.get("selected_ids"),
    )
    # Log the configuration before starting the computation
    check_config(cfg.kohn_sham, dataset.kohn_sham_data_dir / "config.yaml")
    save_config(cfg.kohn_sham, dataset.kohn_sham_data_dir / "config.yaml")

    # Setup FileQueue
    if cfg.override_filequeue:
        shutil.rmtree(dataset.kohn_sham_data_dir / cfg.filequeue.queue_dir, ignore_errors=True)
    fq = FileQueue(dataset.kohn_sham_data_dir, **cfg.filequeue)
    logger.info(f"Using file queue at {fq.root}")
    if len(ids_this_run) == 0:
        existing_tasks = sorted(
            int(task.stem) for task in fq.todo.glob("*.task") if task.is_file()
        )
        if existing_tasks:
            logger.warning(
                f"Requested indices are already done, but found {len(existing_tasks)} pending tasks in the filequeue; processing those instead."
            )
            ids_this_run = np.asarray(existing_tasks, dtype=int)
    else:
        fq.seed_once(ids_this_run, suffix=".task")
    num_processes = min(cfg.num_processes, len(ids_this_run))
    if num_processes == 0:
        logger.info("No tasks to process; exiting.")
        return

    # Setup progress bar
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) % cfg.pbar_every_n_tasks == 0:
        pbar, monitor_thread = setup_progress_monitor_and_pbar(fq, cfg)
    else:
        pbar = tqdm(desc="Progress bar disabled for this task", total=0)
        monitor_thread = None

    try:
        if num_processes == 1:
            if cfg.num_threads_per_process == 1:
                logger.warning(
                    "Running in single process mode using 1 thread, this may take a while"
                )
            else:
                logger.info(
                    f"Running in single process mode using {cfg.num_threads_per_process} threads."
                )
            _ks_worker(
                dataset,
                cfg,
                cfg.filequeue,
            )
        else:
            logger.info(
                f"Using {num_processes} processes with {cfg.num_threads_per_process} threads each."
            )
            import multiprocessing

            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(
                    _ks_worker,
                    [(dataset, cfg, cfg.filequeue)] * num_processes,
                )
    except KeyboardInterrupt:
        logger.info("Main process interrupted, shutting down...")
        raise
    finally:
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=1)
        pbar.close()

    if cfg.verify_files:
        dataset.verify_files()
    logger.info("Done!")


@hydra.main(version_base="1.3", config_path="../../configs/datagen", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the Kohn-Sham computation on the whole dataset. Sets up the hydra
    specific logging and then calls the compute_kohn_sham_dataset function.

    Args:
        cfg: Hydra configuration object.
    """
    logger.remove()
    logger.add(cfg.log_file, rotation="10 MB", enqueue=True, backtrace=True, diagnose=True)
    cfg.num_processes, cfg.num_threads_per_process = resolve_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    cfg.max_memory_per_process = resolve_max_memory_per_process(cfg.get("max_memory_per_process"))
    extras(cfg)
    # Log configuration
    with open(cfg.config_file, "w") as f:
        logger.info(f"Saving config file to {cfg.config_file}.")
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    compute_kohn_sham_dataset(cfg)


if __name__ == "__main__":
    main()
