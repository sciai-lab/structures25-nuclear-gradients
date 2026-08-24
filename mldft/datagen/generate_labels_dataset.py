"""Compute labels for molecules in the dataset and save them as zarr.zip files.

Can be run with the `mldft_labelgen` command.
"""

import multiprocessing
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.datagen.kohn_sham_dataset import (
    check_config,
    setup_progress_monitor_and_pbar,
)
from mldft.datagen.methods.label_generation import str_to_calculation_fct
from mldft.datagen.methods.label_generation_from_ground_state import (
    calculate_labels_with_ground_state_perturbations,
)
from mldft.datagen.methods.save_labels_in_zarr_file import save_density_fitted_data
from mldft.utils import extras
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path
from mldft.utils.filequeue import FileQueue
from mldft.utils.molecules import build_molecule_np, construct_aux_mol, load_scf
from mldft.utils.multiprocess import (
    configure_max_memory_per_process,
    configure_processes_and_threads,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula


def get_id_and_sample_id_from_chk_file(file: Path) -> tuple[int, int | None]:
    """Get the molecule id and sample id from the .chk file.

    Args:
        file: The .chk file.

    Returns:
        Tuple of molecule id and sample id.
    """
    split_filename = file.stem.split("_")[-1].split(".")
    if len(split_filename) == 2:
        molecule_id, sample_id = int(split_filename[0]), int(split_filename[1])
    else:
        molecule_id = int(split_filename[0])
        sample_id = None
    return molecule_id, sample_id


def get_zarr_file_path(label_dir: Path, molecule_id: int, sample_id: int | None = None) -> Path:
    if sample_id is not None:
        return label_dir / f"{molecule_id:012}.{sample_id:012}.zarr.zip"
    else:
        return label_dir / f"{molecule_id:012}.zarr.zip"


def run_labelgen_task(
    dataset: DataGenDataset,
    chk_file: Path,
    calculation_fct: Callable,
    orbital_basis: str,
    kohn_sham_basis: str,
    of_basis_set: str,
):
    """Run the label generation task for a single molecule."""
    molecule_id, sample_id = get_id_and_sample_id_from_chk_file(chk_file)
    # Load charges and build molecule in a way similar to the improved Kohn-Sham logic:
    charges, positions, charge, multiplicity = dataset.load_molecular_properties(molecule_id)
    # Convert multiplicity to spin: spin = multiplicity - 1
    spin = multiplicity - 1 if multiplicity is not None else 0
    mol_orbital_basis = build_molecule_np(
        charges, positions, mol_charge=charge, spin=spin, basis=orbital_basis, unit="Angstrom"
    )
    # Preserve spin in the auxiliary molecule so zarr stores the correct multiplicity.
    mol_density_basis = construct_aux_mol(
        mol_orbital_basis,
        aux_basis_name=of_basis_set,
        unit="Bohr",
        spin=spin,
    )
    logger.info(
        f"Computing molecule {molecule_id} {mole_to_sum_formula(mol_orbital_basis, True)} with "
        f"{len(mol_orbital_basis.atom_charges())} atoms."
    )

    label_path = get_zarr_file_path(dataset.label_dir, molecule_id, sample_id)
    results, initialization, data_of_iteration = load_scf(chk_file)

    data = calculation_fct(
        results=results,
        initialization=initialization,
        data_of_iteration=data_of_iteration,
        mol_orbital_basis=mol_orbital_basis,
        mol_density_basis=mol_density_basis,
    )
    save_density_fitted_data(
        label_path,
        mol_density_basis,
        kohn_sham_basis=kohn_sham_basis,
        path_to_basis_info=(chk_file.parent / "basis_info.npz").as_posix(),
        molecular_data=data,
        mol_id=chk_file.stem,
    )


def save_dataset_info(cfg: DictConfig, label_dir: Path):
    """Save relevant config information for the dataset in a yaml file."""
    keys_to_save = ("kohn_sham", "density_fitting_method", "of_basis_set", "dataset")
    dataset_info = {key: cfg.get(key) for key in keys_to_save}
    OmegaConf.save(config=dataset_info, f=label_dir / "dataset_info.yaml")


def _labelgen_worker(
    dataset: DataGenDataset,
    calculation_fct: Callable,
    orbital_basis: str,
    kohn_sham_basis: str,
    of_basis_set: str,
    filequeue_kwargs: dict[str, Any],
):
    """Worker function for multiprocessing label generation using FileQueue.

    It repeatedly leases tasks from a FileQueue, processes all associated .chk files for each
    molecule, and generates labels using the provided calculation function and
    basis set parameters.
    Args:
        dataset: DataGenDataset instance containing molecule and label info.
        calculation_fct: Callable for label generation (e.g., DFT calculation).
        orbital_basis: Name of the orbital basis set to use.
        kohn_sham_basis: Name of the Kohn-Sham basis set to use.
        of_basis_set: Name of the orbital-free basis set to use.
        filequeue_kwargs: Additional keyword arguments for FileQueue setup.
    """

    fq = FileQueue(dataset.label_dir, **filequeue_kwargs)
    while (lease := fq.lease()) is not None:
        with lease:
            chk_files = dataset.get_all_chk_files_from_id(int(lease.task.stem))
            for chk_file in chk_files:
                run_labelgen_task(
                    dataset=dataset,
                    chk_file=chk_file,
                    calculation_fct=calculation_fct,
                    orbital_basis=orbital_basis,
                    kohn_sham_basis=kohn_sham_basis,
                    of_basis_set=of_basis_set,
                )


def run_label_generation(cfg: DictConfig):
    """Run the label generation for the dataset."""
    # Setup logger to log via tqdm and to the configured log file
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        level=cfg.get("log_level", "INFO"),
        colorize=True,
        enqueue=True,
    )

    configured_num_processes, num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )

    # Instantiate Dataset
    dataset_settings = cast(dict[str, Any], OmegaConf.to_container(cfg.dataset, resolve=True))
    dataset_settings["num_processes"] = configured_num_processes
    dataset = hydra.utils.instantiate(dataset_settings)
    kohn_sham_settings = cfg.kohn_sham
    check_config(cfg.kohn_sham, dataset.kohn_sham_data_dir / "config.yaml")
    orbital_basis = kohn_sham_settings.basis

    # Save dataset info (config) in the directory above the labels directory
    save_dataset_info(cfg, dataset.label_dir.parent)

    density_fitting_method = cfg.get("density_fitting_method")
    of_basis_set = cfg.get("of_basis_set")
    ids_this_run = dataset.get_ids_todo_labelgen(
        start_idx=cfg.start_idx,
        max_num_molecules=cfg.n_molecules,
        selected_ids=cfg.get("selected_ids"),
    )

    calculation_fct_name = cfg.label_src.calculation_fct

    if calculation_fct_name == "calculate_labels_with_ground_state_perturbations":
        calculation_fct = partial(
            calculate_labels_with_ground_state_perturbations,
            perturbation_cfg=cfg.label_src.perturbation_cfg,
            density_fitting_method=density_fitting_method,
        )
    else:
        calculation_fct = str_to_calculation_fct[calculation_fct_name]
        calculation_fct = partial(calculation_fct, density_fitting_method=density_fitting_method)

    if calculation_fct_name == "active_learning":
        run_path = parse_run_path(cfg.label_src.run_path)
        iterations_per_mol = cfg.label_src.iterations_per_mol
        max_scale = cfg.label_src.max_scale

        # Check dataset compatibility
        model_config = OmegaConf.load(run_path / ".hydra/config.yaml")
        model_dataset_name = model_config.data.dataset_name
        if model_dataset_name.split("_")[0] != dataset.name.split("_")[0]:
            logger.warning(f"Dataset mismatch: {dataset.name} vs {model_dataset_name}")

        model_config.data.transforms.use_cached_data = False
        model_config.data.transforms.add_transformation_matrix = True
        master_transform = hydra.utils.instantiate(model_config.data.transforms)
        basis_info = hydra.utils.instantiate(model_config.data.basis_info)
        checkpoint_path = run_to_checkpoint_path(run_path)
        torch.set_default_dtype(torch.float64)
        calculation_fct = partial(
            calculation_fct,
            checkpoint_path=checkpoint_path,
            master_transform=master_transform,
            basis_info=basis_info,
            iterations_per_mol=iterations_per_mol,
            max_scale=max_scale,
        )

    # Setup FileQueue
    if cfg.override_filequeue:
        shutil.rmtree(dataset.label_dir / cfg.filequeue.queue_dir, ignore_errors=True)
    fq = FileQueue(dataset.label_dir, **cfg.filequeue)
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

    # Setup progress bar
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) % cfg.pbar_every_n_tasks == 0:
        pbar, monitor_thread = setup_progress_monitor_and_pbar(fq, cfg)
    else:
        pbar = tqdm(desc="Progress bar disabled for this task", total=0)
        monitor_thread = None

    try:
        num_processes = min(configured_num_processes, len(ids_this_run))
        if num_processes == 0:
            logger.info("No tasks to process; exiting.")
            return
        if num_processes == 1:
            if num_threads_per_process == 1:
                logger.warning(
                    "Running in single process mode using 1 thread, this may take a while"
                )
            else:
                logger.info(
                    f"Running in single process mode using {num_threads_per_process} threads."
                )
            _labelgen_worker(
                dataset,
                calculation_fct,
                orbital_basis,
                kohn_sham_settings.basis,
                of_basis_set,
                cfg.filequeue,
            )
        else:
            logger.info(
                f"Using {num_processes} processes with {num_threads_per_process} threads each."
            )
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(
                    _labelgen_worker,
                    [
                        (
                            dataset,
                            calculation_fct,
                            orbital_basis,
                            kohn_sham_settings.basis,
                            of_basis_set,
                            cfg.filequeue,
                        )
                    ]
                    * num_processes,
                )
    except KeyboardInterrupt:
        logger.info("Main process interrupted, shutting down...")
        raise
    finally:
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=1)
        pbar.close()

    logger.info("Label generation and saving done.")


@hydra.main(version_base="1.3", config_path="../../configs/datagen", config_name="config.yaml")
def main(cfg: DictConfig):
    """Hydra entry point for label generation."""
    logger.remove()
    logger.add(cfg.log_file, rotation="10 MB", enqueue=True, backtrace=True, diagnose=True)
    cfg.num_processes, cfg.num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    cfg.max_memory_per_process = configure_max_memory_per_process(
        cfg.get("max_memory_per_process")
    )
    extras(cfg)
    run_label_generation(cfg)


if __name__ == "__main__":
    main()
