"""Applying a basis transformation to the dataset.

To convert a dataset to a new basis, this script can be run with:

.. code-block::

    python mldft/datagen/transform_dataset.py data=md17 data/transforms=local_frames

You can adapt ``data`` and ``data/transforms`` to your needs. The ``use_cached_data`` flag has to be set to ```false```.
Set ``verify_files=false`` to skip checking existing transformed files for validity.
"""

import multiprocessing
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import zarr
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import mldft.utils.omegaconf_resolvers  # noqa
from mldft.ml.data.components.basis_transforms import (
    ApplyBasisTransformation,
    MasterTransformation,
)
from mldft.ml.data.components.convert_transforms import ToNumpy, ToTorch
from mldft.ml.data.components.of_data import OFData
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.filequeue import FileQueue
from mldft.utils.multiprocess import configure_processes_and_threads
from mldft.utils.utils import set_default_torch_dtype


def validate_label(path: Path, spatial_keys_to_check: list[str]) -> tuple[bool, str]:
    """Validate a label file and return a reason if invalid."""
    required_root_keys = ("of_labels", "ks_labels", "geometry")
    required_of_label_keys = ("basis", "n_scf_steps", "spatial", "energies")
    try:
        with zarr.open(path, mode="r") as root:
            missing_root = [key for key in required_root_keys if key not in root]
            if missing_root:
                return False, f"missing root keys: {missing_root}"

            missing_of_labels = [
                key for key in required_of_label_keys if key not in root["of_labels"]
            ]
            if missing_of_labels:
                return False, f"missing of_labels keys: {missing_of_labels}"

            spatial_keys = set(root["of_labels/spatial"].keys())
            missing_spatial = [key for key in spatial_keys_to_check if key not in spatial_keys]
            if missing_spatial:
                return False, f"missing spatial keys: {missing_spatial}"

            return True, ""
    except Exception as e:
        return False, f"exception while checking file: {e}"


def remove_broken_file(args):
    path, old_label_dir, new_label_dir = args
    relative_path = path.relative_to(new_label_dir)
    old_path = old_label_dir / relative_path
    old_valid, old_reason = validate_label(old_path, spatial_keys_to_check=[])
    if not old_valid:
        raise ValueError(
            f"No valid labels in old path {old_path}. Consider removing it. Reason: {old_reason}"
        )

    # all spatial keys present in the old file should be present in the new file
    with zarr.open(str(old_path), mode="r") as root:
        old_spatial_keys = root["of_labels/spatial"].keys()
    # Exclude dual_basis_integrals as it's removed during transformation
    old_spatial_keys_to_check = [key for key in old_spatial_keys if key != "dual_basis_integrals"]
    new_valid, new_reason = validate_label(path, spatial_keys_to_check=old_spatial_keys_to_check)
    if not new_valid:
        logger.warning(
            f"Removing invalid label {path}, because it does not match {old_path}. "
            f"Reason: {new_reason}"
        )
        try:
            path.unlink()
            return path
        except Exception as e:
            logger.exception(f"Error while removing {path}: {e}")
    return None


def remove_broken_files(
    path_list: list[Path], old_label_dir: Path, new_label_dir: Path, num_processes: int = 1
):
    """Scan the list of zarr files and remove those that are broken, eg. can not be opened or have
    no of_labels.

    Args:
        path_list (list): List of paths to zarr files.
    """
    args_list = [(path, old_label_dir, new_label_dir) for path in sorted(path_list)]
    with multiprocessing.Pool(num_processes) as pool:
        removed = list(
            tqdm(
                pool.imap_unordered(remove_broken_file, args_list),
                total=len(args_list),
                position=0,
                smoothing=0,
                dynamic_ncols=True,
                desc="Verifying Files",
            )
        )
    return [path for path in removed if path is not None]


def convert_zarr_file(args):
    """Convert one label file with the new transforms.

    Args:
        args: Tuple of arguments for the convert function.
    """
    path, new_path, basis_info, transform = args
    logger.info(f"Processing {path.name}")
    try:
        with zarr.open(path, mode="r") as root:
            # These checks are not strictly necessary as we also demand the integrals to be present in the data loading,
            # but they are good for piece of mind.
            assert (
                "basis_integrals" in root["of_labels/spatial"]
            ), f"basis_integrals not in {path}. Probably the file was generated with an outdated version of mldft"

            scf_iterations = root["of_labels/n_scf_steps"][()]

            # Copy labels to memory store
            temp_store = zarr.MemoryStore()
            new_root_temp = zarr.open(temp_store, mode="w")

            try:
                zarr.copy_all(source=root, dest=new_root_temp)
            except Exception as e:
                logger.exception(f"Error copying {path}: {e}")
                raise

            spatial_keys = root["of_labels/spatial"].keys()
            # remove dual_basis_integrals as they are not needed anymore
            spatial_keys = [key for key in spatial_keys if key != "dual_basis_integrals"]

        # Delete copied spatial labels, to make sure none are incorrectly left untransformed
        of_spatial = new_root_temp.create_group("of_labels/spatial", overwrite=True)

        new_of_spatial = defaultdict(list)

        # Compute full transformation once and cache the resulting transformation matrices.
        sample = OFData.from_file_with_all_gradients(
            path,
            0,
            basis_info,
            energy_key="e_tot",
            gradient_key="grad_tot",
        )
        transform.add_transformation_matrix = True
        sample = transform(sample)
        if not (
            hasattr(sample, "transformation_matrix")
            and hasattr(sample, "inv_transformation_matrix")
        ):
            raise ValueError(f"Could not extract transformation matrices for {path}.")
        transformation_matrix = torch.as_tensor(sample.transformation_matrix, dtype=torch.float64)
        inv_transformation_matrix = torch.as_tensor(
            sample.inv_transformation_matrix, dtype=torch.float64
        )
        for key in spatial_keys:
            new_of_spatial[key].append(sample[key])

        # Reuse the same transformation for all other SCF iterations of the molecule.
        for i in range(1, scf_iterations):
            sample = OFData.from_file_with_all_gradients(
                path,
                i,
                basis_info,
                energy_key="e_tot",
                gradient_key="grad_tot",
            )
            if not transform.use_cached_data:
                for t in transform.additional_pre_transforms:
                    sample = t(sample)
            for t in transform.pre_transforms:
                sample = t(sample)
            sample = ApplyBasisTransformation()(
                sample, transformation_matrix, inv_transformation_matrix
            )
            for t in transform.post_transforms:
                sample = t(sample)
            for key in spatial_keys:
                new_of_spatial[key].append(sample[key])

        # Stack the lists to tensors and save them
        new_of_spatial = {key: np.stack(value) for key, value in new_of_spatial.items()}
        for key, value in new_of_spatial.items():
            chunks = (1, *value.shape[1:])
            of_spatial.create_dataset(key, data=value, chunks=chunks, dtype=np.float64)

        # Create new zip store and move everything from the temp store to the new store
        # This should be much better to prevent too many broken files
        with zarr.ZipStore(new_path, mode="w") as new_store:
            new_root = zarr.open(new_store, mode="w")
            zarr.copy_all(source=new_root_temp, dest=new_root)
        temp_store.clear()

    except Exception as e:
        logger.exception(f"Error while processing {path}: {e}")
        if new_path.exists():
            try:
                if new_path.is_dir():
                    shutil.rmtree(new_path)
                else:
                    new_path.unlink()
            except Exception as cleanup_error:
                logger.warning(
                    f"Could not remove partial output {new_path} after failure: {cleanup_error}"
                )
        raise


def _extract_numeric_prefix(path: Path) -> str:
    id_str = path.name.split(".", 1)[0]
    if not id_str.isdigit():
        raise ValueError(f"Could not parse numeric id from filename: {path.name}")
    return id_str


def _extract_molecule_id(path: Path) -> int:
    return int(_extract_numeric_prefix(path))


def _extract_id_padding(path: Path) -> int:
    return len(_extract_numeric_prefix(path))


def _path_from_id(old_label_dir: Path, idx: int, id_padding: int) -> Path:
    zarr_zip_path = old_label_dir / f"{idx:0{id_padding}d}.zarr.zip"
    if zarr_zip_path.exists():
        return zarr_zip_path
    return old_label_dir / f"{idx:0{id_padding}d}.zarr"


def _requeue_selected_terminal_tasks(fq: FileQueue, task_ids: list[int]) -> int:
    """Move selected tasks from done/failed back to todo for recomputation."""
    moved = 0
    lock = fq._acquire_lock()
    try:
        for idx in task_ids:
            name = f"{idx:012}.task"
            todo_task = fq.todo / name
            if todo_task.exists() or (fq.running / name).exists():
                continue
            for src_dir in (fq.done, fq.failed):
                src = src_dir / name
                if src.exists():
                    try:
                        os.replace(str(src), str(todo_task))
                        moved += 1
                    except FileNotFoundError:
                        pass
                    break
    finally:
        fq._release_lock(lock)
    return moved


def _transform_worker(
    lease_batch_size,
    id_padding,
    old_label_dir,
    new_label_dir,
    basis_info,
    transform,
    filequeue_kwargs,
):
    fq = FileQueue(new_label_dir, **filequeue_kwargs)
    while True:
        leases = fq.lease_many(lease_batch_size)
        if len(leases) == 0:
            break
        for lease in leases:
            try:
                with lease:
                    idx = int(lease.task.stem)
                    path = _path_from_id(old_label_dir, idx, id_padding)
                    new_path = new_label_dir / path.relative_to(old_label_dir)
                    if new_path.exists():
                        continue
                    convert_zarr_file(
                        (
                            path,
                            new_path,
                            basis_info,
                            transform,
                        )
                    )
            except Exception:
                logger.exception(
                    f"Task {lease.task.name} failed during transformation; continuing with remaining tasks."
                )


@set_default_torch_dtype(torch.float64)
def transform_dataset(cfg: DictConfig) -> None:
    """Script to basis transform the dataset.

    Applies the given transform to the dataset and saves the new labels in a new folder.

    Args:
        cfg (DictConfig): Config, see `configs/ml/train.yaml`.
    """
    num_processes, num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    log_level = cfg.get("log_level", "INFO")
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
        level=log_level,
    )
    # Configure transforms
    data_path = get_mldft_data_path()
    dataset_path = data_path / cfg.data.dataset_name
    basis_info = hydra.utils.instantiate(cfg.data.basis_info)
    cached_transforms = hydra.utils.instantiate(cfg.data.transforms.cached_transforms)
    master_transform = MasterTransformation(
        name=cfg.data.transforms.cached_transforms.name,
        use_cached_data=False,
        pre_transforms=[ToTorch(float_dtype=torch.float64)],
        cached_transforms=cached_transforms,
        post_transforms=[ToNumpy()],
    )
    old_label_dir = dataset_path / master_transform.label_subdir
    logger.info(f"Applying transform: {master_transform.name}")
    # This file handling here is not nice but good enough
    new_label_dir = old_label_dir.parent / f"labels_{master_transform.name}"
    new_label_dir.mkdir(exist_ok=True)
    try:
        new_label_dir.chmod(0o770)
    except PermissionError:
        logger.warning(f"Could not change permissions of {new_label_dir}.")
    log_file = cfg.get("log_file", str(new_label_dir / "transform_dataset.log"))
    logger.add(log_file, rotation="10 MB", enqueue=True, level=log_level)
    logger.info(f"Writing logs to {log_file} at level {log_level}")
    all_paths = sorted(old_label_dir.glob("*.zarr*"))
    if len(all_paths) == 0:
        logger.info("No source labels found; exiting.")
        return
    id_padding = _extract_id_padding(all_paths[0])

    start_idx = cfg.get("start_idx", 0)
    num_molecules = cfg.get("num_molecules", None)
    if num_molecules is None:
        num_molecules = len(all_paths)
    if start_idx + num_molecules > len(all_paths):
        end_idx = len(all_paths)
    else:
        end_idx = start_idx + num_molecules
    path_list = all_paths[start_idx:end_idx]

    paths = set(path_list)
    verify_files = cfg.get("verify_files", True)

    removed_broken_paths = []
    if verify_files:
        to_check = list(sorted(new_label_dir.glob("*.zarr*")))
        # Only check files in the range we're transforming
        to_check = [p for p in to_check if old_label_dir / p.relative_to(new_label_dir) in paths]
        removed_broken_paths = remove_broken_files(
            to_check, old_label_dir, new_label_dir, num_processes=num_processes
        )
    else:
        logger.info("Skipping verification of existing transformed files.")

    # Scan existing files (after optional deletion of broken files)
    new_paths_list = list(new_label_dir.glob("*.zarr*"))
    for zarr_file in tqdm(new_paths_list, desc="Identifying already converted files", leave=False):
        old_name = old_label_dir / zarr_file.relative_to(new_label_dir)
        if old_name in paths:
            paths.remove(old_name)
    selected_paths = sorted(paths)
    selected_ids = {_extract_molecule_id(path) for path in selected_paths}
    filequeue_kwargs = dict(cfg.get("filequeue", {}))
    lease_batch_size = int(filequeue_kwargs.pop("lease_batch_size", 1))
    if cfg.get("override_filequeue", False):
        shutil.rmtree(
            new_label_dir / filequeue_kwargs.get("queue_dir", "filequeue"), ignore_errors=True
        )
    fq = FileQueue(new_label_dir, **filequeue_kwargs)
    logger.info(f"Using file queue at {fq.root}")
    task_ids = sorted(selected_ids)
    removed_ids = sorted({_extract_molecule_id(path) for path in removed_broken_paths})
    requeued = _requeue_selected_terminal_tasks(fq, removed_ids)
    if requeued > 0:
        logger.info(f"Requeued {requeued} tasks from done/failed due to deleted broken outputs.")
    fq.seed_once(task_ids, suffix=".task")
    num_queued_tasks = len(list(fq.todo.glob("*.task")))
    if num_queued_tasks == 0:
        logger.info("No tasks in file queue; exiting.")
        return
    num_processes = min(num_processes, num_queued_tasks)
    if num_processes > 0:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                _transform_worker,
                [
                    (
                        lease_batch_size,
                        id_padding,
                        old_label_dir,
                        new_label_dir,
                        basis_info,
                        master_transform,
                        filequeue_kwargs,
                    )
                ]
                * num_processes,
            )
    logger.info("Finished transforming dataset!")
    return


@hydra.main(version_base="1.3", config_path="../../configs/ml", config_name="transform_dataset")
def main(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point for the script."""
    transform_dataset(cfg)


if __name__ == "__main__":
    main()
