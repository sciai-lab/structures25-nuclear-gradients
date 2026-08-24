"""Precompute metadata cache for OMol dataset.

This script iterates through all molecules in the OMol dataset once and extracts per-molecule
metadata (charge, spin, data_id, atomic_numbers, etc.) into numpy arrays. The resulting cache file
enables fast in-memory filtering in generate_omol_dataset.py.
"""

import os
import pickle
from pathlib import Path

import hydra
import numpy as np
from fairchem.core.datasets import AseDBDataset
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from mldft.utils import extras


def precompute_omol_metadata(cfg: DictConfig) -> None:
    """Extract and cache metadata from all molecules in the OMol dataset.

    Args:
        cfg: Hydra configuration object.
    """
    output_path = Path(cfg.metadata_cache)
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Metadata cache {output_path} already exists. Set overwrite=True to overwrite."
        )

    dataset = AseDBDataset(config=dict(src=cfg.dataset_src, a2g_args=dict(task_name="omol")))
    total = len(dataset)
    logger.info(f"Dataset size: {total}")

    if cfg.omol_split == "train_4M":
        mapping_path = Path(cfg.dataset_src).parent / "4M_id_mapping.pkl"
        with open(mapping_path, "rb") as f:
            convert_idx = pickle.load(f)
    elif cfg.omol_split == "train":
        convert_idx = None
    else:
        raise ValueError(f"Unknown or unsupported omol_split: {cfg.omol_split}")

    # Preallocate arrays for scalar metadata
    charges = np.zeros(total, dtype=np.int8)
    spins = np.zeros(total, dtype=np.int8)
    n_atoms = np.zeros(total, dtype=np.int16)
    n_heavy_atoms = np.zeros(total, dtype=np.int16)
    # For variable-length data, use lists then convert
    data_ids = []
    sources = []
    # Store all atomic numbers per molecule as variable-length arrays
    atomic_numbers_list = []
    # Store global OMol indices
    global_indices = np.zeros(total, dtype=np.int64)

    pbar = tqdm(range(total), total=total, desc="Extracting metadata")
    for idx in pbar:
        atoms = dataset.get_atoms(idx)
        mol = dataset[idx]

        # Extract metadata
        charges[idx] = atoms.info.get("charge")
        spins[idx] = atoms.info.get("spin")
        data_ids.append(atoms.info.get("data_id"))
        sources.append(os.path.dirname(atoms.info.get("source")))

        atomic_numbers_np = mol.atomic_numbers.numpy()
        n_atoms[idx] = len(atomic_numbers_np)
        n_heavy_atoms[idx] = int((atomic_numbers_np > 1).sum())
        atomic_numbers_list.append(atomic_numbers_np.astype(np.int8))

        # Store global index
        if convert_idx is not None:
            global_indices[idx] = convert_idx[idx]
        else:
            global_indices[idx] = idx

    # Convert lists to arrays
    data_ids_arr = np.array(data_ids, dtype=object)
    sources_arr = np.array(sources, dtype=object)
    atomic_numbers_arr = np.array(atomic_numbers_list, dtype=object)

    # Save to npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        charges=charges,
        spins=spins,
        n_atoms=n_atoms,
        n_heavy_atoms=n_heavy_atoms,
        data_ids=data_ids_arr,
        sources=sources_arr,
        atomic_numbers=atomic_numbers_arr,
        global_indices=global_indices,
    )
    logger.info(f"Saved metadata cache to {output_path}")
    logger.info(f"Processed {len(charges)} molecules")


@hydra.main(
    version_base="1.3",
    config_path="../../configs/datagen",
    config_name="omol_dataset.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for metadata precomputation."""
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""), format=logger_format, colorize=True, enqueue=True
    )

    extras(cfg)
    precompute_omol_metadata(cfg)


if __name__ == "__main__":
    main()
