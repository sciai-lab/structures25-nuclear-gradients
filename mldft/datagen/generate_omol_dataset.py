"""Filter OMol dataset.

This script selects molecules from an AseDB OMol source according to a set of optional filters and
writes information about the selected molecules into a dedicated dataset folder which can then be
used for OF data generation.

For fast filtering, run precompute_omol_metadata.py first to create a metadata cache. The cache
contains per-molecule metadata (charge, spin, data_id, atomic_numbers, etc.) that can be filtered
in-memory instead of iterating through 100M molecules.
"""

import hashlib
import os
import pickle
import shlex
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath

import hydra
import numpy as np
from fairchem.core.datasets import AseDBDataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pyscf import gto
from tqdm import tqdm

from mldft.utils import extras
from mldft.utils.molecules import xyz_string_with_charge_and_multiplicity

# Sized for the 4M subset: biggest family (omol) lands at ~5k files per shard.
SHARD_WIDTH = 2


def _flatten_source(source: str) -> str:
    """Map an OMol source path to the flat sharded destination path.

    The first path component stays as the dataset/family directory. The next
    directory is the first SHARD_WIDTH characters of SHA1(source), spreading
    files across bounded shard directories. The remaining source components are
    flattened into a single filename with "__" separators, and the first 8 hash
    characters are inserted before the original filename to avoid collisions
    while keeping suffix-based filtering simple.

    Example:
        ani1xbb/aniBB_007_252373_0_1/density_mat.npz
        -> ani1xbb/15/aniBB_007_252373_0_1__154788a8__density_mat.npz
    """
    parts = PurePosixPath(source).parts
    family = parts[0] if parts else "unknown"
    digest = hashlib.sha1(source.encode()).hexdigest()
    relative_parts = parts[1:] or (family,)
    flat = "__".join((*relative_parts[:-1], digest[:8], relative_parts[-1]))
    return f"{family}/{digest[:SHARD_WIDTH]}/{flat}"


def _load_metadata_cache(cache_path: str) -> dict:
    """Load precomputed metadata cache.

    Args:
        cache_path: Path to the metadata_cache.npz file.

    Returns:
        Dictionary with arrays: charges, spins, n_atoms, n_heavy_atoms,
        data_ids, sources, atomic_numbers, global_indices.
    """
    logger.info(f"Loading metadata cache from {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _filter_with_cache(cfg: DictConfig, cache: dict) -> tuple[np.ndarray, dict]:
    """Filter molecules using cached metadata.

    Args:
        cfg: Hydra configuration containing filter settings.
        cache: Dictionary with cached metadata arrays.

    Returns:
        Tuple of (selected_local_indices, data_stats).
    """
    total = len(cache["charges"])

    # Create permutation of all indices first for reproducibility
    rng = np.random.default_rng(0)
    indices = rng.permutation(total)

    mask = np.ones(total, dtype=bool)

    # Filter by charge-multiplicity pairs
    if cfg.get("allowed_charges_per_multiplicity") is not None:
        charge_mult_dict = {
            int(mult): [int(c) for c in charges]
            for mult, charges in cfg.allowed_charges_per_multiplicity.items()
        }
        pair_mask = np.zeros(total, dtype=bool)
        for mult, charges in charge_mult_dict.items():
            m_mask = cache["spins"] == mult
            c_mask = np.isin(cache["charges"], np.array(charges, dtype=np.int8))
            pair_mask |= m_mask & c_mask
        mask &= pair_mask
        logger.info(f"After charge-multiplicity filter: {mask.sum()} / {total}")

    # Filter by data_id
    if cfg.get("allowed_data_ids") is not None:
        allowed = set(cfg.allowed_data_ids)
        data_id_mask = np.array([d in allowed for d in cache["data_ids"]], dtype=bool)
        mask &= data_id_mask
        logger.info(f"After data_id filter: {mask.sum()} / {total}")

    # Filter by heavy atom count
    if cfg.get("max_num_heavy_atoms") is not None:
        mask &= cache["n_heavy_atoms"] <= cfg.max_num_heavy_atoms
        logger.info(f"After max_heavy_atoms filter: {mask.sum()} / {total}")
    if cfg.get("min_num_heavy_atoms") is not None:
        mask &= cache["n_heavy_atoms"] >= cfg.min_num_heavy_atoms
        logger.info(f"After min_heavy_atoms filter: {mask.sum()} / {total}")

    # Filter by allowed elements
    if cfg.get("allowed_elements") is not None:
        allowed = {int(x) for x in cfg.allowed_elements}
        # Check if all elements in each molecule are in allowed set
        element_mask = np.array(
            [set(nums.tolist()).issubset(allowed) for nums in cache["atomic_numbers"]],
            dtype=bool,
        )
        mask &= element_mask
        logger.info(f"After element filter: {mask.sum()} / {total}")

    # Apply filters in permuted order
    shuffled_mask = mask[indices]
    passing_indices = indices[shuffled_mask]
    logger.info(f"Total passing basic filters: {len(passing_indices)}")

    # Apply mols_per_atom_count constraint (requires sequential iteration)
    if cfg.get("mols_per_atom_count") is not None:
        mols_per_count = cfg.mols_per_atom_count
        heavy_atom_counts = defaultdict(int)
        selected = []
        for idx in passing_indices:
            n_heavy = cache["n_heavy_atoms"][idx]
            if heavy_atom_counts[n_heavy] < mols_per_count:
                selected.append(idx)
                heavy_atom_counts[n_heavy] += 1
        passing_indices = np.array(selected, dtype=np.int64)
        logger.info(f"After mols_per_atom_count filter: {len(passing_indices)}")

    # Apply max_num_mols limit
    if cfg.get("max_num_mols") is not None and len(passing_indices) > cfg.max_num_mols:
        passing_indices = passing_indices[: cfg.max_num_mols]
        logger.info(f"After max_num_mols limit: {len(passing_indices)}")

    # Compute statistics from selected molecules
    data_stats = {
        "multiplicities": defaultdict(int),
        "charges": defaultdict(int),
        "data_ids": defaultdict(int),
        "elements": defaultdict(int),
        "n_heavy_atoms": defaultdict(int),
        "n_atoms": defaultdict(int),
    }
    for idx in passing_indices:
        data_stats["multiplicities"][int(cache["spins"][idx])] += 1
        data_stats["charges"][int(cache["charges"][idx])] += 1
        data_stats["data_ids"][cache["data_ids"][idx]] += 1
        data_stats["n_heavy_atoms"][int(cache["n_heavy_atoms"][idx])] += 1
        data_stats["n_atoms"][int(cache["n_atoms"][idx])] += 1
        for Z in cache["atomic_numbers"][idx]:
            data_stats["elements"][int(Z)] += 1

    return passing_indices, data_stats


def _molecule_allowed(
    cfg: DictConfig,
    atoms_info,
    mol,
    data_stats: defaultdict,
) -> bool:
    """Decide whether a molecule meets the selection criteria.

    Args:
        cfg: Hydra configuration containing filter settings.
        atoms_info: metadata dict returned by ``dataset.get_atoms`` for this
            entry (contains keys like "spin", "charge", "data_id", "source").
        mol: the dataset molecule object with attributes ``atomic_numbers`` and
            ``pos`` (a torch Tensor).
        data_stats: defaultdict collecting statistics about the selected
            molecules. In particular, the number of heavy atoms already
            selected for each size bin is stored here to enforce the
            ``mols_per_atom_count`` restriction.

    Returns:
        bool: True if the molecule meets all selection criteria, False otherwise.
    """
    # Restrict charge-multiplicity pairs
    if cfg.allowed_charges_per_multiplicity is not None:
        charge_mult_dict = {
            int(mult): {int(c) for c in charges}
            for mult, charges in cfg.allowed_charges_per_multiplicity.items()
        }
        multiplicity = atoms_info.get("spin")
        charge = atoms_info.get("charge", None)
        if multiplicity not in charge_mult_dict:
            return False
        if charge not in charge_mult_dict[multiplicity]:
            return False

    # Restrict data_id (OMol subset)
    allowed_data_ids = set(cfg.allowed_data_ids) if cfg.allowed_data_ids is not None else None
    data_id = atoms_info.get("data_id", None)
    if allowed_data_ids is not None and data_id not in allowed_data_ids:
        return False

    # Restrict molecule size
    atomic_numbers_np = mol.atomic_numbers.numpy()
    heavy_atoms_mask = atomic_numbers_np > 1
    num_heavy_atoms = int(heavy_atoms_mask.sum())
    if cfg.max_num_heavy_atoms is not None and num_heavy_atoms > cfg.max_num_heavy_atoms:
        return False
    if cfg.min_num_heavy_atoms is not None and num_heavy_atoms < cfg.min_num_heavy_atoms:
        return False
    if (
        cfg.mols_per_atom_count is not None
        and data_stats["n_heavy_atoms"][num_heavy_atoms] >= cfg.mols_per_atom_count
    ):
        return False

    # Restrict allowed elements
    allowed_elements_arr = (
        np.array([int(x) for x in cfg.allowed_elements], dtype=np.int64)
        if cfg.allowed_elements is not None
        else None
    )
    if allowed_elements_arr is not None:
        bad_mask = ~np.isin(atomic_numbers_np, allowed_elements_arr)
        bad_count = int(bad_mask.sum())
        if bad_count != 0:
            return False

    return True


def _save_outputs(
    cfg: DictConfig,
    selected_indices: list[int],
    source_files: list[str],
    data_stats: dict,
    total: int,
) -> None:
    """Save all output files."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    np.save(os.path.join(cfg.output_dir, "ids.npy"), np.array(selected_indices, dtype=np.int64))

    config_file = os.path.join(cfg.output_dir, "config.yaml")
    with open(config_file, "w") as f:
        logger.info(f"Saving config file to {config_file}.")
        f.write(OmegaConf.to_yaml(cfg, resolve=False))

    # Expand each molecule directory into one entry per configured filename.
    filenames = list(cfg.remote_files)
    source_files = [f"{d}/{name}" for d in source_files for name in filenames]
    destination_files = [_flatten_source(s) for s in source_files]

    chunk_size = cfg.globus_chunk_size
    chunk_dir = Path(cfg.output_dir) / "source_files_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)
    for chunk_start in range(0, len(source_files), chunk_size):
        chunk_file = chunk_dir / f"source_files_{chunk_start // chunk_size:06d}.txt"
        with open(chunk_file, "w") as f:
            for src, dst in zip(
                source_files[chunk_start : chunk_start + chunk_size],
                destination_files[chunk_start : chunk_start + chunk_size],
            ):
                f.write(f"{shlex.quote(str(src))} {shlex.quote(dst)}\n")

    data_stats_file = os.path.join(cfg.output_dir, "data_stats.pkl")
    with open(data_stats_file, "wb") as f:
        pickle.dump(data_stats, f)

    atomic_numbers_file = os.path.join(cfg.output_dir, "atomic_numbers.npy")
    unique_atomic_numbers = data_stats["elements"].keys()
    np.save(
        atomic_numbers_file,
        np.array(sorted(unique_atomic_numbers), dtype=np.int64),
    )
    logger.info(f"Selected molecules: {len(selected_indices)} / {total}")
    logger.info(f"Saved Globus source batch chunks to {chunk_dir}")
    logger.info(f"Saved outputs to {cfg.output_dir}")


def generate_omol_dataset_with_cache(cfg: DictConfig) -> None:
    """Filter OMol dataset using precomputed metadata cache.

    This is much faster than iterating through all molecules, as filtering
    happens in-memory on numpy arrays. Only selected molecules are loaded
    from the dataset (for xyz writing).

    Args:
        cfg: Hydra configuration object.
    """
    subset_name = Path(cfg.output_dir).name
    assert subset_name.startswith(
        cfg.dataset.name
    ), f"Output directory name must start with '{cfg.dataset.name}'"
    if os.path.exists(cfg.output_dir) and not cfg.overwrite:
        raise FileExistsError(
            f"Output directory {cfg.output_dir} already exists. Set overwrite=True to overwrite."
        )

    # Load and filter using cache
    cache = _load_metadata_cache(cfg.metadata_cache)
    total = len(cache["charges"])
    logger.info("Filtering using cached metadata...")
    selected_local_indices, data_stats = _filter_with_cache(cfg, cache)
    logger.info(f"Total selected molecules: {len(selected_local_indices)}")

    # Get global indices and source files from cache
    selected_indices = cache["global_indices"][selected_local_indices].tolist()
    source_files = cache["sources"][selected_local_indices].tolist()

    _save_outputs(cfg, selected_indices, source_files, data_stats, total)

    # Write xyz files if requested
    if cfg.write_xyz:
        logger.info("Writing xyz files...")
        xyz_dir = cfg.dataset.raw_data_dir
        os.makedirs(xyz_dir, exist_ok=True)

        # Load dataset only for xyz writing
        dataset = AseDBDataset(config=dict(src=cfg.dataset_src, a2g_args=dict(task_name="omol")))

        pbar = tqdm(
            zip(selected_local_indices, selected_indices),
            total=len(selected_indices),
            desc="Writing xyz files",
        )
        for local_idx, global_idx in pbar:
            filename = os.path.join(xyz_dir, f"omol_{global_idx:012d}.xyz")
            if os.path.exists(filename):
                continue

            atoms = dataset.get_atoms(local_idx)
            mol = dataset[local_idx]

            positions_np = mol.pos.numpy()
            atomic_numbers_np = mol.atomic_numbers.numpy()
            charge_positions = list(zip(atomic_numbers_np, positions_np))
            spin = atoms.info.get("spin")
            charge = atoms.info.get("charge")
            pyscf_mol = gto.M(
                atom=charge_positions,
                unit="Angstrom",
                basis="6-31G(2df,p)",
                output=None,
                spin=int(spin) - 1,
                charge=charge,
            )
            xyz_string = xyz_string_with_charge_and_multiplicity(pyscf_mol, charge, spin)
            with open(filename, "w") as f:
                f.write(xyz_string)


def generate_omol_dataset(cfg: DictConfig):
    """Scan dataset, apply filters, and save results.

    The function iterates over the dataset, applies the
    selection criteria via ``_molecule_allowed`` and collects metadata for the
    selected molecules.

    Args:
        cfg: Hydra configuration object.
    """
    subset_name = Path(cfg.output_dir).name
    assert subset_name.startswith(
        cfg.dataset.name
    ), f"Output directory name must start with '{cfg.dataset.name}'"
    if os.path.exists(cfg.output_dir) and not cfg.overwrite:
        raise FileExistsError(
            f"Output directory {cfg.output_dir} already exists. Set overwrite=True to overwrite."
        )

    dataset = AseDBDataset(config=dict(src=cfg.dataset_src, a2g_args=dict(task_name="omol")))
    total = len(dataset)

    if cfg.omol_split == "train_4M":
        convert_idx = pickle.load(open(Path(cfg.dataset_src).parent / "4M_id_mapping.pkl", "rb"))
    elif cfg.omol_split == "train":
        convert_idx = None
    else:
        raise ValueError(f"Unknown or unsupported omol_split: {cfg.omol_split}")

    selected_indices = []
    source_files = []
    data_stats = {
        "multiplicities": defaultdict(int),
        "charges": defaultdict(int),
        "data_ids": defaultdict(int),
        "elements": defaultdict(int),
        "n_heavy_atoms": defaultdict(int),
        "n_atoms": defaultdict(int),
    }
    selected_count = 0

    xyz_dir = cfg.dataset.raw_data_dir
    if cfg.write_xyz:
        os.makedirs(xyz_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    indices = rng.permutation(total)
    pbar = tqdm(indices, total=total)
    try:
        for idx in pbar:
            atoms = dataset.get_atoms(int(idx))
            mol = dataset[int(idx)]

            # Convert idx to global OMol identifier
            if convert_idx is not None:
                idx = convert_idx[idx]

            if not _molecule_allowed(cfg, atoms.info, mol, data_stats):
                continue

            # accept molecule
            selected_indices.append(int(idx))
            selected_count += 1
            # Update data info
            mult = atoms.info.get("spin")
            charge = atoms.info.get("charge")
            data_id = atoms.info.get("data_id")
            data_stats["multiplicities"][mult] += 1
            data_stats["charges"][charge] += 1
            data_stats["data_ids"][data_id] += 1
            n_heavy_atoms = int((mol.atomic_numbers.numpy() > 1).sum())
            data_stats["n_heavy_atoms"][n_heavy_atoms] += 1
            n_atoms = len(mol.atomic_numbers)
            data_stats["n_atoms"][n_atoms] += 1
            for Z in mol.atomic_numbers.numpy().tolist():
                data_stats["elements"][int(Z)] += 1

            source_files.append(os.path.dirname(atoms.info.get("source", "")))

            # write xyz if requested
            if cfg.write_xyz:
                filename = os.path.join(xyz_dir, f"omol_{int(idx):012d}.xyz")
                if not os.path.exists(filename):  # avoid overwriting existing files
                    positions_np = mol.pos.numpy()
                    atomic_numbers_np = mol.atomic_numbers.numpy()
                    charge_positions = list(zip(atomic_numbers_np, positions_np))
                    spin = atoms.info.get("spin")
                    charge = atoms.info.get("charge")
                    pyscf_mol = gto.M(
                        atom=charge_positions,
                        unit="Angstrom",
                        basis="6-31G(2df,p)",
                        output=None,
                        spin=int(spin) - 1,
                        charge=charge,
                    )
                    xyz_string = xyz_string_with_charge_and_multiplicity(pyscf_mol, charge, spin)
                    with open(filename, "w") as f:
                        f.write(xyz_string)

            pbar.set_postfix(suitable_mols=selected_count)
            if cfg.max_num_mols is not None and selected_count >= cfg.max_num_mols:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user, saving progress.")
    finally:
        try:
            pbar.close()
        except Exception:
            pass

        _save_outputs(cfg, selected_indices, source_files, data_stats, total)


@hydra.main(
    version_base="1.3",
    config_path="../../configs/datagen",
    config_name="omol_dataset.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the OMol dataset generation.

    Args:
        cfg: Hydra configuration object.
    """
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""), format=logger_format, colorize=True, enqueue=True
    )
    log_file = cfg.get("log_file")
    if log_file is not None:
        log_file = Path(log_file)
        logger.add(
            log_file, level="TRACE", rotation="10 MB", enqueue=True, backtrace=True, diagnose=True
        )
        logger.info(f'Logging to "{log_file}"')

    extras(cfg)

    # Use cached filtering if metadata_cache is specified
    if cfg.get("metadata_cache") is not None and Path(cfg.metadata_cache).exists():
        logger.info("Using cached metadata for fast filtering")
        generate_omol_dataset_with_cache(cfg)
    else:
        if cfg.get("metadata_cache") is not None:
            logger.warning(
                f"Metadata cache not found at {cfg.metadata_cache}, falling back to slow path. "
                "Run precompute_omol_metadata.py first to create the cache."
            )
        generate_omol_dataset(cfg)


if __name__ == "__main__":
    main()
