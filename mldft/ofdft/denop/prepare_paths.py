import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import ListConfig

from mldft.utils.continue_denop import (
    get_existing_sample_paths,
    get_missing_paths_and_scf_iters,
)
from mldft.utils.environ import get_mldft_data_path

# TODO: I think there is potential here to simplify the molecule selection
# the stuff with start_index and n_molecules can be confusing and might be
# more hassle than it's worth?
# maybe it would be better to make all this stuff a class?


def construct_label_path(data_dir: Path, dataset: str, label_path: str) -> Path:
    """Build absolute path to one molecule label archive.

    Args:
        data_dir (Path): Root MLDFT data directory.
        dataset (str): Dataset split namespace (for example ``qm9``).
        label_path (str): Relative label filename from split metadata.

    Returns:
        Path: Full filesystem path to the zipped zarr label file.
    """
    # Side effects:
    # - None. This helper only composes and returns a filesystem path.

    return data_dir / dataset / "labels" / label_path


def get_paths_from_split(
    split_dict: dict, split_name: str, data_dir: Path
) -> tuple[list[Path], list[int]]:
    """Extract molecule paths and SCF iteration counts for one split.

    Args:
        split_dict (dict): Loaded split-file dictionary.
        split_name (str): Split key (for example ``train``/``val``/``test``).
        data_dir (Path): Root MLDFT data directory.

    Returns:
        tuple[list[Path], list[int]]: Tuple of label paths and per-path SCF iteration
        counts.
    """
    # Side effects:
    # - None. This helper only extracts and returns path/iteration lists.

    split_files = split_dict[split_name]
    paths = [
        construct_label_path(data_dir, dataset, label_path)
        for dataset, label_path, _ in split_files
    ]
    scf_iterations = [scf_iterations for _, _, scf_iterations in split_files]
    return paths, scf_iterations


def find_molecule_path_by_id(
    molecule_id: int, split_dict: dict, data_dir: Path, user_split: str
) -> tuple[Path, int] | None:
    """Find molecule label path by integer molecule id across splits.

    Args:
        molecule_id (int): Molecule id encoded in filename suffix.
        split_dict (dict): Loaded split-file dictionary.
        data_dir (Path): Root MLDFT data directory.
        user_split (str): Preferred split to search first.

    Returns:
        tuple[Path, int] | None: ``(path, scf_iterations)`` if found, else ``None``.
    """
    # Side effects:
    # - None. This helper performs lookup only.

    # TODO: I think this can be made faster by changing the lookup pattern
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


def configure_dataset_indices_from_string(
    dataset_size: int,
    n_molecules: int,
    molecule_choice: str,
    seed: int | None = None,
    start_idx: int = 0,
) -> np.ndarray:
    """Create dataset indices from a string selection strategy.

    Args:
        dataset_size (int): Number of samples available in the selected split.
        n_molecules (int): Number of molecules requested.
        molecule_choice (str): Selection mode (``first_n``, ``random``,
            ``seeded_random``, ``always_*``).
        seed (int | None): RNG seed when selection uses randomness.
        start_idx (int): Start offset applied after generating candidate indices.

    Raises:
        ValueError: If requested molecules exceed dataset size.
        ValueError: If ``seeded_random`` is requested without a seed.
        ValueError: If ``molecule_choice`` is unknown.

    Returns:
        np.ndarray: Selected dataset indices as a NumPy array.
    """
    # Side effects:
    # - Emits warnings for ignored/ambiguous selection settings.
    # - Uses a local NumPy RNG for random selection modes.

    if dataset_size < n_molecules:
        raise ValueError("n_molecules cannot be greater than dataset_size")

    if molecule_choice.startswith("always_"):
        index = int(molecule_choice.split("_")[1])
        dataset_indices = np.array([index] * n_molecules)
    elif molecule_choice == "first_n":
        dataset_indices = np.arange(n_molecules)[start_idx:]
    elif molecule_choice == "random":
        if start_idx != 0:
            logger.warning(
                f"Random molecule choice with start_idx {start_idx}. Ignoring start_idx."
            )
        rng = np.random.default_rng(seed)
        dataset_indices = rng.choice(dataset_size, n_molecules, replace=False)
    elif molecule_choice == "seeded_random":
        if seed is None:
            raise ValueError("seed must be provided for seeded_random molecule choice")
        rng = np.random.default_rng(seed)
        dataset_indices = rng.choice(dataset_size, n_molecules, replace=False)[
            start_idx:
        ]  # FIXME: is start index any use here? The meaning of this changes with the choice, it's essentially only a mask?
    else:
        raise ValueError(
            f"Unknown molecule choice {molecule_choice}. "
            f"Must be 'first_n', 'random', 'seeded_random'"
        )

    return dataset_indices


def configure_dataset_indices_from_list(
    n_molecules: int,
    molecule_choice: str | list | ListConfig,
    start_idx: int = 0,
) -> np.ndarray:
    """Create dataset indices from an explicit index list.

    Args:
        n_molecules (int): Number of molecules requested.
        molecule_choice (str | list | ListConfig): Explicit sequence of dataset indices.
        start_idx (int): Offset into the provided list.

    Returns:
        np.ndarray: Selected indices clipped to available entries and request size.
    """
    # Side effects:
    # - Emits warnings when requested molecule count and explicit index count differ.

    if len(molecule_choice) < n_molecules:
        logger.warning(
            f"Requested {n_molecules} molecules, but only {len(molecule_choice)} are "
            f"available. Setting n_molecules to {len(molecule_choice)}."
        )
        n_molecules = len(molecule_choice)

    if len(molecule_choice) > n_molecules:
        logger.warning(
            f"Requested {n_molecules} molecules, but {len(molecule_choice)} are available. "
            f"Using the first {n_molecules} molecules."
        )

    dataset_indices = np.array(molecule_choice[start_idx:n_molecules])

    return dataset_indices


def configure_dataset_indices(
    dataset_size: int,
    n_molecules: int | None,
    molecule_choice: str | list | ListConfig,
    seed: int | None,
    start_idx: int = 0,
) -> np.ndarray:
    """Resolve final dataset indices for denop execution.

    Args:
        dataset_size (int): Number of samples in current split.
        n_molecules (int | None): Requested number of molecules, or ``None`` for all.
        molecule_choice (str | list): Selection mode string or explicit index list.
        seed (int | None): RNG seed for random selection modes.
        start_idx (int): Offset used by index-generation helpers.

    Raises:
        ValueError: If ``n_molecules`` is negative.
        ValueError: If ``molecule_choice`` type or value is unsupported.

    Returns:
        np.ndarray: NumPy array of dataset indices to run.
    """
    # Side effects:
    # - Delegates to selection helpers that may emit warnings for invalid/ambiguous requests.

    if n_molecules is None:
        n_molecules = dataset_size

    if n_molecules < 0:
        raise ValueError("n_molecules must be non-negative")

    if isinstance(molecule_choice, str):
        dataset_indices = configure_dataset_indices_from_string(
            dataset_size, n_molecules, molecule_choice, seed, start_idx
        )
    elif isinstance(molecule_choice, (list, ListConfig)):
        dataset_indices = configure_dataset_indices_from_list(
            n_molecules,
            molecule_choice,
            start_idx,
        )
    else:
        raise ValueError(
            f"Unknown molecule choice {molecule_choice}. Must be 'first_n', 'random', 'seeded_random', or a list of indices."
        )
    return dataset_indices


def prepare_molecule_paths(
    data_dir: Path,
    split_dict: dict,
    n_molecules: int | None,
    molecule_ids: list[int] | None,
    molecule_choice: list[int] | str,
    split: str = "val",
    seed: int | None = None,
) -> tuple[list[Path], list[int]]:
    """Prepare molecule file paths and SCF-iteration metadata for a run.

    Args:
        data_dir (Path): Root MLDFT data directory.
        split_dict (dict): Loaded split-file dictionary.
        n_molecules (int | None): Number of molecules to select when ids are not
            explicitly provided.
        molecule_ids (list[int] | None): Explicit molecule ids to resolve across splits.
        molecule_choice (list[int] | str): Selection strategy or explicit index list.
        split (str): Primary split used for default selection.
        seed (int | None): RNG seed for random selection strategies.

    Raises:
        ValueError: If any requested ``molecule_id`` is missing from all splits.

    Returns:
        tuple[list[Path], list[int]]: Tuple of selected label paths and associated SCF
        iteration limits.
    """
    # Side effects:
    # - Emits info logs describing the selected molecule mode/paths.

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
        logger.info("Running density optimization for molecules on full set of dataset indices")
    return all_paths, all_num_scf_iterations_per_path


def prepare_paths(
    denop_dir: str | Path,
    n_molecules: int | None,
    molecule_ids: list[int] | None,
    molecule_choice: list[int] | str,
    continue_denop: bool,
    split_file_path: str | Path,
    split: str = "val",
    seed: int | None = None,
) -> tuple[dict, list, list]:
    """Load split metadata, select molecule paths, and filter already finished runs.

    Args:
        denop_dir (str | Path): Output directory used for continue mode checks.
        n_molecules (int | None): Number of molecules to process, or ``None`` for all.
        molecule_ids (list[int] | None): Optional explicit molecule IDs.
        molecule_choice (list[int] | str): Selection strategy when ``molecule_ids`` is
            not provided.
        continue_denop (bool): Whether to skip molecules already present in ``denop_dir``.
        split_file_path (str | Path): Path to pickled split dictionary.
        split (str): Dataset split key to sample from.
        seed (int | None): RNG seed for random selection modes.

    Returns:
        tuple[dict, list, list]: Tuple of
        ``(split_dict, paths, num_scf_iterations_per_path)``.
    """
    # Side effects:
    # - Loads split metadata from disk.
    # - Emits info logs about selection/continue-mode filtering decisions.

    data_dir = get_mldft_data_path()
    with open(split_file_path, "rb") as f:
        split_dict = pickle.load(f)

    all_paths, all_num_scf_iterations_per_path = prepare_molecule_paths(
        data_dir, split_dict, n_molecules, molecule_ids, molecule_choice, split, seed
    )

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
        else:
            logger.info(f"Found {len(paths)} missing molecules, that will be run.")
    else:
        paths = all_paths
        num_scf_iterations_per_path = all_num_scf_iterations_per_path

    return split_dict, paths, num_scf_iterations_per_path
