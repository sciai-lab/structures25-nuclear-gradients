import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from mldft.utils.create_dataset_splits import (
    _create_train_val_test_split_dict,
    _iteration_count,
    check_paths_and_save,
    get_path_list_for_split_file,
)
from mldft.utils.environ import get_mldft_data_path


def qmugs_dataset_to_csv(name: str):
    dataset_name = name.split("_")[0]
    return f"{dataset_name}.csv"


def get_qmugs_paths_and_group_ids(
    qmugs_path: Path, csv_file: Path
) -> tuple[list[Path], list[int]]:
    """Determine paths of QMUGS label files based on the id given in the csv file."""
    # To determine which QMUGS files belong to this dataset, we need to read the ids from the csv file
    assert csv_file.exists(), f"CSV file {csv_file} does not exist."
    dataframe = pl.read_csv(csv_file)
    ids = dataframe["id"]
    paths = []
    group_ids = []
    label_dir = qmugs_path / "labels"
    # Ensure deterministic ordering across filesystems/runs.
    label_files = sorted(label_dir.glob("*.zarr*"))
    for label in tqdm(label_files, desc="Identifying qmugs paths and chembl ids"):
        if int(label.name.split(".")[0]) in ids:
            paths.append(label)
            group_ids.append(int(label.name.split(".")[0]) // 3)
    return paths, group_ids


def create_split_file_qmugs_large_bins(dataset: str, override: bool):
    """Create a split file containing only validation molecules for the QMUGSLargeBins dataset.

    Train and test splits are empty.
    """
    logger.info(f"Creating QMUGS large bins split file for {dataset}.")
    data_dir = get_mldft_data_path()
    qmugs_base_path = data_dir / "QMUGS"
    qmugs_large_bins_path = data_dir / "QMUGSLargeBins"
    # To determine which QMUGS files belong to this dataset, we need to read the ids from the csv file
    csv_file = qmugs_base_path / qmugs_dataset_to_csv(dataset)
    paths, groups = get_qmugs_paths_and_group_ids(qmugs_base_path, csv_file)
    iterations = []
    for path in tqdm(paths, desc="Creating Dataset, counting scf iterations per path"):
        iterations.append(_iteration_count(path))
    iterations = np.array(iterations)
    paths_test = get_path_list_for_split_file(paths, iterations)
    test_size = int(sum(iterations))
    yaml_dict = {
        "sizes": {"train": 0, "val": 0, "test": test_size},
        "train": [],
        "val": [],
        "test": paths_test,
    }
    yaml_path = qmugs_large_bins_path / "split.yaml"
    pickle_path = qmugs_large_bins_path / "split.pkl"
    check_paths_and_save(yaml_dict, yaml_path, pickle_path, override)


def create_split_file_qmugs_bin(dataset: str, override: bool):
    """Create a split 90% train, 10% validation split for a bin of QMUGS.

    The test set is set to the large QMUGS bins if available. Every molecule which is randomly
    sampled into the train set but is also in the test set, is moved to the validation set.
    """
    logger.info(f"Creating QMUGS bin split file for {dataset}.")
    data_dir = get_mldft_data_path()
    if "perturbed_fock" in dataset:
        qmugs_base_path = data_dir / "QMUGS_perturbed_fock"
    else:
        qmugs_base_path = data_dir / "QMUGS"
    qmugs_csv_dir = data_dir / "QMUGS"
    csv_file = qmugs_csv_dir / qmugs_dataset_to_csv(dataset)
    logger.info(f"Reading molecule ids from csv file {csv_file}.")
    paths, group_ids = get_qmugs_paths_and_group_ids(qmugs_base_path, csv_file)

    large_bin_split_file = data_dir / "QMUGSLargeBins" / "split.pkl"
    # Check if files are already in large bins split and move them from train to val
    large_bin_paths_set = set()
    if large_bin_split_file.exists():
        with open(large_bin_split_file, "rb") as f:
            large_bin_splits = pickle.load(f)
        # Create a set of paths that are in the large bins split for fast lookup
        for split_paths in large_bin_splits["test"]:
            large_bin_paths_set.add(Path(split_paths[1]))

    qmugs_bin_splits = _create_train_val_test_split_dict(
        paths, group_ids, (0.9, 0.1, 0), processes=1
    )

    # Move any files from train to val if they're in large bins
    if large_bin_paths_set:
        train_to_move = []
        for item in qmugs_bin_splits["train"]:
            if Path(item[1]) in large_bin_paths_set:
                train_to_move.append(item)

        for item in train_to_move:
            qmugs_bin_splits["train"].remove(item)
            qmugs_bin_splits["val"].append(item)
            qmugs_bin_splits["sizes"]["train"] -= item[2]
            qmugs_bin_splits["sizes"]["val"] += item[2]

        if train_to_move:
            logger.info(
                f"Moved {len(train_to_move)} files from train to val (already in large bins)."
            )

    # If the large mol split file exists, use it for the test split
    if large_bin_split_file.exists():
        qmugs_bin_splits["test"] = large_bin_splits["test"]
        qmugs_bin_splits["sizes"]["test"] = large_bin_splits["sizes"]["test"]
    else:
        logger.warning(
            f"Large bin split file {large_bin_split_file} does not exist. Test split will be empty."
        )
    yaml_path = data_dir / dataset / "split.yaml"
    pickle_path = data_dir / dataset / "split.pkl"
    check_paths_and_save(qmugs_bin_splits, yaml_path, pickle_path, override)


def create_split_file_qmugs_bin0_qm9(dataset: str, override: bool):
    """Create a split file that combines QMUGS bin0 with QM9.

    - Train split: `qm9.train`, `qm9.val`, and `qmugs_bin0.train`.
    - Validation split: `qm9.test` together with `qmugs_bin0.val` (QM9 test is assigned to validation).
    - Test split: `qmugs_large_bins.test`.
    """
    logger.info(f"Creating QMUGS and QM9 merged split file for {dataset}.")
    data_dir = get_mldft_data_path()
    with open(data_dir / "QMUGSLargeBins" / "split.pkl", "rb") as f:
        qmugs_large_bins_splits = pickle.load(f)
    if "perturbed_fock" in dataset:
        with open(data_dir / "QMUGSBin0_perturbed_fock" / "split.pkl", "rb") as f:
            qmugs_bin0_splits = pickle.load(f)
        with open(data_dir / "QM9_perturbed_fock" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0QM9_perturbed_fock"
    else:
        with open(data_dir / "QMUGSBin0" / "split.pkl", "rb") as f:
            qmugs_bin0_splits = pickle.load(f)
        with open(data_dir / "QM9" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0QM9"

    # Merge splits
    merged_splits = qmugs_bin0_splits.copy()
    merged_splits["train"] += qm9_splits["train"] + qm9_splits["val"]
    merged_splits["val"] += qm9_splits["test"]
    merged_splits["sizes"]["train"] += qm9_splits["sizes"]["train"] + qm9_splits["sizes"]["val"]
    merged_splits["sizes"]["val"] += qm9_splits["sizes"]["test"]
    merged_splits["test"] = qmugs_large_bins_splits["test"]
    merged_splits["sizes"]["test"] = qmugs_large_bins_splits["sizes"]["test"]
    yaml_path = merged_path / "split.yaml"
    pickle_path = merged_path / "split.pkl"
    check_paths_and_save(merged_splits, yaml_path, pickle_path, override)


def create_split_file_qmugs_bin0_bin1_qm9(dataset: str, override: bool):
    """Create a split file that merges QMUGS bins 0/1 with QM9.

    - Train split: `qm9.train`, `qm9.val`, `qmugs_bin0.train`, and `qmugs_bin1.train` (up to the generated ~40k entries).
    - Validation split: `qm9.test`, `qmugs_bin0.val`, and `qmugs_bin1.val`. (QM9 test is deliberately treated as validation.)
    - Test split: `qmugs_large_bins.test`.
    """
    logger.info(f"Creating QMUGS Bin0, Bin 1 and QM9 merged split file for {dataset}.")
    data_dir = get_mldft_data_path()
    with open(data_dir / "QMUGSLargeBins" / "split.pkl", "rb") as f:
        qmugs_large_bins_splits = pickle.load(f)
    if "perturbed_fock" in dataset:
        with open(data_dir / "QMUGSBin0_perturbed_fock" / "split.pkl", "rb") as f:
            qmugs_bin0_splits = pickle.load(f)
        with open(data_dir / "QMUGSBin1_perturbed_fock" / "split.pkl", "rb") as f:
            qmugs_bin1_splits = pickle.load(f)
        with open(data_dir / "QM9_perturbed_fock" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0Bin1QM9_perturbed_fock"
    else:
        with open(data_dir / "QMUGSBin0" / "split.pkl", "rb") as f:
            qmugs_bin0_splits = pickle.load(f)
        with open(data_dir / "QM9" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0Bin1QM9"

    # Merge splits
    merged_splits = qmugs_bin0_splits.copy()
    merged_splits["train"] += qmugs_bin1_splits["train"] + qm9_splits["train"] + qm9_splits["val"]
    merged_splits["val"] += qmugs_bin1_splits["val"] + qm9_splits["test"]
    merged_splits["sizes"]["train"] += (
        qmugs_bin1_splits["sizes"]["train"]
        + qm9_splits["sizes"]["train"]
        + qm9_splits["sizes"]["val"]
    )
    merged_splits["sizes"]["val"] += (
        qmugs_bin1_splits["sizes"]["val"] + qm9_splits["sizes"]["test"]
    )
    merged_splits["test"] = qmugs_large_bins_splits["test"]
    merged_splits["sizes"]["test"] = qmugs_large_bins_splits["sizes"]["test"]
    yaml_path = merged_path / "split.yaml"
    pickle_path = merged_path / "split.pkl"
    check_paths_and_save(merged_splits, yaml_path, pickle_path, override)


def create_omol_split_file(
    dataset: str,
    subset: str,
    yaml_path: str | Path | None,
    pickle_path: str | Path | None,
    override: bool,
    split_percentages: tuple[float, float, float],
    processes: int,
    all_test: bool = False,
    ids_path: str | Path | None = None,
):
    """Create a split file for an OMOL subset."""
    label_dir = get_mldft_data_path() / dataset / "labels"
    assert label_dir.exists(), f"Label directory {label_dir} does not exist."
    paths = sorted(label_dir.glob("*.zarr*"))
    if len(paths) == 0:
        if len(os.listdir(label_dir)) == 1:
            paths = sorted(label_dir.rglob("*.zarr*"))
        else:
            raise FileNotFoundError(
                f"There are no .zarr files in {label_dir} and an ambiguous amount of subfolders."
            )

    if ids_path is not None:
        ids = set(np.load(ids_path))
    elif (get_mldft_data_path() / subset / "ids.npy").exists():
        ids = set(np.load(get_mldft_data_path() / subset / "ids.npy"))
    else:
        ids = None

    # Filter paths according to ids
    if ids is not None:
        filtered_paths = []
        for i, p in enumerate(paths):
            m = re.search(r"(\d+)", p.name)
            if not m:
                logger.warning(f"Could not extract id from filename {p}; skipping.")
                continue
            file_id = int(m.group(1))
            if file_id in ids:
                filtered_paths.append(p)
        paths = filtered_paths
        logger.info(f"Filtered paths according to ids, {len(paths)} paths remaining.")

    yaml_dict = _create_train_val_test_split_dict(
        paths,
        split_percentages=split_percentages,
        processes=processes,
        all_test=all_test,
    )
    subset_dir = get_mldft_data_path() / subset
    if yaml_path is None:
        yaml_path = subset_dir / "split.yaml"
    if pickle_path is None:
        pickle_path = subset_dir / "split.pkl"
    check_paths_and_save(yaml_dict, yaml_path, pickle_path, override)


if __name__ == "__main__":
    # parse label_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument(
        "--split_percentages",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        help="Percentages of the dataset to use for train, val and test.",
    )
    parser.add_argument("--override", action="store_true", help="Override existing split file.")
    parser.add_argument("--yaml_path", type=Path, help="Path to the yaml file.")
    parser.add_argument("--pickle_path", type=Path, help="Path to the pickle file.")
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Use all data for test. split_percentages will be ignored.",
    )
    parser.add_argument("--subset", type=str, help="Name of the subset.")
    args = parser.parse_args()
    if args.dataset == "QMUGSLargeBins":
        create_split_file_qmugs_large_bins(args.dataset, args.override)
    elif args.dataset.startswith("QMUGSBin0QM9"):
        create_split_file_qmugs_bin0_qm9(args.dataset, args.override)
    elif args.dataset.startswith("QMUGSBin0Bin1QM9"):
        create_split_file_qmugs_bin0_bin1_qm9(args.dataset, args.override)
    elif args.dataset.startswith("QMUGSBin"):
        create_split_file_qmugs_bin(args.dataset, args.override)
    elif args.dataset.startswith("OMol"):
        create_omol_split_file(
            dataset=args.dataset,
            subset=args.subset,
            yaml_path=args.yaml_path,
            pickle_path=args.pickle_path,
            override=args.override,
            split_percentages=args.split_percentages,
            processes=1,
            all_test=args.all_test,
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
