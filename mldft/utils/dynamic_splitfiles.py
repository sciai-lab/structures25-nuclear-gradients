import pickle
from pathlib import Path
from typing import List

import numpy as np

from mldft.utils import RankedLogger
from mldft.utils.create_dataset_splits import check_paths_and_save

log = RankedLogger(__name__, rank_zero_only=True)


def subsample_data(data_list: List, n_samples: int, method: str = "seeded_random"):
    """Returns a subset of the given data_list according to the specified subsampling method."""
    generator = np.random.default_rng(42)
    if method == "seeded_random":
        indices = generator.choice(range(len(data_list)), size=n_samples, replace=False).tolist()
        return [data_list[i] for i in indices]
    elif method == "random":
        indices = np.random.choice(range(len(data_list)), size=n_samples, replace=False).tolist()
        return [data_list[i] for i in indices]
    elif method == "first":
        return data_list[:n_samples]
    elif method == "last":
        return data_list[-n_samples:]
    else:
        raise ValueError("Invalid subsampling method")


def merge_split_dicts(dicts: List[dict]) -> dict:
    """Merges train splits for a list of split dictionaries into a single dictionary."""
    base_split = dicts[0].copy()
    # Use a set for O(1) lookup instead of O(n) list membership check
    seen = {tuple(item) if isinstance(item, list) else item for item in base_split["train"]}

    for d in dicts[1:]:  # Skip first dict since it's already in base_split
        for item in d["train"]:
            item_key = tuple(item) if isinstance(item, list) else item
            if item_key not in seen:
                base_split["train"].append(item)
                seen.add(item_key)

    base_split["sizes"]["train"] = sum([item[2] for item in base_split["train"]])
    return base_split


class Subsample:
    """Class to load a splitfile and subsample the training set."""

    def __init__(
        self,
        split_file: str | Path,
        percentage: float = 1.0,
        n_samples: int | None = None,
        method: str = "seeded_random",
    ):
        """Initializes the Subsample class.

        args:
            split_file: Path to the splitfile.
            percentage: Percentage of the training set to keep. This is ignored if n_samples is set.
            n_samples: Number of samples to keep (overrides percentage if set).
            method: Subsampling method. Options are "seeded_random", "random", "first", "last".
        """
        self.split_file = split_file
        with open(self.split_file, "rb") as f:
            self.split_dict = pickle.load(f)
        self.n_samples = n_samples
        self.train_size = len(self.split_dict["train"])
        if n_samples is None:
            self.n_samples = int(self.train_size * percentage)
        self.method = method

        if self.n_samples < self.train_size and self.n_samples > 0:
            self.split_dict["train"] = subsample_data(
                self.split_dict["train"], n_samples=self.n_samples, method=self.method
            )
        elif not self.n_samples == self.train_size:
            raise ValueError(
                "Resulting number of samples must be lower or equal to the training set size and greater than 0."
            )

        self.split_dict["sizes"]["train"] = sum([item[2] for item in self.split_dict["train"]])

    def __str__(self) -> str:
        """Info string for logging of split composition."""
        if self.n_samples == self.train_size:
            return f"{self.split_file}, contributing {self.n_samples} molecules (100.00%)"
        else:
            return f"{self.split_file}, contributing {self.n_samples} molecules ({self.n_samples / self.train_size * 100:.2f}%), subsampling method: {self.method}"


def construct_splitfile(split_files: str | Subsample | List[Subsample | str], output_file: str):
    """Constructs a splitfile from a list of subsample objects or splitfiles. Merges the training
    sets according to the specified subsampling, while val and test sets are taken from the first
    splitfile. If a single splitfile is provided, it will be copied to the output location
    unchanged.

    Args:
        split_files: List of subsample objects or splitfiles.
            Subsample objects allow to specify a number of samples or a percentage of the training set to keep and a subsampling method.
        output_file: Path to the output splitfile.
    Returns:
        Path to the output splitfile.
    """
    if isinstance(split_files, (str, Path)) or isinstance(split_files, Subsample):
        split_files = [split_files]
    if len(split_files) == 1 and isinstance(split_files[0], (str, Path)):
        log.info("Using unchanged splitfile:")
    else:
        val_split = (
            split_files[0].split_file if isinstance(split_files[0], Subsample) else split_files[0]
        )
        log.info(f"Using val and test set from {val_split}")
        log.info("Constructing the train split from the following splitfiles:")

    split_dicts = []
    for sample in split_files:
        # Check and convert types
        if isinstance(sample, (str, Path)):
            sample = Subsample(sample, 1.0)
        elif not isinstance(sample, Subsample):
            raise ValueError("split_files must be a list of str, Path or Subsample objects")

        log.info(str(sample))
        split_dicts.append(sample.split_dict)

    # merge the splitfiles
    split_dicts_merged = merge_split_dicts(split_dicts)
    check_paths_and_save(
        split_dicts_merged,
        Path(output_file.replace(".pkl", ".yaml")),
        Path(output_file),
        override=True,
    )
    return output_file
