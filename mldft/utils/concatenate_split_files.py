import argparse
import pickle
from pathlib import Path

from loguru import logger

from mldft.utils.create_dataset_splits import check_paths_and_save

SPLIT_KEYS = ["train", "val", "test"]


def concatenate_split_dicts(list_of_files: list) -> dict:
    """Concatenate multiple split dictionaries into one.

    Args:
        list_of_files (list): List of paths to split files to concatenate.

    Returns:
        A dictionary containing concatenated splits and their sizes.
    """
    concatenated_split = {key: [] for key in SPLIT_KEYS}
    concatenated_split["sizes"] = {key: 0 for key in SPLIT_KEYS}

    for file in list_of_files:
        logger.info(f"Concatenating: {file}")

        with open(file, "rb") as pklfile:
            split_dict = pickle.load(pklfile)

        for key in SPLIT_KEYS:
            concatenated_split["sizes"][key] += split_dict["sizes"][key]
            concatenated_split[key] += split_dict[key]

    return concatenated_split


def concatenate_and_save(list_of_files: list, outpath: Path, override=False):
    """Concatenate multiple split files into one and save it.

    Args:
        list_of_files (list): List of paths to split files to concatenate.
        outpath (Path): Output path for the concatenated split file.
        override (bool): If True, overrides existing output files.
    """
    outdict = concatenate_split_dicts(list_of_files)

    outpath_yaml = outpath.with_suffix(".yaml")
    outpath_pkl = outpath.with_suffix(".pkl")

    check_paths_and_save(outdict, outpath_yaml, outpath_pkl, override=override)


def main():
    """Main function to parse arguments and call the concatenate function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", type=str, nargs="+", help="split pickle files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Name of the output file",
        default=Path("split.pkl"),
    )
    parser.add_argument("--override", action="store_true", help="Override existing split file.")

    args = parser.parse_args()

    concatenate_and_save(args.pickle_files, args.output, override=args.override)


if __name__ == "__main__":
    main()
