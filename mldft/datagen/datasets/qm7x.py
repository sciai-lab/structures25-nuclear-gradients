"""QM7x dataset.

Contains ~4.2 Mio molecules from the QM7X dataset. The ids of the
molecules are given by the hash of the xyz file name.

The dataset is split into 8 subsets:

- 1000.xz
- 2000.xz
- 3000.xz
- 4000.xz
- 5000.xz
- 6000.xz
- 7000.xz
- 8000.xz
"""

import multiprocessing
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.download import download_file, extract_tar
from mldft.utils.molecules import read_xyz_file

subsets_to_counter = {
    "1000": 0,
    "2000": 1000000,
    "3000": 2000000,
    "4000": 3000000,
    "5000": 4000000,
    "6000": 5000000,
    "7000": 6000000,
    "8000": 7000000,
}


class QM7XFull(DataGenDataset):
    """Class for the full QM7X dataset. Contains ~4.2 Mio molecules/geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = [
        "1000.xz",
        "2000.xz",
        "3000.xz",
        "4000.xz",
        "5000.xz",
        "6000.xz",
        "7000.xz",
        "8000.xz",
    ]

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QM7X",
        num_processes: int = 1,
        allowed_atomic_numbers: tuple[int, ...] = (1, 6, 7, 8),
    ):
        """Initialize the QM7x dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
            allowed_atomic_numbers: Tuple of allowed atomic numbers, molecules with other elements will be filtered out.

        Raises:
            AssertionError: If the subset is not in the list of available subsets.
        """
        self.allowed_atomic_numbers = np.asarray(allowed_atomic_numbers)
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.filename = filename.split(".")[0]
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """Download the raw data."""
        logger.info("Downloading QM7x")
        base_url = "https://zenodo.org/api/records/4288677/files/"
        for subset in self.subsets:
            downloaded_file = download_file(
                base_url + subset + "/content", self.raw_data_dir, filename=subset
            )
            logger.info(f"Extracting {subset}")
            extract_tar(downloaded_file, self.raw_data_dir, mode="r")
            downloaded_file.unlink()
        self.convert_xyz_files()

    def convert_xyz_files(self) -> None:
        """Convert the hdf5 files from QM7X to have xyz files."""
        logger.info(f"Converting xyz files to {self.raw_data_dir}")
        convert_folder_sorted_parallel(
            self.raw_data_dir,
            self.raw_data_dir,
            self.num_processes,
            self.allowed_atomic_numbers,
        )

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return len(list(self.raw_data_dir.glob("*.xyz")))

    def get_all_atomic_numbers(self) -> np.ndarray:
        return np.array([1, 6, 7, 8])

    def load_molecular_properties(self, ids: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Load nuclear charges and positions for the given molecule indices from the .xyz files.
        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (A).
            np.ndarray: Array of atomic positions (A, 3).
        """
        # We iterate over this list of files often, but it's still negligible compared to the kohn-sham time
        file_name = self.raw_data_dir / f"{ids:07d}.xyz"
        charges, positions, charge, multiplicity = read_xyz_file(file_name)
        return charges, positions, charge, multiplicity

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        return np.sort(np.array([int(f.stem) for f in self.raw_data_dir.glob("*.xyz")]))


class QM7XFirstBatch(QM7XFull):
    """A subset of the QM7X dataset containing the first batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["1000.xz"]


class QM7XSecondBatch(QM7XFull):
    """A subset of the QM7X dataset containing the second batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["2000.xz"]


class QM7XThirdBatch(QM7XFull):
    """A subset of the QM7X dataset containing the third batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["3000.xz"]


class QM7XFourthBatch(QM7XFull):
    """A subset of the QM7X dataset containing the fourth batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["4000.xz"]


class QM7XFifthBatch(QM7XFull):
    """A subset of the QM7X dataset containing the fifth batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["5000.xz"]


class QM7XSixthBatch(QM7XFull):
    """A subset of the QM7X dataset containing the sixth batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["6000.xz"]


class QM7XSeventhBatch(QM7XFull):
    """A subset of the QM7X dataset containing the seventh batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["7000.xz"]


class QM7XEightBatch(QM7XFull):
    """A subset of the QM7X dataset containing the eight batch of ca. 340k geometries.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        subset: Subset of the QM7X dataset.
    """

    subsets = ["8000.xz"]


def create_xyz_from_h5(
    h5_file: Path,
    out_folder: Path,
    conformer_id: str,
    sample_id: int,
    allowed_atomic_numbers=None,
) -> None:
    """Create an xyz file from a hdf5 file from the QM7X dataset.

    Args:
        h5_file: Path to the hdf5 file.
        out_folder: Path to the output folder.
        conformer_id: Id of the conformer.
    """
    if allowed_atomic_numbers is None:
        # if not specified, allow all atomic numbers up to 117 (Tenness) should be enough
        # as these elements are not even stable enough to do actual chemistry with
        allowed_atomic_numbers = np.arange(1, 118)
    with h5py.File(h5_file, "r") as f:
        conformer = f[conformer_id]
        positions = conformer["atXYZ"][:]
        charges = conformer["atNUM"][:]

    # Create the xyz file
    if np.all(np.isin(charges, allowed_atomic_numbers)):
        xyz_file = out_folder / f"{sample_id:07d}.xyz"
        with open(xyz_file, "w") as f:
            f.write(f"{len(charges)}\n")
            f.write("\n")
            for charge, position in zip(charges, positions):
                f.write(f"{charge} {position[0]} {position[1]} {position[2]}\n")
    else:
        not_allowed = np.unique(charges[~np.isin(charges, allowed_atomic_numbers)])
        logger.info(
            f"Conformer {conformer_id} contains elements {not_allowed} not in the allowed_atomic_numbers list {allowed_atomic_numbers}"
        )


def get_all_conformers_names(h5_file: Path) -> list:
    """Get all conformer names from a hdf5 file.

    Args:
        h5_file: Path to the hdf5 file.

    Returns:
        list: List of conformer names.
    """
    conformer_ids = []
    with h5py.File(h5_file, "r") as f:
        for key in f.keys():
            for subkey in f[key].keys():
                conformer_ids.append(key + "/" + subkey)
    return conformer_ids


def convert_folder_sorted_parallel(
    in_folder: Path,
    out_folder: Path,
    num_processes: int,
    allowed_atomic_numbers=None,
) -> None:
    """Apply the conversion function to all xyz files in the folder in parallel.

    Args:
        in_folder: Path to the input folder
        out_folder: Path to the output folder
        num_processes: Number of processes to use
    """
    # Get the xyz files in the folder, sort and combine with the output folder
    out_folder.mkdir(exist_ok=True)
    out_folder.chmod(0o770)
    h5_files = sorted(in_folder.glob("*.hdf5"))
    logger.info(f"Converting {len(h5_files)} files")

    # Multiprocess for faster conversion
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(num_processes) as pool:
        conformer_ids_per_h5_file = pool.map(get_all_conformers_names, h5_files)

    logger.info(f"Converting {sum(len(ids) for ids in conformer_ids_per_h5_file)} conformers")

    for h5_file, conformer_ids in zip(h5_files, conformer_ids_per_h5_file):
        counter = subsets_to_counter[h5_file.stem]
        with multiprocessing.Pool(num_processes) as pool:
            pool.starmap(
                create_xyz_from_h5,
                [
                    (
                        h5_file,
                        out_folder,
                        conformer_id,
                        sample_id,
                        allowed_atomic_numbers,
                    )
                    for sample_id, conformer_id in enumerate(conformer_ids, start=counter)
                ],
            )
