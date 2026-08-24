"""DataGenDataset for subsets of the OMol25 dataset."""

from pathlib import Path

import numpy as np
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.molecules import read_xyz_file


class OMol(DataGenDataset):
    """OMol class. Run 'generate_omol_dataset' before datagen to filter the complete OMol25 dataset
    resulting in the desired subset.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        xyz_files: Dictionary mapping molecule IDs to `.xyz` file paths.
        ids: Numpy array of integer ids loaded from `ids.npy` in `subset_dir`.
        atomic_numbers: Numpy array of unique atomic numbers for the subset.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        subset: str,
        subset_dir: str,
        name: str = "OMol",
        num_processes: int = 1,
    ):
        """Initialize the OMol dataset.

        Args:
            raw_data_dir: Path to the raw data directory containing `.xyz` files.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing labels.
            filename: The base filename to use for output files (extension is
                stripped).
            subset: Name of the subset.
            subset_dir: Directory that must contain `ids.npy` and
                `atomic_numbers.npy` for this subset.
            name: Optional dataset name (default: "OMol").
            num_processes: Number of processes to use for dataset verifying or
                loading.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        self.filename = filename.split(".")[0]

        # Load ids
        ids_path = Path(subset_dir) / "ids.npy"
        self.ids = np.load(ids_path)
        self.num_molecules = len(self.ids)

        atomic_numbers_file = Path(subset_dir) / "atomic_numbers.npy"
        if not atomic_numbers_file.is_file():
            raise FileNotFoundError(f"Atomic numbers file not found at {atomic_numbers_file}. ")
        self.atomic_numbers = np.load(atomic_numbers_file)

        # Collect .xyz files and assert that all files are present
        self.xyz_files = {}
        for id in self.ids:
            file_path = Path(raw_data_dir) / f"omol_{id:012d}.xyz"
            if not file_path.is_file():
                raise FileNotFoundError(
                    f".xyz file for molecule ID {id} not found at {file_path}."
                )
            self.xyz_files[id] = file_path

    def download(self) -> None:
        """This is just a stub as this kind of dataset can not be downloaded."""
        logger.info("No download required/possible for the OMol dataset.")

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return self.num_molecules

    def get_all_atomic_numbers(self) -> np.ndarray:
        """Get all atomic numbers present in the dataset.

        Returns:
            np.ndarray: Array of atomic numbers present in the dataset.
        """
        return self.atomic_numbers

    def load_molecular_properties(self, id: int) -> tuple[list, list, int, int]:
        """Load nuclear charges and positions for the given molecule index from the .xyz files.

        Args:
            id: Index of the molecule to load.

        Returns:
            tuple: (charges, positions, mol_charge, multiplicity) where charges is a list of atomic numbers,
                   positions is a list of atomic coordinates (A, 3), mol_charge is the molecular charge,
                   multiplicity is the spin multiplicity.
        """
        file_name = self.xyz_files[id]
        charges, positions, mol_charge, multiplicity = read_xyz_file(file_name)
        return charges, positions, mol_charge, multiplicity

    def load_mol_info(self, id: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Load molecular information for the given molecule index from the .xyz file (including
        charge and multiplicity).

        Args:
            id: Index of the molecule to load.
        Returns:
            tuple: (charges, positions, charge, multiplicity) where charges is a numpy array of atomic numbers,
                   positions is a numpy array of atomic coordinates (A, 3), charge is the net charge of the molecule,
                   and multiplicity is the spin multiplicity.
        """

        file_name = self.xyz_files[id]
        charges, positions, charge, multiplicity = read_xyz_file(
            file_name, return_charge_and_multiplicity=True
        )
        return charges, positions, charge, multiplicity

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        return self.ids
