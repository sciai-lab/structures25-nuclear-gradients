"""Arbitrary dataset of molecules in .xyz format.

This dataset is intended for small case tests of molecules that are not part of specific datasets.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.molecules import read_xyz_file


class MiscXYZ(DataGenDataset):
    """Class for arbitrary molecules in .xyz format.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
        xyz_files: Dictionary mapping molecule IDs to .xyz file paths.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "MiscXYZ",
        num_processes: int = 1,
    ):
        """Initialize the MiscXYZ dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.

        Raises:
            AssertionError: If the subset is not in the list of available subsets.
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

        # Collect .xyz files and check for duplicate IDs
        self.xyz_files = {}
        seen_ids = set()
        for f in Path(raw_data_dir).glob("*.xyz"):
            if f.is_file():
                molecule_id = int(f.stem.split("_")[1])
                if molecule_id in seen_ids:
                    raise ValueError(
                        f"Duplicate molecule ID {molecule_id} found in file: {f}. "
                        "Each molecule ID must be unique."
                    )
                self.xyz_files[molecule_id] = f
                seen_ids.add(molecule_id)

        self.num_molecules = len(self.xyz_files)

    def download(self) -> None:
        """This is just a stub as this kind of dataset can not be downloaded."""
        logger.info("No download required/possible for MiscXYZ dataset.")

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return self.num_molecules

    @lru_cache(maxsize=1)
    def get_all_atomic_numbers(self) -> np.ndarray:
        """Get all atomic numbers present in the dataset.

        Iterates over all molecules in the dataset and collects all atomic numbers.

        Returns:
            np.ndarray: Array of atomic numbers present in the dataset.
        """
        all_atomic_numbers = set()
        for id in self.get_ids():
            charges, _, _, _ = self.load_molecular_properties(id)
            all_atomic_numbers.update(charges)

        return np.array(sorted(all_atomic_numbers))

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
        atom_charges, positions, charge, multiplicity = read_xyz_file(file_name)
        return atom_charges, positions, charge, multiplicity

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules in the dataset.

        Returns:
            np.ndarray: Array of indices of the molecules in the dataset.
        """
        return np.sort(np.array(list(self.xyz_files.keys())))
