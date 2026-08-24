"""MD17 dataset.

Defines a subset of 100.000 molecules from the MD17 dataset.
"""

import numpy as np
from numpy.random import default_rng

from mldft.datagen.datasets.dataset import DataGenDataset
from mldft.utils.download import download_file


class MD17(DataGenDataset):
    """Class for the MD17 dataset.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the raw data directory.
        kohn_sham_data_dir: Path to the kohn-sham data directory.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "MD17",
        num_processes: int = 1,
    ):
        """Initialize the dataset by setting attributes.

        Args:
            raw_data_dir: Path to the directory containing the raw data.
            kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
            label_dir: Path to the directory containing the labels.
            filename: The filename to use for the output files.
            name: Name of the dataset, used as the folder name.
            num_processes: Number of processes to use for dataset verifying or loading.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        npz_file = np.load(self.raw_data_dir / "md17_ethanol.npz")
        self.charges = npz_file["z"]
        self.positions = npz_file["R"]
        self.num_molecules = self.get_num_molecules()

    def download(self) -> None:
        """Download the dataset from the source."""
        download_file(
            "http://www.quantum-machine.org/gdml/data/npz/md17_ethanol.npz",
            self.raw_data_dir,
        )

    def get_all_atomic_numbers(self) -> np.ndarray:
        return np.array([1, 6, 8])

    def get_num_molecules(self) -> int:
        """Get the number of molecules in the dataset.

        Returns:
            int: Number of molecules in the dataset.
        """
        return len(self.positions)

    def load_charges_and_positions(self, id: int) -> tuple[np.ndarray, np.ndarray]:
        """Load nuclear charges and positions from a .npz file.

        Args:
            ids: Array of indices of the molecules to compute.

        Returns:
            np.ndarray: Array of atomic numbers (N, A).
            np.ndarray: Array of atomic positions (N, A, 3).
        """
        positions = self.positions[id]
        return self.charges, positions

    def load_molecular_properties(self, id: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Load charges/positions plus molecular charge and multiplicity."""
        charges, positions = self.load_charges_and_positions(id)
        return charges, positions, 0, 1

    def get_ids(self) -> np.ndarray:
        """Get the indices of the molecules to compute.

        Returns:
            np.ndarray: Array of indices of the molecules to compute.
        """
        rng = default_rng(8)
        ids = np.sort(rng.choice(self.num_molecules, size=100000, replace=False))
        return ids
