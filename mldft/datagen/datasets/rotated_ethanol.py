import copy
from pathlib import Path

import numpy as np
from e3nn.o3 import rand_matrix
from pyscf import gto

from mldft.datagen.datasets.small_dataset import SmallDataset


class RotatedEthanolDataset(SmallDataset):
    """Class for the rotated ethanol dataset.

    Attributes:
        name: Name of the dataset.
        raw_data_dir: Path to the directory containing the raw data.
        kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
        num_processes: Number of processes to use for the computation.
        num_molecules: Number of molecules in the dataset.
        molecules: List of molecules in the dataset.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        rotations_dir: str,
        filename: str,
        name: str,
        num_processes: int = 1,
        number_of_rotations: int = 10,
        seed: int = 42,
    ):
        """Define molecules and initialize using parent class.

        Args:
            raw_data_dir: Path to the directory containing the raw data.
            kohn_sham_data_dir: Path to the directory containing the Kohn-Sham data.
            label_dir: Path to the directory containing the labels.
            rotations_dir: Path to the directory containing the rotation matrices.
            filename: The filename to use for the output files.
            name: Name of the dataset, used as the folder name.
            num_processes: Number of processes to use for dataset verifying or loading.
            number_of_rotations: Number of rotations to perform on the molecule.
            seed: Seed for the random rotations.
        """

        np.random.seed(seed)

        ethanol_structure = [
            ["H", (1.8853, -0.0401, 1.0854)],
            ["C", (1.2699, -0.0477, 0.1772)],
            ["H", (1.5840, 0.8007, -0.4449)],
            ["H", (1.5089, -0.9636, -0.3791)],
            ["C", (-0.2033, 0.0282, 0.5345)],
            ["H", (-0.4993, -0.8287, 1.1714)],
            ["H", (-0.4235, 0.9513, 1.1064)],
            ["O", (-0.9394, 0.0157, -0.6674)],
            ["H", (-1.8540, 0.0626, -0.4252)],
        ]
        self.molecules = [gto.M(atom=ethanol_structure, unit="angstrom")]
        rotations_dir = Path(rotations_dir)
        rotations_dir.mkdir(parents=True, exist_ok=True)
        for ind in range(number_of_rotations):
            rotated_molecule_structure, rotation_matrix = self._rotate_molecule_structure_randomly(
                ethanol_structure
            )
            self._save_rotation_matrix(rotation_matrix, rotations_dir, ind)
            self.molecules.append(
                gto.M(
                    atom=rotated_molecule_structure,
                    unit="angstrom",
                )
            )

        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
            molecules=self.molecules,
        )

    def _rotate_molecule_structure_randomly(self, molecule_structure: list):
        """Rotate the molecule structure randomly.

        Args:
            molecule_structure: The structure of the molecule.
        """
        rotation_matrix = rand_matrix(1)[0]
        rotated_molecule_structure = copy.deepcopy(molecule_structure)
        for atom in rotated_molecule_structure:
            atom[1] = tuple(rotation_matrix @ np.array(atom[1]))

        return rotated_molecule_structure, rotation_matrix

    def _save_rotation_matrix(self, rotation_matrix, rotations_dir, ind):
        """Save the rotation matrix to a file.

        Args:
            rotation_matrix: The rotation matrix to save.
            rotations_dir: The directory to save the rotation matrix to.
            ind: The index of the rotation matrix.
        """
        np.save(rotations_dir / f"rotation_{ind:07}.npy", rotation_matrix)
