from itertools import product
from pathlib import Path

import numpy as np
import pyscf

from mldft.datagen.datasets.small_dataset import SmallDataset

OUTER_C_DIST = 2.45  # bohr
INNER_C_DIST = 2.27  # bohr
C_H_DIST = 2.05  # bohr
C_H_ANGLE = 120.6 * np.pi / 180  # radians


class CumuleneDataset(SmallDataset):
    """Class for the cumulene dataset."""

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        torsion_dir: str,
        filename: str,
        name: str,
        n_carbons: list,
        n_angles: int,
        num_processes: int = 1,
    ):
        """Define molecules and initialize using parent class.

        Cumulene molecules are a series of carbon atoms connected by double bonds with hydrogen
        atoms on the ends, so the sum formula is :math:`C_n H_2`. The hydrogen atoms can be rotated
        to create different torsion angles.

        Args:
            raw_data_dir: Directory containing raw data.
            kohn_sham_data_dir: Directory containing Kohn-Sham data.
            label_dir: Directory containing labels.
            torsion_dir: Directory to save torsion angles.
            filename: Name of the file to save the dataset to.
            name: Name of the dataset.
            n_carbons: List of the number of carbon atoms in each cumulene molecule.
            n_angles: Number of angles to rotate the hydrogen atoms by.
            num_processes: Number of processes to use.
        """

        torsion_dir = Path(torsion_dir)
        torsion_dir.mkdir(parents=True, exist_ok=True)
        molecules = []
        # only need to rotate up to 90 degrees due to symmetry
        torsion_angles = np.linspace(0, np.pi / 2, n_angles)

        for i, (n_carbon, torsion_angle) in enumerate(product(n_carbons, torsion_angles)):
            print(f"{n_carbon=}, {torsion_angle=}")
            mol = create_cumulene_mol(n_carbon, torsion_angle)
            molecules.append(mol)
            # Save the torsion angle to a file
            np.save(torsion_dir / f"torsion_{i:07}.npy", torsion_angle)

        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
            molecules=molecules,
        )


def create_cumulene_mol(n_carbon: int, torsion_angle: float):
    """Create a cumulene molecule with the specified number of carbons and torsion angle.

    Args:
        n_carbon: Number of carbon atoms in the cumulene.
        torsion_angle: Angle to rotate the hydrogen atoms by.
    """
    if n_carbon < 2:
        raise ValueError("Cumulene must have at least 2 carbons.")

    carbon_distances = np.full(n_carbon - 1, INNER_C_DIST)
    carbon_distances[[0, -1]] = OUTER_C_DIST
    carbon_positions = np.insert(np.cumsum(carbon_distances), 0, 0)

    cumulene_string = ""
    for i in range(n_carbon):
        cumulene_string += f"C {carbon_positions[i]:.2f} 0 0; "

    hydrogen_positions = np.zeros((4, 3))

    hydrogen_positions[0] = [C_H_DIST * np.cos(C_H_ANGLE), C_H_DIST * np.sin(C_H_ANGLE), 0]
    hydrogen_positions[1] = [C_H_DIST * np.cos(C_H_ANGLE), -C_H_DIST * np.sin(C_H_ANGLE), 0]

    x = carbon_positions[-1] - C_H_DIST * np.cos(C_H_ANGLE)
    hydrogen_positions[2] = [x, C_H_DIST * np.sin(C_H_ANGLE), 0]
    hydrogen_positions[3] = [x, -C_H_DIST * np.sin(C_H_ANGLE), 0]

    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(torsion_angle), -np.sin(torsion_angle)],
            [0, np.sin(torsion_angle), np.cos(torsion_angle)],
        ]
    )
    hydrogen_positions[2:] = hydrogen_positions[2:] @ rotation_matrix

    for x, y, z in hydrogen_positions:
        cumulene_string += f"H {x:.4f} {y:.4f} {z:.4f}; "

    cumulene_mol = pyscf.M(atom=cumulene_string, unit="Bohr")
    return cumulene_mol
