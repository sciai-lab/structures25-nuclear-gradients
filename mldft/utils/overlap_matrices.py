# Collection of functions to calculate the different flavors of overlap matrices

from typing import Optional

import numpy as np
import torch
from pyscf.data import elements

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.basis_integrals import get_overlap_matrix
from mldft.utils.molecules import build_molecule_np, build_molecule_ofdata


def get_sample_overlap_matrix(
    basis_info: BasisInfo, sample: OFData, spin: Optional[int] = None
) -> torch.tensor:
    """Constructs the overlap matrix for the given sample.

    Args:
        basis_info: The basis_info for the given sample.
        sample: The sample for which the overlap matrix is constructed.
        spin: The spin of the molecule. Defaults to None.

    Returns:
        overlap_matrix: The overlap matrix for the given sample.
    """
    mol = build_molecule_ofdata(ofdata=sample, basis=basis_info, spin=None)
    sample_overlap_matrix = get_overlap_matrix(mol)

    return sample_overlap_matrix


def get_atomwise_overlap_matrix(atomic_number: int, basis_dict: dict, spin=None) -> torch.tensor:
    """Construct a one atom molecule for the given atomic number, basis_info and spin to return the
    overlap matrix of the one atom molecule.

    Args:
        atomic_number: The atomic number of the atom, e.g. 1 for hydrogen.
        basis_dict: Dictionary mapping atomic numbers to their basis sets.
        spin: The spin of the molecule. Defaults to None.

    Returns:
        atomwise_overlap_matrix: The overlap matrix of the one atom molecule of shape
            (basis_dims, basis_dims).
    """
    atom_positions = np.array([[0, 0, 0]])  # dummy position

    mol = build_molecule_np(
        charges=np.array([atomic_number]), positions=atom_positions, basis=basis_dict, spin=spin
    )

    # return as torch tensor
    per_atom_overlap_matrix = torch.as_tensor(get_overlap_matrix(mol))

    return per_atom_overlap_matrix


def get_atomwise_overlap_dict(
    basis_dict: dict, spin: Optional[int] = None
) -> dict[int, torch.tensor]:
    """Construct a dictionary that maps the basis_dicts' atomic numbers to their atomwise overlap
    matrix.

    Args:
        basis_dict: The basis_infos dictionary mapping atomic numbers to basis sets.
        spin: The spin of the molecule. Defaults to None.

    Returns:
        atomwise_overlap_dict: Dictionary of atomic numbers to their atomwise overlap matrix.
    """

    # create a dict of one atom overlap matrices. charge() maps atomic symbols to atomic numbers
    atomwise_overlap_dict = {
        elements.charge(atomic_number): get_atomwise_overlap_matrix(
            atomic_number=elements.charge(atomic_number), basis_dict=basis_dict, spin=spin
        )
        for atomic_number in basis_dict.keys()
    }

    return atomwise_overlap_dict
