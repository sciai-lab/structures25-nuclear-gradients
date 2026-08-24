import numpy as np
import pytest
from pyscf import gto

from mldft.utils.conversions import pyscf_to_rdkit
from mldft.utils.molecular_geometry import (
    get_bond_lengths,
    get_unique_bond_angles,
    get_unique_dihedral_angles,
)


# Fixture for sample molecule with 4 atoms
# such that dihedrals are defined
@pytest.fixture
def sample_molecule():
    """Fixture for a sample molecule."""
    # Create a simple molecule with 4 atoms
    mol = gto.Mole()
    mol.atom = """
    O      0.000000    0.000000    0.000000
    O      1.638752   -0.000000   -0.000000
    H     -0.731444    0.872482    0.090110
    H      2.370192   -0.872497   -0.089994
    """
    mol.basis = "sto-3g"
    mol.build()

    bond_lengths = np.array(
        [
            [0, 1, 1.639],
            [0, 2, 1.142],
            [1, 3, 1.142],
        ]
    )
    bond_angles = np.array(
        [
            [1, 0, 2, 129.825],
            [0, 1, 3, 129.825],
        ]
    )
    dihedral_angles = np.array(
        [
            [2, 0, 1, 3, 180.0],
        ]
    )
    return mol, bond_lengths, bond_angles, dihedral_angles


# Test for get_bond_lengths
def test_get_bond_lengths(sample_molecule):
    """Test the get_bond_lengths function."""
    mol, bond_lengths_expected, *_ = sample_molecule
    rdkit_mol = pyscf_to_rdkit(mol)
    bond_lengths = get_bond_lengths(rdkit_mol)
    assert bond_lengths.shape[1] == 3, "Bond lengths shape is incorrect."
    assert (
        pytest.approx(bond_lengths[:, -1], abs=0.001) == bond_lengths_expected[:, -1]
    ), "Bond lengths do not match expected values."


# Test for get_unique_bond_angles
def test_get_unique_bond_angles(sample_molecule):
    """Test the get_unique_bond_angles function.

    This barely tests the function.
    """
    mol, _, bond_angles_expected, _ = sample_molecule
    rdkit_mol = pyscf_to_rdkit(mol)
    angles = get_unique_bond_angles(rdkit_mol)
    assert len(angles) > 0, "No unique bond angles found."
    assert angles.shape[1] == 4, "Bond angles shape is incorrect."
    assert (
        pytest.approx(angles, abs=0.1) == bond_angles_expected
    ), "Bond angles do not match expected values."


def test_get_unique_dihedral_angles(sample_molecule):
    """Test the get_unique_dihedral_angles function.

    This barely tests the function.
    """
    mol, _, _, dihedral_angles_expected = sample_molecule
    rdkit_mol = pyscf_to_rdkit(mol)
    dihedrals = get_unique_dihedral_angles(rdkit_mol)
    assert len(dihedrals) > 0, "No unique dihedral angles found."
    assert dihedrals.shape[1] == 5, "Dihedral angles shape is incorrect."
    assert (
        pytest.approx(dihedrals[:, -1], abs=0.1) == dihedral_angles_expected[:, -1]
    ), "Dihedral angles do not match expected values."


def test_get_unique_dihedral_angles_reject_linear_endpoint():
    """Dihedrals with a linear endpoint angle should be rejected."""
    mol = gto.Mole()
    mol.atom = """
    H      0.000000    0.000000    0.000000
    C      1.060000    0.000000    0.000000
    C      2.260000    0.000000    0.000000
    H      3.320000    0.000000    0.000000
    """
    mol.basis = "sto-3g"
    mol.build()

    rdkit_mol = pyscf_to_rdkit(mol)
    dihedrals = get_unique_dihedral_angles(rdkit_mol)

    assert dihedrals.size == 0, "Linear endpoint dihedrals should be rejected."
