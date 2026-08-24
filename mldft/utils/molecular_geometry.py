import numpy as np
from irmsd import get_irmsd_rdkit
from loguru import logger
from pyscf import gto
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdmolops, rdMolTransforms

from mldft.utils.conversions import pyscf_to_rdkit

LINEAR_ANGLE_THRESHOLD_DEG = 179.0


def get_bond_lengths(rdkit_mol: Chem.Mol) -> np.ndarray:
    """Get bond lengths for all unique bonds in a molecule.

    Args:
        rdkit_mol (rdkit.Chem.Mol): RDKit molecule with 3D coordinates.

    Returns:
        np.ndarray: Array of bond lengths.
    """
    # Get the number of conformers
    # This assumes that the molecule has been properly set up with conformers
    if rdkit_mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have at least one conformer.")

    # Get the number of conformers
    n_confs = rdkit_mol.GetNumConformers()
    bond_lengths = []
    for bond in rdkit_mol.GetBonds():
        id1 = bond.GetBeginAtomIdx()
        id2 = bond.GetEndAtomIdx()
        bond_length = []
        for i, conf in enumerate(rdkit_mol.GetConformers()):
            bond_length.append(rdMolTransforms.GetBondLength(conf, id1, id2))
        bond_lengths.append((id1, id2, *bond_length))

    return np.asarray(bond_lengths)


def get_unique_bond_angles(mol: Chem.Mol) -> np.ndarray:
    """Find all unique bond angles (A–B–C) in a molecule.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule with 3D coordinates.

    Returns:
        np.ndarray: Array of unique bond angles in degrees, with each row containing
            (atom_a_idx, atom_b_idx, atom_c_idx, angle_1, angle_2, ..., angle_n), where n is the
            number of conformers.
    """
    unique_angles = set()
    angles = []

    n_confs = mol.GetNumConformers()

    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for i, idx1 in enumerate(neighbors[:-1]):
            for j, idx2 in enumerate(neighbors[i + 1 :]):
                logger.trace(f"{idx0} {idx1} {idx2}")
                a, c = sorted([idx1, idx2])  # again ensure uniqueness
                b = idx0
                triplet = (a, b, c)
                if triplet not in unique_angles:
                    unique_angles.add(triplet)

                    angle = []
                    for i, conf in enumerate(mol.GetConformers()):
                        angle.append(rdMolTransforms.GetAngleDeg(conf, a, b, c))
                    angles.append((a, b, c, *angle))

    logger.trace(f"No. Angles: {len(angles)}")

    return np.asarray(angles)


def _has_linear_dihedral_endpoint(
    mol: Chem.Mol, atom_i: int, atom_j: int, atom_k: int, atom_l: int, conf_id: int = 0
) -> bool:
    """Return whether a proper dihedral is ill-defined due to a nearly linear endpoint angle."""
    conformer = mol.GetConformer(conf_id)
    start_angle = rdMolTransforms.GetAngleDeg(conformer, atom_i, atom_j, atom_k)
    end_angle = rdMolTransforms.GetAngleDeg(conformer, atom_j, atom_k, atom_l)
    return max(start_angle, end_angle) > LINEAR_ANGLE_THRESHOLD_DEG


def get_unique_dihedral_angles(mol: Chem.Mol) -> np.ndarray:
    """Find unique dihedral angles in a molecule.

    This function computes only the dihedrals along the bonds of the molecule and rejects
    quartets whose start or end bond angle is nearly linear in conformer 0.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule with 3D coordinates.

    Returns:
        np.ndarray: Array of unique dihedral angles in degrees, with each row containing
            (atom_i_idx, atom_j_idx, atom_k_idx, atom_l_idx, angle_1, angle_2, ..., angle_n),
            where n is the number of conformers.
    """
    unique_dihedrals = set()
    angles = []

    n_confs = mol.GetNumConformers()

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(begin)
        end_atom = mol.GetAtomWithIdx(end)

        # Get neighbors of begin and end atoms (excluding each other)
        begin_neighbors = [a.GetIdx() for a in begin_atom.GetNeighbors() if a.GetIdx() != end]
        end_neighbors = [a.GetIdx() for a in end_atom.GetNeighbors() if a.GetIdx() != begin]

        # Form dihedrals
        for i in begin_neighbors:
            for j in end_neighbors:
                # The first three atoms are plane1 and the last three are plane2
                # If I am correct we then have the following symmetries in the indices
                # (i, j, k, l) = (l, k, j, i) = (i, k, j, l) = (l, j, k, i)
                # as these define the same planes
                # no further symmetries are taken into account
                dihedral = tuple([i, begin, end, j])
                if (
                    dihedral in unique_dihedrals
                    or tuple(reversed(dihedral)) in unique_dihedrals
                    or (dihedral[0], dihedral[2], dihedral[1], dihedral[3]) in unique_dihedrals
                    or (dihedral[3], dihedral[1], dihedral[2], dihedral[0]) in unique_dihedrals
                ):
                    continue
                if _has_linear_dihedral_endpoint(mol, i, begin, end, j):
                    continue
                unique_dihedrals.add(dihedral)

                # Compute angle
                angle = []
                for n in range(n_confs):
                    angle.append(
                        rdMolTransforms.GetDihedralDeg(mol.GetConformer(n), i, begin, end, j)
                    )
                angles.append([i, begin, end, j, *angle])

    return np.asarray(angles)


def get_geometry_features(
    mol1: gto.Mole, mol2: gto.Mole, align: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Chem.Mol, float | None]:
    """Compares the molecular structure of two Molecules and computes relevant properties.

    Returns:
        dist_matrix1 (np.ndarray): Distance matrix for the first molecule.
        dist_matrix2 (np.ndarray): Distance matrix for the second molecule.
        bond_lengths (np.ndarray): Bond lengths for the first molecule.
        bond_angles (List[Tuple[int, int, int, float]]): Unique bond angles for the first molecule.
        dihedral_angles (List[Tuple[int, int, int, int, float]]): Unique dihedral angles for the first molecule.
        rdkit_mol (rdkit.Chem.Mol): RDKit molecule with both conformers.
    """
    best_rmsd = None
    # Ensure both molecules have the same number of atoms
    if mol1.natm != mol2.natm:
        raise ValueError("Molecules must have the same number of atoms for comparison.")

    # get bond lengths, angles and dihedrals
    rdkit_mol1 = pyscf_to_rdkit(mol1, sanitize=False)
    rdkit_mol2 = pyscf_to_rdkit(mol2, sanitize=False)
    logger.trace("Converted PySCF molecules to RDKit.")
    if align:
        try:
            best_rmsd, rdkit_mol1, rdkit_mol2 = get_irmsd_rdkit(rdkit_mol1, rdkit_mol2)
            logger.debug(f"Aligned mol1 to mol2 with RMSD: {best_rmsd:.4f} Å")
            try:
                rdDetermineBonds.DetermineConnectivity(rdkit_mol1, charge=mol1.charge)
                rdDetermineBonds.DetermineBondOrders(rdkit_mol1, charge=mol1.charge)
            except Exception as e:
                logger.error(f"Error during determining bonds of molecules: {e}")

        except Exception as e:
            logger.error(f"Error during alignment of molecules: {e}")
    else:
        atomic_numbers1 = np.asarray(
            [atom.GetAtomicNum() for atom in rdkit_mol1.GetAtoms()], dtype=int
        )
        atomic_numbers2 = np.asarray(
            [atom.GetAtomicNum() for atom in rdkit_mol2.GetAtoms()], dtype=int
        )
        assert all(
            atomic_numbers1 == atomic_numbers2
        ), f"Atomic numbers do not match between the two molecules. {atomic_numbers1} != {atomic_numbers2}"

    rdkit_mol = Chem.Mol(rdkit_mol1)
    rdkit_mol.AddConformer(rdkit_mol2.GetConformer(0), assignId=True)

    dist_matrix1 = rdmolops.Get3DDistanceMatrix(rdkit_mol, confId=0)
    dist_matrix2 = rdmolops.Get3DDistanceMatrix(rdkit_mol, confId=1)

    try:
        bond_lengths = get_bond_lengths(rdkit_mol)
        bond_angles = get_unique_bond_angles(rdkit_mol)
        dihedral_angles = get_unique_dihedral_angles(rdkit_mol)
    except Exception as e:
        bond_lengths = np.array([])
        bond_angles = np.array([])
        dihedral_angles = np.array([])
        logger.error(f"Error computing geometry features: {e}")

    return (
        dist_matrix1,
        dist_matrix2,
        bond_lengths,
        bond_angles,
        dihedral_angles,
        rdkit_mol,
        best_rmsd,
    )
