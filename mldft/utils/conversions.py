import numpy as np
import pyscf
import rdkit
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def pyscf_to_rdkit(
    mol: pyscf.gto.mole.Mole,
    infer_bonds=True,
    max_nr_ambiguous_valence_atoms=30,
    sanitize=True,
) -> rdkit.Chem.rdchem.Mol:
    """Convert PySCF molecules to an RDKit molecule with different geometries as conformers, when
    multiple input molecules are given.

    Args:
        mol: The PySCF molecule.
        infer_bonds: Whether to infer bonds from the geometry.
        max_nr_ambiguous_valence_atoms: Maximum total number of N,O,S,P for molecule to be considered for bond order calculation.
        sanitize: Whether to try to sanitize the RDKit molecule.

    Returns:
        rdkit_mol: The RDKit molecule (with conformers).
    """

    xyz_string = mol.tostring("xyz")
    charge = mol.charge
    rdkit_mol = Chem.MolFromXYZBlock(xyz_string)

    if infer_bonds:
        rdkit_mol = Chem.Mol(rdkit_mol)

        rdDetermineBonds.DetermineConnectivity(rdkit_mol, charge=charge)

        # cutoff is set to circumvent large computation times, algorithm scales with 2^n; n being number of ambiguous valence atoms
        target_atomic_num = [7, 8, 15, 16]
        count = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() in target_atomic_num:
                count += 1

        if count <= max_nr_ambiguous_valence_atoms:
            try:
                rdDetermineBonds.DetermineBondOrders(rdkit_mol, charge=charge)

            except Exception as e:
                logger.error(f"Error determining bonds for {mol}: {e}")

    if sanitize:
        try:
            Chem.SanitizeMol(rdkit_mol)
        except Exception as e:
            logger.error(f"Sanitization error for molecule {mol}: {e}")

    return rdkit_mol


def _as_scalar_item(value, *, name: str, expected: str):
    if value is None:
        raise TypeError(f"{name} must be a scalar {expected}, got None.")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one value, got shape {tuple(value.shape)}."
            )
        return value.detach().cpu().item()

    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar-like, got shape {array.shape}.")
    return array.item()


def as_scalar_float(value, *, name: str = "value") -> float:
    return float(_as_scalar_item(value, name=name, expected="float"))


def as_scalar_bool(value, *, name: str = "value") -> bool:
    return bool(_as_scalar_item(value, name=name, expected="bool"))


def str_to_torch_float_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a string to a torch float dtype.

    Useful to set the dtype in hydra configs.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "torch.float64" or dtype == "float64":
        return torch.float64
    elif dtype == "torch.float32" or dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype {dtype}")
