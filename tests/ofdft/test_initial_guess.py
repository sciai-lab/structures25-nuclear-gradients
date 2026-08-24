import pytest
import torch
from pyscf import gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import ToTorch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft import OFDFT_DTYPE, basis_integrals, initial_guess
from mldft.utils.molecules import build_mol_with_even_tempered_basis


def test_hueckel_guess(molecule_medium: gto.Mole):
    """Tests shape and electron number of the hueckel_guess."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=OFDFT_DTYPE)(sample)
    guess = initial_guess.hueckel_guess(
        sample, mol, ks_basis=molecule_medium.basis, transform=False
    )

    assert guess.shape[0] == mol.nao
    assert guess.ndim == 1
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)


def test_minao_guess(molecule_medium: gto.Mole):
    """Tests shape and electron number of the minao_guess."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=OFDFT_DTYPE)(sample)
    guess = initial_guess.minao_guess(sample, mol, ks_basis=molecule_medium.basis, transform=False)

    assert guess.shape[0] == mol.nao
    assert guess.ndim == 1
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)


def test_dumb_guess(molecule_medium: gto.Mole):
    """Tests the dumb_guess function."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=OFDFT_DTYPE)(sample)
    sample.basis_info = BasisInfo.from_mol(mol)
    guess = initial_guess.dumb_guess(sample, mol)

    assert guess.shape[0] == normalization_vector.shape[0]
    assert torch.all(guess[normalization_vector == 0] == 0)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)


def test_initial_guess_from_last_safe(molecule_medium):
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    sample = OFData.minimal_sample_from_mol(mol, add_transformation_matrix=True)
    sample = ToTorch(float_dtype=OFDFT_DTYPE)(sample)
    sample.add_item("mol", mol, representation=Representation.NONE)

    coeffs = torch.randn(mol.nao, dtype=OFDFT_DTYPE)
    prior_normalization = coeffs @ normalization_vector

    initial_guess_vector = initial_guess.initial_guess_from_last_safe(sample, coeffs)

    assert (
        pytest.approx(initial_guess_vector.numpy() @ normalization_vector) == prior_normalization
    )
    assert pytest.approx(initial_guess_vector[normalization_vector == 0].sum()) == 0
