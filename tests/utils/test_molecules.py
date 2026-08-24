from pyscf import gto

from mldft.utils import molecules


def test_build_mol_with_even_tempered_basis(molecule_small: gto.Mole):
    """Tests if the build_mol_with_even_tempered_basis function works as intended."""
    mol = molecules.build_mol_with_even_tempered_basis(molecule_small)

    assert mol._atom == molecule_small._atom
    assert mol.spin == molecule_small.spin
    assert mol.nelectron == molecule_small.nelectron


def test_get_mol_view_link_uses_unsimplified_smiles_when_rdkit_parse_fails(
    molecule_small: gto.Mole, monkeypatch
):
    """Guard against RDKit returning ``None`` for the simplification parse step."""

    raw_smiles = "C1=CC=CC=C1"  # dummy value for this mocked path

    monkeypatch.setattr(molecules, "pyscf_to_rdkit", lambda _: object())
    monkeypatch.setattr(molecules.Chem, "MolToSmiles", lambda _: raw_smiles)
    monkeypatch.setattr(molecules.Chem, "MolFromSmiles", lambda _: None)

    link = molecules.get_mol_view_link(molecule_small)

    assert link == f"https://molview.org/?smiles={raw_smiles}"


def test_get_mol_view_link_handles_missing_rdkit_molecule(molecule_small: gto.Mole, monkeypatch):
    """Ensure graceful fallback when molecule conversion returns ``None``."""

    monkeypatch.setattr(molecules, "pyscf_to_rdkit", lambda _: None)

    link = molecules.get_mol_view_link(molecule_small)

    assert link == "Can't compute smiles, rdkit returned no molecule."
