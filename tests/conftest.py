import copy
import itertools
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import open_dict
from pyscf import dft, gto, scf

from mldft.datagen.generate_labels_dataset import run_label_generation
from mldft.datagen.kohn_sham_dataset import compute_kohn_sham_dataset
from mldft.datagen.methods.ksdft_calculation import ksdft
from mldft.ml.compute_dataset_statistics import compute_dataset_statistics
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.utils.create_dataset_splits import create_split_file
from mldft.utils.molecules import load_scf

small_test_molecules = [
    # H2 with a bond length of 0.74 angstrom
    gto.M(atom="H 0 0 0; H 0 0 0.74"),
    gto.M(atom="H 0 0 0; H 0 0 1.4", unit="Bohr"),
    # Helium
    gto.M(atom="He 0 0 0"),
    # Hydrogen fluoride molecule (HF)
    gto.M(atom="H 0 0 0; F 0 0 0.917"),
    gto.M(atom="H 0 0 0; F 0 0 1.7", unit="Bohr"),
]

# Keep non-slow CI coverage broad enough to span atoms, units, and common geometries
# without paying the runtime cost of every molecule in every numerical test.
medium_test_molecules = [
    # H2 with a bond length of 0.74 angstrom
    gto.M(atom="H 0 0 0; H 0 0 0.74"),
    # Helium
    gto.M(atom="He 0 0 0"),
    # Hydrogen fluoride molecule (HF)
    gto.M(atom="H 0 0 0; F 0 0 0.917"),
    # Water molecule (H2O)
    gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0"),
    # Nitrogen molecule (N2)
    gto.M(atom="N 0 0 0; N 0 0 1.1"),
    # Methane molecule (CH4)
    gto.M(
        atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63"
    ),
    # CO2 neutral singlet, 22 electrons (even)
    gto.M(atom="O 0 0 -1.16; C 0 0 0; O 0 0 1.16", spin=0),
]

all_test_molecules = medium_test_molecules + [
    # Partly from https://github.com/nutjunkie/IQmol/tree/master/share/fragments/Molecules
    # Aniline molecule (C6H7N)
    gto.M(
        atom="H 1.5205 -0.1372 2.5286; C 0.9575 -0.0905 1.5914; C -0.4298 -0.1902 1.6060; H -0.9578 -0.3156 2.5570;"
        " C -1.1520 -0.1316 0.4215; H -2.2452 -0.2104 0.4492; C -0.4779 0.0324 -0.7969; N -1.2191 0.2008 -2.0081;"
        " H -2.0974 -0.2669 -1.9681; H -0.6944 -0.0913 -2.8025; C 0.9208 0.1292 -0.8109; H 1.4628 0.2560 -1.7555;"
        " C 1.6275 0.0685 0.3828; H 2.7196 0.1470 0.3709"
    ),
    # Sulfur hexafluoride molecule (SF6)
    gto.M(
        atom="S 0 0 0; F 1.58 0 0; F -1.58 0 0; F 0 1.58 0; F 0 -1.58 0; F 0 0 1.58; F 0 0 -1.58"
    ),
    # Hydrogen chloride molecule (HCl)
    gto.M(atom="Cl 0 0 0; H 0 0 1.27"),
]

three_atom_or_more_test_molecules = [
    # all test molecules with at least three heavy atoms
    mol
    for mol in all_test_molecules
    if (mol.atom_charges() > 1).sum() >= 3
]


test_basis_sets = [
    "cc-pvdz",
    "6-31G(2df,p)",
    # "sto-3g", # minimal basis set, problematic in certain density fitting cases
    # These seem to be less accurate for numerical integration
    # "def2-universal-jfit",
    # "pc-3",
    # "aug-cc-pvdz",
]
density_basis = [
    BasisInfo.from_atomic_numbers_with_even_tempered_basis(np.arange(1, 11)).basis_dict
]

small_test_molecules_and_basis_sets = itertools.product(
    copy.deepcopy(small_test_molecules), test_basis_sets
)
small_test_molecules_with_density_basis_sets = itertools.product(
    copy.deepcopy(small_test_molecules), density_basis
)
medium_test_molecules_and_basis_sets = itertools.product(
    copy.deepcopy(medium_test_molecules), test_basis_sets
)
medium_test_molecules_with_density_basis_sets = itertools.product(
    copy.deepcopy(medium_test_molecules), density_basis
)
three_atoms_or_more_molecules_and_basis_sets = itertools.product(
    copy.deepcopy(three_atom_or_more_test_molecules), test_basis_sets
)
all_test_molecules_and_basis_sets = itertools.product(
    copy.deepcopy(all_test_molecules), test_basis_sets
)


@pytest.fixture(scope="module", params=small_test_molecules_and_basis_sets)
def molecule_small(request) -> gto.Mole:
    """Get a representative set of small molecules as a pytest fixture.

    This generates 10 fixtures in total, 5 molecules with 2 basis sets.
    """
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    # mol.basis = basis
    # mol.build()
    return mol


@pytest.fixture(scope="module", params=small_test_molecules_with_density_basis_sets)
def molecule_small_with_density_basis(request) -> gto.Mole:
    """Get a representative set of small molecules as a pytest fixture.

    This generates 5 fixtures in total, 5 molecules with 1 density basis.
    """
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    return mol


@pytest.fixture(scope="module", params=medium_test_molecules_and_basis_sets)
def molecule_medium(request) -> gto.Mole:
    """Get a representative medium-size test set as a pytest fixture.

    This generates 14 fixtures in total, 7 molecules with 2 basis sets.
    """
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    return mol


@pytest.fixture(scope="module", params=medium_test_molecules_with_density_basis_sets)
def molecule_medium_with_density_basis(request) -> gto.Mole:
    """Get the medium-size test set with the density basis as a pytest fixture.

    This generates 7 fixtures in total, 7 molecules with 1 density basis.
    """
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    return mol


@pytest.fixture(scope="module", params=three_atoms_or_more_molecules_and_basis_sets)
def molecule_three_atoms_and_more(request) -> gto.Mole:
    """Get many defined molecules as a pytest fixture."""
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    return mol


@pytest.fixture(scope="module", params=all_test_molecules_and_basis_sets)
def molecule_all(request) -> gto.Mole:
    """Get a broad but runtime-conscious test set as a pytest fixture.

    This generates 20 fixtures in total, 10 molecules with 2 basis sets.
    """
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis, charge=mol.charge)
    return mol


@pytest.fixture(scope="module")
def molecule_ao_grid_level_5(molecule_medium) -> tuple[gto.Mole, np.ndarray, dft.Grids]:
    """Get a medium molecule with a grid level of 5 and evaluated atomic orbitals as a pytest
    fixture."""
    grid = dft.Grids(molecule_medium)
    grid.level = 5
    grid.build()
    ao = dft.numint.eval_ao(
        molecule_medium, grid.coords, deriv=1
    )  # also handle derivatives that might occur
    return molecule_medium, ao, grid


@pytest.fixture(scope="module", params=[3, 4, 5])
def small_molecule_ao_multiple_grid_levels(
    molecule_small, request
) -> tuple[gto.Mole, np.ndarray, dft.Grids]:
    """Get a medium molecule with a grid level of 5 and evaluated atomic orbitals as a pytest
    fixture."""
    grid_ = dft.Grids(molecule_small)
    grid_.level = request.param
    grid_.build()
    ao_ = dft.numint.eval_ao(
        molecule_small, grid_.coords, deriv=1
    )  # also handle derivatives that might occur
    return molecule_small, ao_, grid_


def _save_additional_data_callback(envs: dict) -> None:
    """Save the data of each iteration in the chkfile. Modified version to provide additional data
    for tests.

    Args:
        envs: Dictionary with the local environment variables of the iteration.
    Returns:
        None
    """
    cycle = envs["cycle"]
    mf = envs["mf"]
    info = {
        # By default, we save cropped version, here we need the full arrays, so overwrite
        "occupation_numbers_orbitals": envs["mo_occ"],
        "molecular_coeffs_orbitals": envs["mo_coeff"],
        # Two additional variables are stored, which are useful for tests
        "fock_matrix": envs["fock"],
        "mo_energy": envs["mo_energy"],
    }
    # Load the existing data, update it with the additional data and save it again
    # Otherwise the data would be overwritten
    data = scf.chkfile.load(mf.chkfile, f"KS-iteration/{cycle:d}")
    data.update(info)
    scf.chkfile.save(mf.chkfile, f"KS-iteration/{cycle:d}", data)


@pytest.fixture
def ksdft_results_small(molecule_small):
    """Run Kohn-Sham DFT on the small molecules and return the results."""
    mol_orbital_basis = molecule_small
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        filesave_path = temp_dir_path / "test.chk"
        assert not filesave_path.exists()
        ksdft(
            mol_orbital_basis,
            filesave_path,
            diis_method="CDIIS",
            extra_callback=_save_additional_data_callback,
        )
        assert filesave_path.exists()
        results, initialization, data_of_iteration = load_scf(filesave_path)
        return results, initialization, data_of_iteration, mol_orbital_basis


@pytest.fixture
def ksdft_results_medium(molecule_medium):
    """Run Kohn-Sham DFT on the medium molecules and return the results."""
    mol_orbital_basis = molecule_medium
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        filesave_path = temp_dir_path / "test.chk"
        assert not filesave_path.exists()
        ksdft(
            mol_orbital_basis,
            filesave_path,
            diis_method="CDIIS",
            extra_callback=_save_additional_data_callback,
        )
        assert filesave_path.exists()
        results, initialization, data_of_iteration = load_scf(filesave_path)
        return results, initialization, data_of_iteration, mol_orbital_basis


@pytest.fixture
def ksdft_results_all(molecule_all):
    """Run Kohn-Sham DFT on all molecules and return the results."""
    mol_orbital_basis = molecule_all
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        filesave_path = temp_dir_path / "test.chk"
        assert not filesave_path.exists()
        ksdft(
            mol_orbital_basis,
            filesave_path,
            diis_method="CDIIS",
            extra_callback=_save_additional_data_callback,
        )

        assert filesave_path.exists()
        results, initialisation, data_of_iteration = load_scf(filesave_path)
        return results, initialisation, data_of_iteration, mol_orbital_basis


@pytest.fixture(scope="package")
def create_two_electron_dataset(tmp_path_factory):
    """Run the full data generation pipeline for the two electron dataset and set the DFT_DATA
    environment variable to a temporary directory containing the dataset."""
    output_path = tmp_path_factory.mktemp("two_electron")
    with initialize(version_base=None, config_path="../configs/datagen/"):
        cfg_datagen = compose(
            config_name="config.yaml",
            overrides=["dataset=two_electron", "n_molecules=-1", "num_processes=1"],
        )
    kohn_sham_dir = Path(cfg_datagen.dataset.kohn_sham_data_dir)
    if kohn_sham_dir.exists():
        shutil.rmtree(kohn_sham_dir)
    label_dir = Path(cfg_datagen.dataset.label_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    compute_kohn_sham_dataset(cfg_datagen)
    run_label_generation(cfg_datagen)
    create_split_file(
        dataset="TwoElectron",
        split_percentages=(0.8, 0.2, 0.0),
        override=True,
        yaml_path=None,
        pickle_path=None,
        processes=1,
    )
    # Compute the dataset statistics for this setting
    output_path = output_path
    output_path.mkdir(exist_ok=True)
    with initialize(version_base="1.3", config_path="../configs/ml"):
        cfg_statistics = compose(
            config_name="statistics.yaml",
            overrides=["data=two_electron", "data/transforms=no_basis_transforms"],
        )
        # set defaults for dataset_statistics
        with open_dict(cfg_statistics):
            cfg_statistics.data.datamodule.num_workers = 0
            cfg_statistics.paths.output_dir = str(output_path)
            cfg_statistics.paths.log_dir = str(output_path)
            cfg_statistics.paths.work_dir = str(output_path)
            # cfg_statistics.

    compute_dataset_statistics(cfg_statistics)
    yield output_path
