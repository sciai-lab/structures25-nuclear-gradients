import shutil
from pathlib import Path

import basis_set_exchange
import numpy as np
import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict
from pyscf import gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import (
    PrepareForDensityOptimization,
    to_torch,
)
from mldft.ml.data.components.of_data import OFData
from mldft.ml.train import train
from mldft.ofdft import nuclear_gradients as nuclear_gradients_module
from mldft.ofdft.energies import Energies


@pytest.fixture(scope="module", params=[(1, 2, 3, 4)])
def energies(molecule_medium, request):
    """Returns an Energies object for each given molecule."""
    kinetic, hartree, xc, nuclear_attraction = request.param
    return Energies(
        kinetic=kinetic,
        hartree=hartree,
        xc=xc,
        nuclear_attraction=nuclear_attraction,
        mol=molecule_medium,
    )


@pytest.fixture(scope="module")
def basis_info():
    """Returns a BasisInfo object for the 6-31G basis set."""
    basis_name = "6-31G"
    atomic_numbers = list(range(1, 30))
    elements = [gto.param.ELEMENTS[atomic_number] for atomic_number in atomic_numbers]
    nwchem_string = basis_set_exchange.api.get_basis(basis_name, elements=elements, fmt="nwchem")
    basis_info = BasisInfo.from_nwchem(nwchem_string)
    return basis_info


@pytest.fixture(scope="module")
def ofdata_three_atoms_and_more(molecule_three_atoms_and_more, basis_info):
    """Returns an OFData object for molecules with more than three atoms containing a BasisInfo
    object."""
    mol = molecule_three_atoms_and_more
    pos = mol.atom_coords()
    atomic_numbers = mol.atom_charges()
    coeffs = np.random.rand(mol.nao)
    ofdata = OFData.construct_new(
        basis_info,
        pos,
        atomic_numbers,
        coeffs,
        add_irreps=True,
        basis_integrals="infer_from_basis",
    )
    # basis transformation tests require float64 to be accurate enough
    ofdata = to_torch(ofdata, float_dtype=torch.float64)
    return ofdata


@pytest.fixture(scope="module")
def ofdata_single(basis_info):
    """Returns an OFData object for a single molecule containing a BasisInfo object."""
    # Ethanol molecule (C2H5OH)
    mol = gto.M(
        atom="""    H      1.8853     -0.0401      1.0854
                    C      1.2699     -0.0477      0.1772
                    H      1.5840      0.8007     -0.4449
                    H      1.5089     -0.9636     -0.3791
                    C     -0.2033      0.0282      0.5345
                    H     -0.4993     -0.8287      1.1714
                    H     -0.4235      0.9513      1.1064
                    O     -0.9394      0.0157     -0.6674
                    H     -1.8540      0.0626     -0.4252"""
    )
    pos = mol.atom_coords()
    atomic_numbers = mol.atom_charges()
    coeffs = np.random.rand(mol.nao)
    ofdata = OFData.construct_new(
        basis_info,
        pos,
        atomic_numbers,
        coeffs,
        add_irreps=True,
        basis_integrals="infer_from_basis",
    )
    ofdata.coeffs = np.random.rand(ofdata.n_basis_per_atom.sum())
    # basis transformation tests require float64 to be accurate enough
    ofdata = to_torch(ofdata, float_dtype=torch.float64)
    return ofdata


@pytest.fixture(scope="module")
def dummy_basis_info() -> BasisInfo:
    """Create a dummy BasisInfo object containing information about an OF basis including elements
    H, C and O."""

    atomic_numbers = np.array([1, 6, 8], dtype=np.uint8)
    basis_dict = {"H": "6-31++G**", "C": "6-31++G**", "O": "6-31++G**"}
    irreps_per_atom = ["3x0e+1x1o", "4x0e+3x1o+1x2e", "4x0e+3x1o+1x2e"]

    # integrals over non-S basis functions are zero
    integrals = [
        np.array([1.1, 1.2, 1.3, 0, 0, 0], dtype=np.float64),
        np.array([2.1, 2.2, 2.3, 2.4] + [0] * 14, dtype=np.float64),
        np.array([3.1, 3.2, 3.3, 3.4] + [0] * 14, dtype=np.float64),
    ]

    return BasisInfo(
        atomic_numbers=atomic_numbers,
        basis_dict=basis_dict,
        irreps_per_atom=irreps_per_atom,
        integrals=integrals,
    )


@pytest.fixture
def ml_model_data(tmp_path: Path, monkeypatch):
    """A simple small model and the corresponding datamodule."""
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=two_electron",
                "data/transforms=no_basis_transforms",
                "model=graphformer_small",
                "trainer.precision=64",
                "trainer.deterministic=True",
                "paths.output_dir=" + str(tmp_path),
            ],
        )
    datamodule = instantiate(cfg.data.datamodule)
    datamodule.setup("fit")

    model = instantiate(cfg.model)
    model = model.to("cpu")
    model.eval()
    # One central test policy: keep auto-selected nuclear-gradient device on CPU.
    monkeypatch.setattr(
        nuclear_gradients_module, "get_optimal_device", lambda _mol: torch.device("cpu")
    )
    basis_info = instantiate(cfg.data.basis_info)
    transforms = instantiate(cfg.data.transforms)
    transforms.pre_transforms.insert(0, PrepareForDensityOptimization(basis_info, add_grid=False))
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False

    return model, datamodule, transforms, cfg


@pytest.fixture(scope="package")
def generate_run_path(tmp_path_factory, create_two_electron_dataset):
    """Generate run path and model files for test."""
    run_path = tmp_path_factory.mktemp("run_path") / "model"
    run_path.mkdir()

    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=two_electron",
                "data/transforms=no_basis_transforms",
                # "data/transforms=global_natrep",
                # "data/transforms=local_frames_global_natrep",
                "model=graphformer_small",
            ],
        )
        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.datamodule.batch_size = 2
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 2
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.limit_test_batches = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.denop_val = True
            # these options could be used to speed up this fixture if denop val is needed
            cfg.data.datamodule.n_denop_mols = 1
            cfg.density_optimizer.max_cycle = 2
            cfg.data.num_workers = 0
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.paths.output_dir = str(run_path)
            cfg.paths.log_dir = str(run_path)
            cfg.paths.work_dir = str(run_path)
            cfg.trainer.check_val_every_n_epoch = 1
    train(cfg)

    yield run_path

    # cleanup
    shutil.rmtree(run_path, ignore_errors=True)
