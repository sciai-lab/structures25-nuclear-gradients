import copy
import itertools
from functools import partial
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from pyscf import gto

import mldft.ofdft.nuclear_gradients as gradients_module
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import (
    PrepareForDensityOptimization,
    ToTorch,
)
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.local_frames_module import (
    LocalFramesTransformMatrixDense,
)
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_nuclear_attraction_vector,
)
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.nuclear_gradients import (
    AnalyticalNuclearGradients,
    NumericalNuclearGradients,
    _should_add_hellmann_contribution,
    get_basis_transform_gradient_from_config,
    get_gradient_method,
    global_symmetric_natrep_gradient_generator,
    hartree_basisfunction_gradient,
    hellmann_gradient,
    hellmann_gradient_torch,
    local_frames_global_symmetric_natrep_gradient_lf_sparse_generator,
    local_frames_gradient_generator,
    local_frames_gradient_generator_sparse,
    no_basis_transforms_gradient_generator,
    nuclear_attraction_basisfunction_gradient,
)
from mldft.utils.molecules import (
    build_molecule_ofdata,
)
from mldft.utils.sample_generator import SampleGenerator
from mldft.utils.utils import set_default_torch_dtype
from scripts.add_nuclear_gradients import (
    apply_post_basis_transform_to_gradient,
    get_post_basis_transform_matrix,
)

# small molecules with only H, C, N, O atoms
test_molecules = [
    # H2 with a bond length of 0.74 angstrom
    gto.M(atom="H 0 0 0; H 0 0 0.74"),
    # Water molecule (H2O)
    gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0"),
    # Nitrogen molecule (N2)
    gto.M(atom="N 0 0 0; N 0 0 1.1"),
    # Carbon monoxide molecule (CO)
    gto.M(atom="C 0 0 0; O 0 0 1.08"),
    # Methane molecule (CH4)
    gto.M(
        atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63"
    ),
    # Ammonia molecule (NH3)
    gto.M(atom="N 0 0 0; H 0.93 0.93 0.93; H -0.93 -0.93 0.93; H 0.93 -0.93 -0.93"),
]


test_molecules_local_frames = [
    gto.M(
        atom=(
            "H 1.8853 -0.0401 1.0854; C 1.2699 -0.0477 0.1772; H 1.5840 0.8007 -0.4449; H 1.5089 -0.9636 -0.3791; "
            "C -0.2033 0.0282 0.5345; H -0.4993 -0.8287 1.1714; H -0.4235 0.9513 1.1064; O -0.9394 0.0157 -0.6674; "
            "H -1.8540 0.0626 -0.4252"
        )
    ),
    # Glycerol molecule (C3H8O3)
    gto.M(
        atom=(
            "C           0.24634       -0.68523       -1.26615; C          -0.23965        0.02987        0.00710;"
            "C           0.24919       -0.69161        1.27030; O           0.18005        1.37168        0.00804;"
            "H           1.17539        1.37923        0.00675; O          -0.25494       -0.02623        2.40671;"
            "H           0.09388       -0.52771        3.19406; O          -0.24759       -0.02345       -2.38899;"
            "H           0.09315       -0.52742       -3.17668; H          -0.12425       -1.73715       -1.25348;"
            "H           1.36410       -0.68408       -1.28903; H          -1.34734        0.01027        0.00023;"
            "H           1.35770       -0.68135        1.30207; H          -0.12679       -1.73425        1.26193"
        )
    ),
    # Aniline molecule (C6H7N)
    gto.M(
        atom=(
            "H           2.88099       -0.25383        4.78301; C           1.82147       -0.16147        3.01230;"
            "C          -0.80272       -0.35814        3.04355; H          -1.80408       -0.58913        4.83805;"
            "C          -2.17200       -0.24843        0.80334; H          -4.23295       -0.39751        0.85592;"
            "C          -0.89781        0.06830       -1.50241; N          -2.29704        0.38168       -3.78947;"
            "H          -3.95783       -0.50390       -3.71183; H          -1.30710       -0.16306       -5.29327;"
            "C           1.74444        0.24466       -1.53056; H           2.77185        0.48795       -3.31581;"
            "C           3.08480        0.13777        0.72575; H           5.14662        0.28604        0.70319"
        )
    ),
]

test_basis_sets = [
    "cc-pvdz",
    # "6-31G(2df,p)",
    "sto-3g",  # minimal basis set, problematic in certain density fitting cases
    # These seem to be less accurate for numerical integration
    # "def2-universal-jfit",
    # "pc-3",
    # "aug-cc-pvdz",
]

test_molecules_local_frames = itertools.product(
    copy.deepcopy(test_molecules_local_frames), test_basis_sets
)


@pytest.fixture(scope="module", params=test_molecules_local_frames)
def molecules_local_frames(request) -> gto.Mole:
    """Get molecules large enough to have local frames."""
    mol, basis = request.param
    mol = gto.M(atom=mol.atom, basis=basis)
    return mol


class MockSample:
    def __init__(self, mol, coeffs):
        self.mol = mol
        self.coeffs = coeffs


atoms = [
    "C 0 0 0; H 0.6 0.63 0.63; H -0.63 -0.7 0.63; H -0.61 0.63 -0.59; H 0.63 -0.7 -0.7",
    "N 0 0 0; H 0.99 0.93 0.93; H -0.93 -0.91 0.95; H 0.89 -0.99 -0.93",
]


@pytest.fixture(scope="module", params=atoms)
def molecule_non_degenerate_ovlp_eigenvalues(request) -> gto.Mole:
    """Get many defined molecules as a pytest fixture."""
    atom = request.param
    mol = gto.M(atom=atom, basis="6-31G(2df,p)")
    return mol


@pytest.fixture(scope="module")
def molecule_for_local_frame_global_natrep() -> gto.Mole:
    """Get many defined molecules as a pytest fixture."""
    atom = (
        "H 1.8853 -0.0401 1.0854; C 1.2699 -0.0477 0.1772; H 1.5840 0.8007 -0.4449; H 1.5089 -0.9636 -0.3791; "
        "C -0.2033 0.0282 0.5345; H -0.4993 -0.8287 1.1714; H -0.4235 0.9513 1.1064; O -0.9394 0.0157 -0.6674; "
        "H -1.8540 0.0626 -0.4252"
    )
    mol = gto.M(atom=atom, basis="6-31G(2df,p)")
    return mol


@pytest.fixture(scope="module")
def sample_generator_local_frame_model():
    """Sample generator for local frame model."""
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=local_frames",
                "model=toy_model",
                "model.dataset_statistics=None",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)
    sample_generator = SampleGenerator(cfg, model)

    return sample_generator


@pytest.fixture(scope="module")
def sample_generator_local_frame_global_natrep_model():
    """Sample generator for local frame global natrep model."""
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=local_frames_global_natrep",
                "model=toy_model",
                "model.dataset_statistics=None",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)
    sample_generator = SampleGenerator(cfg, model)

    return sample_generator


def numerical_nuclear_gradient_nuclear_attraction(sample, epsilon=1e-5):
    """Numerical nuclear gradient for the nuclear attraction energy."""
    mol = sample.mol
    unit = "bohr"
    coords = mol.atom_coords(unit=unit)
    gradient = torch.zeros(mol.natm, 3, dtype=OFDFT_DTYPE)

    func_factory = FunctionalFactory("nuclear_attraction")
    for i in range(mol.natm):
        for k in range(3):
            energies = []
            for l, sign in enumerate([1, -1]):
                coords_tmp = coords.copy()
                coords_tmp[i][k] += sign * epsilon
                mol_tmp = mol.set_geom_(coords_tmp, unit=unit, inplace=False)

                integral = torch.as_tensor(
                    get_nuclear_attraction_vector(mol_tmp), dtype=OFDFT_DTYPE
                )
                functional = func_factory.construct(
                    mol_tmp, coulomb_matrix=None, nuclear_attraction_vector=integral
                )
                energy, _ = functional(sample)

                energies.append(energy.total_energy)

            gradient[i, k] = (energies[0] - energies[1]) / (2 * epsilon)

    return gradient.reshape(mol.natm * 3)


def numerical_nuclear_gradient_hartree(sample, epsilon=1e-5):
    """Numerical nuclear gradient for the Hartree energy."""
    mol = sample.mol
    unit = "bohr"
    coords = mol.atom_coords(unit=unit)
    gradient = torch.zeros(mol.natm, 3, dtype=OFDFT_DTYPE)

    func_factory = FunctionalFactory("hartree")
    for i in range(mol.natm):
        for k in range(3):
            energies = []
            for l, sign in enumerate([1, -1]):
                coords_tmp = coords.copy()
                coords_tmp[i][k] += sign * epsilon
                mol_tmp = mol.set_geom_(coords_tmp, unit=unit, inplace=False)

                integral = torch.as_tensor(get_coulomb_matrix(mol_tmp), dtype=OFDFT_DTYPE)
                functional = func_factory.construct(
                    mol_tmp, coulomb_matrix=integral, nuclear_attraction_vector=None
                )
                energy, _ = functional(sample)

                energies.append(energy["hartree"])

            gradient[i, k] = (energies[0] - energies[1]) / (2 * epsilon)

    return gradient.reshape(mol.natm * 3)


def numerical_nuclear_gradient_ovlp_eigenproblem(mol, epsilon=1e-5, threshold=1e-8):
    """Numerical nuclear gradient for the overlap matrix eigenproblem."""
    unit = "bohr"
    coords = mol.atom_coords(unit=unit)
    derivative_eigval = np.zeros((mol.natm, 3, mol.nao), dtype=float)
    derivative_eigvec = np.zeros((mol.natm, 3, mol.nao, mol.nao), dtype=float)

    for i in range(mol.natm):
        for k in range(3):
            eigvals = []
            eigvecs = []
            for l, sign in enumerate([1, -1]):
                coords_tmp = coords.copy()
                coords_tmp[i][k] += sign * epsilon
                mol_tmp = mol.set_geom_(coords_tmp, unit=unit, inplace=False)

                ovlp = mol_tmp.intor("int1e_ovlp")

                eigval, eigvec = np.linalg.eigh(ovlp)
                eigvals.append(eigval)
                eigvecs.append(eigvec)

            derivative_eigval[i, k, ...] = (eigvals[0] - eigvals[1]) / (2 * epsilon)

            parallel_mask = np.einsum("ki,ki->i", eigvecs[0], eigvecs[1]) < 0
            eigvecs[1][:, parallel_mask] *= -1
            deigvecs = eigvecs[0] - eigvecs[1]
            derivative_eigvec[i, k, ...] = deigvecs / (2 * epsilon)
    return derivative_eigval.reshape(mol.natm * 3, mol.nao), derivative_eigvec.reshape(
        mol.natm * 3, mol.nao, mol.nao
    )


def numerical_natrep(mol, epsilon=1e-4):
    """Numerical nuclear gradient for the global symmetric natrep."""
    unit = "bohr"
    coords = mol.atom_coords(unit=unit)
    derivative_natrep = np.zeros((mol.natm, 3, mol.nao, mol.nao), dtype=float)

    for i in range(mol.natm):
        for k in range(3):
            natrep_matrices = []
            for l, sign in enumerate([1, -1]):
                coords_tmp = coords.copy()
                coords_tmp[i][k] += sign * epsilon
                mol_tmp = mol.set_geom_(coords_tmp, unit=unit, inplace=False)

                ovlp = mol_tmp.intor("int1e_ovlp")

                eigval, eigvec = np.linalg.eigh(ovlp)
                eigval_sqrt = np.sqrt(eigval)
                natrep_matrix = eigvec @ np.diag(eigval_sqrt) @ eigvec.T
                natrep_matrices.append(natrep_matrix)

            derivative_natrep[i, k, ...] = (natrep_matrices[0] - natrep_matrices[1]) / (
                2 * epsilon
            )
    return derivative_natrep.reshape((mol.natm * 3, mol.nao, mol.nao))


@set_default_torch_dtype(OFDFT_DTYPE)
def numerical_transformation_matrix_gradient(
    sample, sample_generator: SampleGenerator, epsilon=1e-4
):
    """Numerical nuclear gradient for the full transformation matrix stored on the sample."""
    mol = sample.mol
    coords = mol.atom_coords(unit="bohr")
    transformation_gradient_tensor = torch.zeros(mol.natm, 3, mol.nao, mol.nao, dtype=OFDFT_DTYPE)

    for i in range(mol.natm):
        for j in range(3):
            transformation_matrices = []
            for sign in [1, -1]:
                new_coords = coords.copy()
                new_coords[i][j] += sign * epsilon
                new_mol = mol.set_geom_(new_coords, unit="bohr", inplace=False)
                new_sample = sample_generator.get_sample_from_mol(new_mol)
                transformation_matrices.append(new_sample.transformation_matrix)

            transformation_gradient_tensor[i, j, ...] = (
                transformation_matrices[0] - transformation_matrices[1]
            ) / (2 * epsilon)

    return transformation_gradient_tensor.reshape(mol.natm * 3, mol.nao, mol.nao)


@set_default_torch_dtype(OFDFT_DTYPE)
def numerical_local_frames_global_natrep(sample, model_config: SampleGenerator, epsilon=1e-4):
    """Numerical nuclear gradient for the local frames global symmetric natrep."""
    mol = sample.mol
    coords = mol.atom_coords(unit="bohr")
    transformation_gradient_tensor = torch.zeros(mol.natm, 3, mol.nao, mol.nao, dtype=OFDFT_DTYPE)

    for i in range(mol.natm):
        for j in range(3):
            transformation_matrices = []
            for k, sign in enumerate([1, -1]):
                new_coords = coords.copy()
                new_coords[i][j] += sign * epsilon
                new_mol = mol.set_geom_(new_coords, unit="bohr", inplace=False)
                new_sample = model_config.get_sample_from_mol(new_mol)
                transformation_matrices.append(new_sample.transformation_matrix)

            transformation_gradient_tensor[i, j, ...] = (
                transformation_matrices[0] - transformation_matrices[1]
            ) / (2 * epsilon)

    return transformation_gradient_tensor.reshape(mol.natm * 3, mol.nao, mol.nao)


# NOTE: This test only works if the required Datasetstatistics are actually present on
#       the testing system
@pytest.mark.slow
# def test_local_frame_global_natrep_gradient(molecule_for_local_frame_global_natrep):
def test_local_frame_global_natrep_gradient(molecules_local_frames):
    """Test the local frame global symmetric natrep gradient."""
    mol = molecules_local_frames
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=local_frames_global_natrep",
                "model=toy_model",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)
    model_config = SampleGenerator(cfg, model)
    sample = model_config.get_sample_from_mol(mol)

    numerical_gradient = numerical_local_frames_global_natrep(sample, model_config)
    analytical_gradient = local_frames_global_symmetric_natrep_gradient_lf_sparse_generator(
        sample, device=sample.pos.device
    )

    for generator_matrix, expected_matrix in zip(analytical_gradient, numerical_gradient):
        assert (
            pytest.approx(generator_matrix.detach().to_dense().numpy(), abs=1e-5)
            == expected_matrix.detach().numpy()
        )


@pytest.mark.slow
def test_local_frame_numerical_gradient(molecules_local_frames):
    """Test the local frame global symmetric natrep gradient."""
    mol = molecules_local_frames
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=local_frames",
                "model=toy_model",
            ],
        )
    model = hydra.utils.instantiate(cfg.model)
    model_config = SampleGenerator(cfg, model)
    sample = model_config.get_sample_from_mol(mol)

    numerical_gradient = numerical_local_frames_global_natrep(sample, model_config)
    analytical_gradient = local_frames_gradient_generator_sparse(sample, device=sample.pos.device)
    for generator_matrix, expected_matrinx in zip(analytical_gradient, numerical_gradient):
        assert (
            pytest.approx(generator_matrix.detach().to_dense().numpy(), abs=1e-5)
            == expected_matrinx.detach().numpy()
        )


@set_default_torch_dtype(OFDFT_DTYPE)
def test_natrep_gradient(molecules_local_frames):
    """Test the global symmetric natrep gradient."""
    mol = molecules_local_frames

    MockSample = type("MockSample", (), {"mol": mol})

    num_natrep_grad = torch.as_tensor(numerical_natrep(mol), dtype=OFDFT_DTYPE)

    natrep_gen = global_symmetric_natrep_gradient_generator(MockSample, device="cpu")
    for i, natrep_grad in enumerate(natrep_gen):
        assert pytest.approx(num_natrep_grad[i : i + 1], abs=1e-5) == natrep_grad.numpy()


def test_get_basis_transform_gradient_from_config_applies_post_basis_transforms(
    molecule_non_degenerate_ovlp_eigenvalues,
):
    """Global natrep gradients should include constant post basis transforms like
    ToE3NNConvention."""
    mol = molecule_non_degenerate_ovlp_eigenvalues

    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=global_natrep",
                "model=toy_model",
                "model.dataset_statistics=None",
            ],
        )

    model = hydra.utils.instantiate(cfg.model)
    sample_generator = SampleGenerator(cfg, model)
    sample = sample_generator.get_sample_from_mol(mol)

    analytical_gradient = torch.concatenate(
        [
            mat
            for mat in get_basis_transform_gradient_from_config(cfg)(
                sample, device=sample.pos.device
            )
        ]
    )
    numerical_gradient = numerical_transformation_matrix_gradient(sample, sample_generator)

    assert (
        pytest.approx(numerical_gradient.detach().numpy(), abs=1e-5) == analytical_gradient.numpy()
    )


def test_precomputed_global_natrep_gradient_applies_post_basis_transforms(
    molecule_non_degenerate_ovlp_eigenvalues,
):
    """The nuclear-gradient-loss precompute script must store contractions in the model basis."""
    mol = molecule_non_degenerate_ovlp_eigenvalues

    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=qm9",
                "data/transforms=global_natrep",
                "model=toy_model",
                "model.dataset_statistics=None",
            ],
        )

    model = hydra.utils.instantiate(cfg.model)
    sample_generator = SampleGenerator(cfg, model)
    sample = sample_generator.get_sample_from_mol(mol)

    raw_gradient = torch.concatenate(
        [
            mat
            for mat in global_symmetric_natrep_gradient_generator(sample, device=sample.pos.device)
        ]
    )
    post_transform_matrix = get_post_basis_transform_matrix(
        sample, sample_generator.basis_info, "global_symmetric_natrep"
    )
    precompute_gradient = apply_post_basis_transform_to_gradient(
        raw_gradient, post_transform_matrix
    )
    config_gradient = torch.concatenate(
        [
            mat
            for mat in get_basis_transform_gradient_from_config(cfg)(
                sample, device=sample.pos.device
            )
        ]
    )

    assert torch.allclose(precompute_gradient, config_gradient)


@pytest.mark.slow
def test_local_frame_gradient(molecules_local_frames):
    """Test the local frame gradient."""
    mol = molecules_local_frames

    basis_info = BasisInfo.from_mol(mol)
    transform = PrepareForDensityOptimization(basis_info, add_grid=False)
    sample = OFData.minimal_sample_from_mol(mol, basis_info=basis_info)

    transform(sample)
    ToTorch(float_dtype=OFDFT_DTYPE)(sample)

    n_basis = sample.n_basis_per_atom.sum()
    n_atoms = sample.mol.natm

    local_frames_nuclear_gradient_tensor = torch.concatenate(
        [mat for mat in local_frames_gradient_generator(sample, device=sample.pos.device)]
    )

    func = partial(
        LocalFramesTransformMatrixDense().forward,
        sample.irreps_per_atom,
        atomic_numbers=sample.atomic_numbers,
    )
    pos = sample.pos
    pos.requires_grad = True
    torch_grad_tensor = torch.autograd.functional.jacobian(func, pos)
    torch_grad_tensor = torch_grad_tensor.reshape(n_basis, n_basis, n_atoms * 3)
    torch_grad_tensor = torch_grad_tensor.moveaxis(-1, 0)

    assert (
        pytest.approx(local_frames_nuclear_gradient_tensor.detach().numpy(), abs=1e-5)
        == torch_grad_tensor.detach().numpy()
    )


def test_hartree_basisfunction_gradient_generator(molecule_medium):
    """Basic test for the Hartree basis function gradient generator."""
    mol = molecule_medium
    coeffs = torch.rand(mol.nao, dtype=OFDFT_DTYPE)

    sample = MockSample(mol, coeffs)

    numerical_hartree_gradient = numerical_nuclear_gradient_hartree(sample)
    hartree_gradient = hartree_basisfunction_gradient(sample, coeffs, device=coeffs.device)

    assert (
        pytest.approx(hartree_gradient.detach().numpy(), abs=1e-5)
        == numerical_hartree_gradient.detach().numpy()
    )


def test_nuclear_attraction_basisfunction_gradient(molecule_medium):
    """Basic test for the nuclear attraction basis function gradient."""
    mol = molecule_medium
    coeffs = torch.rand(mol.nao, dtype=OFDFT_DTYPE)

    sample = MockSample(mol, coeffs)

    nuclear_gradient = nuclear_attraction_basisfunction_gradient(
        sample, coeffs, device=coeffs.device
    )
    numerical_nuclear_gradient = numerical_nuclear_gradient_nuclear_attraction(sample)

    # the numerical gradient contains both the response due to the change in the center of the
    # basis functions as well as the contribution to the hellmann forces, so they can be only tested
    # together
    hellmann_gradient = hellmann_gradient_torch(sample, coeffs, device=coeffs.device)

    assert nuclear_gradient.shape == (mol.natm * 3,)
    assert (
        pytest.approx(nuclear_gradient + hellmann_gradient, abs=1e-5) == numerical_nuclear_gradient
    )


def test_hellmann_gradient(molecule_all):
    """Basic test for the OFDFT hellmann forces, does not test the actual values."""
    mol = molecule_all

    coeffs = np.random.rand(mol.nao)

    nuclear_gradient = hellmann_gradient(mol, coeffs)

    assert nuclear_gradient.shape == (mol.natm * 3,)


transforms = [
    (
        "data/transforms=no_basis_transforms",
        no_basis_transforms_gradient_generator,
    ),
    ("data/transforms=global_natrep", global_symmetric_natrep_gradient_generator),
    (
        "data/transforms=local_frames_global_natrep",
        local_frames_global_symmetric_natrep_gradient_lf_sparse_generator,
    ),
]


@pytest.fixture(scope="module", params=transforms)
def get_test_configs(request):
    """Fixture to get test configurations for the test_get_basis_transform_gradient_from_config
    test."""
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        basis_transforms, expected = request.param
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=two_electron",
                basis_transforms,
                "model=graphformer_small",
            ],
        )

    return cfg, expected


def test_get_basis_transform_gradient_from_config(get_test_configs):
    """Test the get_basis_transform_gradient_from_config function."""
    cfg, expected = get_test_configs

    if expected is NotImplementedError:
        with pytest.raises(NotImplementedError):
            get_basis_transform_gradient_from_config(cfg)
    else:
        basis_transform_gradient = get_basis_transform_gradient_from_config(cfg)
        if cfg.data.transforms.name == "global_symmetric_natrep":
            assert callable(basis_transform_gradient)
        else:
            assert basis_transform_gradient == expected


@pytest.mark.parametrize(
    "GradClass",
    [
        (NumericalNuclearGradients),
        (AnalyticalNuclearGradients),
    ],
)
def test_gradients(ml_model_data, GradClass):
    """Test the construction and execution of the gradient method but not the results!"""
    ml_model, data, transforms, cfg = ml_model_data

    sample_generator = SampleGenerator(cfg, ml_model)
    func_factory = sample_generator.get_functional_factory()

    sample = data.val_set[0]
    coeffs = torch.rand(sample.n_basis_per_atom.sum(), dtype=OFDFT_DTYPE)
    mol = build_molecule_ofdata(sample, basis=sample_generator.basis_info.basis_dict)
    # ensure that the tensors have the correct dtype by using the
    # sample generator which can take care of that due to configuring ToTorch
    # correctly
    sample = sample_generator.get_sample_from_mol(mol)
    # Keep device policy explicit at test entrypoint: use the sample's current device
    # and pass it through to the gradient method, rather than forcing CUDA here.
    test_device = sample.coeffs.device
    coeffs = coeffs.to(test_device)
    sample.coeffs = coeffs

    gradient_method = GradClass(sample_generator, func_factory, device=str(test_device))

    assert callable(gradient_method)

    gradients = gradient_method(sample, coeffs)

    assert gradients.shape == (sample.mol.natm * 3,)


class _DummyModel(torch.nn.Module):
    def __init__(self, target_key: str):
        super().__init__()
        self.target_key = target_key


def test_should_add_hellmann_contribution_for_total_energy_model():
    total_energy_model = _DummyModel(target_key="tot")
    functional_factory = FunctionalFactory(total_energy_model)

    assert not _should_add_hellmann_contribution(functional_factory)


def _gradient_contributions(grad_cls, target_key="kin", include_hellmann=True):
    gradients = object.__new__(grad_cls)
    gradients.func_factory = FunctionalFactory(_DummyModel(target_key=target_key))
    gradients.include_hellmann = include_hellmann
    gradients.basis_transform_gradient_wrt_nuclear_coordinates = lambda *_args, **_kwargs: None
    gradients.epsilon = 1e-4
    return gradients.get_contributions(strict=True)


@pytest.mark.parametrize(
    "grad_cls",
    [
        AnalyticalNuclearGradients,
    ],
)
def test_include_hellmann_false_omits_hellmann_for_non_total_energy_model(grad_cls):
    contributions = _gradient_contributions(grad_cls, include_hellmann=False)

    assert hellmann_gradient_torch not in contributions


@pytest.mark.parametrize(
    "grad_cls",
    [
        AnalyticalNuclearGradients,
    ],
)
def test_include_hellmann_default_keeps_hellmann_for_non_total_energy_model(grad_cls):
    contributions = _gradient_contributions(grad_cls)

    assert hellmann_gradient_torch in contributions


@pytest.mark.parametrize(
    ("method_name", "gradient_class_name"),
    [
        ("analytical", "AnalyticalNuclearGradients"),
    ],
)
def test_get_gradient_method_forwards_include_hellmann_to_analytical_methods(
    monkeypatch,
    method_name,
    gradient_class_name,
):
    """The public factory should expose Hellmann suppression for analytical gradients."""
    captured = {}

    class DummyGradient:
        def __init__(
            self,
            sample_generator,
            func_factory,
            *,
            device,
            include_hellmann,
        ):
            captured["sample_generator"] = sample_generator
            captured["func_factory"] = func_factory
            captured["device"] = device
            captured["include_hellmann"] = include_hellmann

    monkeypatch.setattr(gradients_module, gradient_class_name, DummyGradient)

    sample_generator = SimpleNamespace()
    func_factory = SimpleNamespace()
    gradient = get_gradient_method(
        method_name,
        sample_generator,
        func_factory,
        device="cpu",
        include_hellmann=False,
    )

    assert isinstance(gradient, DummyGradient)
    assert captured == {
        "sample_generator": sample_generator,
        "func_factory": func_factory,
        "device": "cpu",
        "include_hellmann": False,
    }


def test_get_gradient_method_constructs_fully_numerical(monkeypatch):
    captured = {}

    class DummyFullyNumericalGradient:
        def __init__(
            self,
            sample_generator,
            func_factory,
            *,
            optimizer,
            epsilon,
            device,
            max_xc_memory,
            normalize_initial_guess,
            ks_basis,
            proj_minao_module,
            sad_guess_kwargs,
            strict,
        ):
            captured["sample_generator"] = sample_generator
            captured["func_factory"] = func_factory
            captured["optimizer"] = optimizer
            captured["epsilon"] = epsilon
            captured["device"] = device
            captured["max_xc_memory"] = max_xc_memory
            captured["normalize_initial_guess"] = normalize_initial_guess
            captured["ks_basis"] = ks_basis
            captured["proj_minao_module"] = proj_minao_module
            captured["sad_guess_kwargs"] = sad_guess_kwargs
            captured["strict"] = strict

    monkeypatch.setattr(
        gradients_module,
        "FullyNumericalNuclearGradients",
        DummyFullyNumericalGradient,
        raising=False,
    )

    sample_generator = SimpleNamespace()
    func_factory = SimpleNamespace()
    optimizer = SimpleNamespace()
    gradient = get_gradient_method(
        "fully_numerical",
        sample_generator,
        func_factory,
        device="cpu",
        optimizer=optimizer,
        epsilon=1e-3,
        max_xc_memory=4000,
        normalize_initial_guess=True,
        ks_basis="6-31G(2df,p)",
        proj_minao_module=None,
        sad_guess_kwargs=None,
        strict=True,
    )

    assert isinstance(gradient, DummyFullyNumericalGradient)
    assert captured == {
        "sample_generator": sample_generator,
        "func_factory": func_factory,
        "optimizer": optimizer,
        "epsilon": 1e-3,
        "device": "cpu",
        "max_xc_memory": 4000,
        "normalize_initial_guess": True,
        "ks_basis": "6-31G(2df,p)",
        "proj_minao_module": None,
        "sad_guess_kwargs": None,
        "strict": True,
    }


def test_fully_numerical_gradient_runs_density_optimization_for_each_displacement(monkeypatch):
    class FakeMol:
        natm = 1

        def __init__(self, coords):
            self.coords = np.asarray(coords, dtype=float)

        def atom_coords(self, unit="Bohr"):
            assert unit == "Bohr"
            return self.coords.copy()

        def set_geom_(self, coords, unit="Bohr", inplace=False):
            assert unit == "Bohr"
            assert inplace is False
            return FakeMol(coords)

    class FakeSample:
        def __init__(self, mol):
            self.mol = mol
            self.transformation_matrix = torch.eye(1, dtype=OFDFT_DTYPE)

    class FakeSampleGenerator:
        def get_sample_from_mol(self, mol):
            return FakeSample(mol)

    density_optimization_calls = []
    transform_calls = []

    def fake_transform_tensor_with_sample(sample, tensor, representation, **kwargs):
        transform_calls.append((sample, tensor, representation, kwargs))
        return tensor

    def fake_density_optimization(sample, mol, optimizer, func_factory, **kwargs):
        density_optimization_calls.append((sample, mol, optimizer, func_factory, kwargs))
        return SimpleNamespace(total_energy=float(mol.coords.sum())), torch.ones(1), True, None

    monkeypatch.setattr(
        gradients_module,
        "transform_tensor_with_sample",
        fake_transform_tensor_with_sample,
    )
    monkeypatch.setattr(gradients_module, "density_optimization", fake_density_optimization)

    gradient = gradients_module.fully_numerical_gradient(
        FakeSampleGenerator(),
        SimpleNamespace(),
        SimpleNamespace(),
        FakeSample(FakeMol([[0.0, 0.0, 0.0]])),
        torch.ones(1, dtype=OFDFT_DTYPE),
        torch.device("cpu"),
        epsilon=1e-3,
    )

    assert torch.allclose(gradient, torch.ones(3, dtype=OFDFT_DTYPE))
    assert len(density_optimization_calls) == 6
    assert len(transform_calls) == 6


def test_fully_numerical_gradient_raises_on_non_convergence(monkeypatch):
    class FakeMol:
        natm = 1

        def atom_coords(self, unit="Bohr"):
            return np.zeros((1, 3))

        def set_geom_(self, coords, unit="Bohr", inplace=False):
            return self

    class FakeSample:
        mol = FakeMol()
        transformation_matrix = torch.eye(1, dtype=OFDFT_DTYPE)

    class FakeSampleGenerator:
        def get_sample_from_mol(self, mol):
            return FakeSample()

    monkeypatch.setattr(
        gradients_module,
        "transform_tensor_with_sample",
        lambda _sample, tensor, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        gradients_module,
        "density_optimization",
        lambda *_args, **_kwargs: (SimpleNamespace(total_energy=0.0), torch.ones(1), False, None),
    )

    with pytest.raises(RuntimeError, match="Density optimization did not converge"):
        gradients_module.fully_numerical_gradient(
            FakeSampleGenerator(),
            SimpleNamespace(),
            SimpleNamespace(),
            FakeSample(),
            torch.ones(1, dtype=OFDFT_DTYPE),
            torch.device("cpu"),
            epsilon=1e-3,
            strict=True,
        )
