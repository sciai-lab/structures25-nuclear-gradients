import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

import mldft.ml.models.components.shellwise_natural_reparameterization as shellwise
from mldft.ml.data.components.convert_transforms import AddOverlapMatrix
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atomwise_natural_reparametrization import (
    AtomwiseNaturalReparametrization,
)
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices,
    natural_reparametrization_matrices_torch,
)
from mldft.utils.molecules import build_molecule_ofdata
from mldft.utils.overlap_matrices import get_atomwise_overlap_matrix
from mldft.utils.utils import set_default_torch_dtype

TEST_DATASET = "MD17subset"
TEST_DATA_DIR_EXISTS = (Path(os.getenv("DFT_DATA")) / TEST_DATASET).exists()


@set_default_torch_dtype(torch.float64)
@pytest.fixture
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def real_dummy_sample_and_basis_info(tmp_path_factory):
    """Create a dummy sample and basis_info from the MD17 dataset."""

    tmp_path = tmp_path = tmp_path_factory.getbasetemp()

    with initialize(
        version_base=None, config_path="../../configs/ml/", job_name="natural_reparametrization"
    ):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=md17subset", "paths.output_dir=" + str(tmp_path)],
        )

    datamodule = instantiate(cfg.data.datamodule)
    datamodule.setup("fit")

    real_dummy_sample = datamodule.train_set[0]  # fixed sample for now
    real_dummy_basis_info = datamodule.train_set.basis_info

    add_overlap_matrix = AddOverlapMatrix(real_dummy_basis_info)
    real_dummy_sample = add_overlap_matrix(real_dummy_sample)

    return real_dummy_sample, real_dummy_basis_info


@pytest.mark.parametrize(
    "overlap_matrix",
    [
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[[1, 0.5], [0.5, 1]], [[2, 0.3], [0.3, 2]]]),
    ],
)
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_natural_reparametrization_matrices(overlap_matrix, orthogonalization):
    """Test that the matrices M and M^-1 are inverses of each other and that M^T M is equal to the
    overlap matrix / matrices for the pre defined overlap matrix."""

    m, m_inv = natural_reparametrization_matrices(overlap_matrix)
    n = overlap_matrix.shape[-1]
    eyes = np.broadcast_to(np.eye(n), overlap_matrix.shape)
    np.testing.assert_allclose(m @ m_inv, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m_inv @ m, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m @ m.swapaxes(-1, -2), overlap_matrix, rtol=1e-05, atol=1e-07)
    if orthogonalization == "symmetric":
        np.testing.assert_allclose(m @ m, overlap_matrix, rtol=1e-05, atol=1e-07)


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_symmetric_natural_reparametrization_matrices(
    real_dummy_sample_and_basis_info, orthogonalization
):
    """Test that the matrices M and M^-1 are inverses of each other and that M^T M is equal to the
    overlap matrix / matrices as well as M = M^T in the symmetric case for actual data samples."""

    real_dummy_sample, _ = real_dummy_sample_and_basis_info
    overlap_matrix = real_dummy_sample.overlap_matrix

    m, m_inv = natural_reparametrization_matrices(
        overlap_matrix, orthogonalization=orthogonalization
    )
    n = overlap_matrix.shape[-1]
    eyes = np.broadcast_to(np.eye(n), overlap_matrix.shape)
    np.testing.assert_allclose(m @ m_inv, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m_inv @ m, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m @ m.T, overlap_matrix, rtol=1e-05, atol=1e-07)
    if orthogonalization == "symmetric":
        np.testing.assert_allclose(m, m.T, rtol=1e-05, atol=1e-07)
        np.testing.assert_allclose(m_inv, m_inv.T, rtol=1e-05, atol=1e-07)
        np.testing.assert_allclose(m @ m, overlap_matrix, rtol=1e-05, atol=1e-07)


# Test atomwise implementation
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_atomwise_natural_reparametrization_inverse(
    real_dummy_sample_and_basis_info, orthogonalization
):
    """Test that the matrices M and M^-1 are inverses of each other and that M^T M is equal to the
    overlap matrix / matrices."""

    real_dummy_sample, real_dummy_basis_info = real_dummy_sample_and_basis_info
    assert isinstance(real_dummy_sample, OFData), "Sample is not an OFData object."
    # This is not the correct overlap matrix for the atom wise case but must have the same shape
    mol = build_molecule_ofdata(real_dummy_sample, real_dummy_basis_info.basis_dict)
    sample_overlap_matrix = deepcopy(real_dummy_sample.overlap_matrix)

    atomwise_nat_rep = AtomwiseNaturalReparametrization.from_basis_info(
        real_dummy_basis_info, orthogonalization, spin=None
    )

    m, m_inv = atomwise_nat_rep.sample_forward(real_dummy_sample)
    n = sample_overlap_matrix.shape[-1]
    eyes = np.broadcast_to(torch.eye(n), sample_overlap_matrix.shape)
    np.testing.assert_allclose(m @ m_inv, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m_inv @ m, eyes, rtol=1e-05, atol=1e-07)


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_block_composition(real_dummy_sample_and_basis_info, orthogonalization):
    """Test whether the atomic natural reparametrization is the correct block diagonal matrix, that
    is, stems from the correct overlap matrix and differs from the global one."""

    real_dummy_sample, real_dummy_basis_info = real_dummy_sample_and_basis_info
    # first get the overlap matrix for the whole sample
    mol = build_molecule_ofdata(real_dummy_sample, real_dummy_basis_info.basis_dict)
    sample_overlap_matrix = real_dummy_sample.overlap_matrix
    # then get the atomic natural reparametrization matrices

    atomwise_nat_rep = AtomwiseNaturalReparametrization.from_basis_info(
        real_dummy_basis_info, orthogonalization=orthogonalization, spin=None
    )

    atomwise_m, atomwise_m_inv = atomwise_nat_rep.sample_forward(real_dummy_sample)

    cum_dims = 0
    for basis_dims in real_dummy_sample.n_basis_per_atom:
        # solely consider the overlap matrix of the current atom
        atom_overlap_matrix = sample_overlap_matrix[
            cum_dims : cum_dims + basis_dims, cum_dims : cum_dims + basis_dims
        ]
        if isinstance(atom_overlap_matrix, np.ndarray):
            atom_overlap_matrix = torch.as_tensor(atom_overlap_matrix)
        eigvals, eigvecs = torch.linalg.eigh(atom_overlap_matrix)
        eigvals_sqrt = torch.sqrt(eigvals)

        if orthogonalization == "symmetric":  # M = Q @ Λ^0.5 @ Q.T
            m = torch.einsum("...ij, ...j, ...kj -> ...ik", eigvecs, eigvals_sqrt, eigvecs)

        elif orthogonalization == "canonical":  # M = Q @ Λ^0.5
            m = torch.einsum("...ij, ...j -> ...ij", eigvecs, eigvals_sqrt)

        torch.testing.assert_close(
            atomwise_m[cum_dims : cum_dims + basis_dims, cum_dims : cum_dims + basis_dims],
            m,
            rtol=1e-05,
            atol=1e-07,
        )
        cum_dims += basis_dims

    # also check that the matrix atomic_m is different from m of the whole sample
    sample_m = natural_reparametrization_matrices_torch(torch.as_tensor(sample_overlap_matrix))[0]
    assert not torch.allclose(
        atomwise_m, sample_m
    ), """The atomwise natural reparametrization matrix is equal to the global one."""


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_atomwise_natural_reparametrization_class(
    real_dummy_sample_and_basis_info, orthogonalization, spin=None
):
    """Test whether the class implementation of the atomwise natural reparametrization constructs
    the correct block diagonal transformation matrix for a given sample and basis_info."""

    real_dummy_sample, real_dummy_basis_info = real_dummy_sample_and_basis_info
    atomwise_nat_rep = AtomwiseNaturalReparametrization.from_basis_info(
        real_dummy_basis_info, orthogonalization=orthogonalization, spin=spin
    )

    class_m, class_m_inv = atomwise_nat_rep.forward(real_dummy_sample.atomic_numbers)
    class_sample_m, class_sample_m_inv = atomwise_nat_rep.sample_forward(real_dummy_sample)

    # check that matrices are truly inverses of each other
    n = class_m.shape[-1]
    torch.testing.assert_close(
        class_m @ class_m_inv, torch.eye(n, dtype=class_m.dtype), rtol=1e-05, atol=1e-07
    ), """Transformation matrix M and its inverse M^-1 are not truly inverses of each other."""

    # compare forward and wrapper for OFData
    torch.testing.assert_close(
        class_m, class_sample_m, rtol=1e-05, atol=1e-07
    ), """Transformation matrices of forward and sample_forward produces different results
        for the transformation matrix M."""
    torch.testing.assert_close(
        class_m_inv, class_sample_m_inv, rtol=1e-05, atol=1e-07
    ), """Transformation matrices of forward and sample_forward produces different results
        for the inverse transformation matrix M^-1."""

    # assert that the diagonal of the transformed overlap matrix is equal to 1
    overlap_matrix = real_dummy_sample.overlap_matrix
    transformed_overlap_matrix = class_m_inv @ overlap_matrix @ class_m_inv.T
    torch.testing.assert_close(
        transformed_overlap_matrix.diagonal(),
        torch.ones_like(transformed_overlap_matrix.diagonal()),
    )


# TODO: Add tests for sparse version
@set_default_torch_dtype(torch.float64)
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
def test_atomwise_natural_reparametrization_save_load(tmp_path_factory):
    """Test saving and loading of the Natural Reparametrization matrices to/from a zarr file."""

    tmp_path = tmp_path = tmp_path_factory.getbasetemp()

    with initialize(
        version_base=None, config_path="../../configs/ml/", job_name="natural_reparametrization"
    ):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=md17subset", "paths.output_dir=" + str(tmp_path)],
        )

    datamodule = instantiate(cfg.data.datamodule)
    datamodule.setup("fit")

    dummy_sample = datamodule.train_set[0]  # fixed sample for now
    dummy_basis_info = datamodule.train_set.basis_info

    atomwise_nat_rep = AtomwiseNaturalReparametrization.from_basis_info(dummy_basis_info)

    # save the matrices
    trafo_tmp_path = tmp_path.joinpath("nat_rep_transformation_dict.zarr.zip")
    print(f"Saving to {trafo_tmp_path}")
    atomwise_nat_rep.save_dict(trafo_tmp_path)

    # load the matrices
    loaded_atomwise_nat_rep = AtomwiseNaturalReparametrization.from_file(trafo_tmp_path)

    m, m_inv = atomwise_nat_rep.sample_forward(dummy_sample)
    # from file
    m_ff, m_inv_ff = loaded_atomwise_nat_rep.sample_forward(dummy_sample)
    # compare the loaded matrices to the original ones
    torch.testing.assert_close(
        m, m_ff, rtol=1e-05, atol=1e-07
    ), """Transformation matrices of loaded and original NaturalReparametrizationTransformMatrix
        are not equal."""
    torch.testing.assert_close(
        m_inv, m_inv_ff, rtol=1e-05, atol=1e-07
    ), """Inverse transformation matrices of loaded and original NaturalReparametrizationTransformMatrix
        are not equal."""


# test shellwise natural reparameterization


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_construct_single_atom_shellwise_reparameterization_matrices(
    real_dummy_sample_and_basis_info, orthogonalization
):
    """The ‘proper maths’ tests are performed here."""
    _, basis_info = real_dummy_sample_and_basis_info

    Z = basis_info.atomic_numbers[int(basis_info.n_types / 2)]
    W = get_atomwise_overlap_matrix(Z, basis_info.basis_dict)
    eye = torch.eye(W.shape[0]).to(dtype=torch.get_default_dtype())

    for sparse in [True, False]:
        M, Minv = shellwise.construct_single_atom_shellwise_reparameterization_matrices(
            Z, basis_info, sparse=sparse, orthogonalization=orthogonalization
        )
        assert M.is_sparse == sparse
        assert Minv.is_sparse == sparse
        torch.testing.assert_close((M @ Minv).to_dense(), eye, rtol=1e-05, atol=1e-07)
        torch.testing.assert_close((M @ M.T).to_dense(), W, rtol=1e-05, atol=1e-07)


def test_shellwise_default_constructor():
    """Test whether the default constructor works."""

    shellwise_nat_rep = shellwise.ShellwiseNaturalReparameterization({})

    assert isinstance(shellwise_nat_rep, shellwise.ShellwiseNaturalReparameterization)
    assert shellwise_nat_rep.transformation_dict == {}


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def test_shellwise_from_basis_info(real_dummy_sample_and_basis_info):
    """Test whether the from_basis_info constructor works."""

    _, basis_info = real_dummy_sample_and_basis_info

    shellwise_nat_rep = shellwise.ShellwiseNaturalReparameterization.from_basis_info(basis_info)

    assert isinstance(shellwise_nat_rep, shellwise.ShellwiseNaturalReparameterization)
    assert all(isinstance(key, int) for key in list(shellwise_nat_rep.transformation_dict.keys()))
    assert all(
        isinstance(value, tuple) for value in shellwise_nat_rep.transformation_dict.values()
    )


@set_default_torch_dtype(torch.float64)
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def test_shellwise_write_load(tmp_path_factory, real_dummy_sample_and_basis_info):
    """Test the write method and the from file constructor for the shellwise natural
    reparameterization."""

    _, basis_info = real_dummy_sample_and_basis_info

    shellwise_nat_rep = shellwise.ShellwiseNaturalReparameterization.from_basis_info(basis_info)

    tmp_path = tmp_path_factory.getbasetemp().joinpath(
        "shellwise_nat_rep_transformation_dict.zarr.zip"
    )

    shellwise_nat_rep.save_dict(tmp_path)

    loaded_shellwise_nat_rep = shellwise.ShellwiseNaturalReparameterization.from_file(tmp_path)

    assert isinstance(loaded_shellwise_nat_rep, shellwise.ShellwiseNaturalReparameterization)
    for Z in shellwise_nat_rep.transformation_dict.keys():
        torch.testing.assert_close(
            shellwise_nat_rep.transformation_dict[Z],
            loaded_shellwise_nat_rep.transformation_dict[Z],
            rtol=1e-05,
            atol=1e-07,
        )


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
def test_shellwise_forward(real_dummy_sample_and_basis_info):
    """Test whether the forward method of the shellwise natural reparameterization works."""

    sample, basis_info = real_dummy_sample_and_basis_info
    for sparse in (False, True):
        shellwise_nat_rep = shellwise.ShellwiseNaturalReparameterization.from_basis_info(
            basis_info, sparse=sparse
        )

        M, Minv = shellwise_nat_rep.sample_forward(sample)

        assert M.is_sparse == sparse
        assert Minv.is_sparse == sparse

        torch.testing.assert_close(
            (M @ Minv).to_dense(), torch.eye(M.shape[0]), rtol=1e-05, atol=1e-07
        )

        # one possible additional test: check if M @ M.T is truly block diagonal
