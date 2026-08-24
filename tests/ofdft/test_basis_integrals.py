import itertools

import numpy as np
import pytest
import torch
from loguru import logger
from pyscf import dft, gto

from mldft.ofdft.basis_integrals import (
    _append_flat_gaussian,
    _get_one_center_integral,
    generate_coulomb_gradient_basisfunction_matrix,
    generate_overlap_nuclear_gradient_tensor,
    generate_overlap_tensor_blocks,
    get_basis_dipole,
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_gga_potential_matrix,
    get_gradient_nuclear_attraction_and_basisfunction_vector,
    get_normalization_vector,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
    get_nuclear_gradient_matrix,
    get_nuclear_gradient_vector,
    get_overlap_matrix,
    get_overlap_tensor,
    get_overlap_tensor_contraction_blocked,
    get_potential_vector,
)
from mldft.utils.molecules import build_mol_with_even_tempered_basis

NUMPY_TO_PYTEST_TOLERANCE = {"rtol": "rel", "atol": "abs"}
# thresholds for comparing analytic and numeric integrals without 1/r on grid
INTEGRATION_THRESHOLD = {"rtol": 1e-5, "atol": 1.5e-5}

# thresholds for comparing analytic and numeric integrals without 1/r on grid
INTEGRATION_THRESHOLD_PYTEST = {
    NUMPY_TO_PYTEST_TOLERANCE[key]: value for key, value in INTEGRATION_THRESHOLD.items()
}

# settings for comparing analytic and numeric integrals with 1/r on grid
INTEGRATION_THRESHOLD_R = {"rtol": 0.01, "atol": 0.15}  # 1/r is numerically less stable
INTEGRATION_THRESHOLD_R_PYTEST = {
    NUMPY_TO_PYTEST_TOLERANCE[key]: value for key, value in INTEGRATION_THRESHOLD_R.items()
}

INTEGRATION_THRESHOLD_R_GRADIENT = {
    # the error here can be reduced if the tests are performed with significantly larger grids
    "rtol": 0.01,
    "atol": 0.25,
}  # 1/r is numerically less stable especially once gradients are introduced
GRID_LEVEL_R = 2  # use less space to fit distance matrix on GPU
GRID_LEVEL_R_GRADIENT = 3  # use less space to fit distance matrix on GPU
PRUNING_SCHEME_R = dft.treutler_prune  # common dft pruning scheme
test_HCl = [
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="cc-pvdz"),
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="pc-3"),
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="6-31G(2df,p)"),
]
# make cartesian product of HCl molecules
test_molecules_rho_orbital = itertools.product(test_HCl, test_HCl)

np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)

if torch.cuda.is_available():
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
else:
    GPU_MEMORY = 0

REQUIRED_GPU_MEMORY = 14e9


def get_point_charge_potential_on_grid(charges, charges_positions, grid):
    """Compute the potential of point charges on a grid."""
    atom_pot = np.zeros_like(grid.weights)
    for Z, coords in zip(charges, charges_positions):
        atom_pot += -Z / np.linalg.norm(grid.coords - coords, axis=1)
    return atom_pot


def get_point_charge_gradient_on_grid(charges, charges_positions, grid):
    """Compute the gradient of the potential of point charges on a grid."""
    atom_pot = np.zeros((charges.size, grid.weights.size, 3))
    for i, (Z, coords) in enumerate(zip(charges, charges_positions)):
        atom_pot[i, ...] += (
            -Z
            * (grid.coords - coords)
            / ((np.linalg.norm(grid.coords - coords, axis=1)) ** 3)[:, None]
        )
    return atom_pot


def numerical_overlap_gradient(mol, epsilon=1e-6):
    unit = "bohr"
    coords = mol.atom_coords(unit=unit)
    derivative = np.zeros((mol.natm, 3, mol.nao, mol.nao), dtype=float)
    for i in range(mol.natm):
        for k in range(3):
            matrices = []
            for l, sign in enumerate([1, -1]):
                coords_tmp = coords.copy()
                coords_tmp[i][k] += sign * epsilon
                mol_tmp = mol.set_geom_(coords_tmp, unit=unit, inplace=False)

                matrices.append(mol_tmp.intor("int1e_ovlp"))

            derivative[i, k, ...] = (matrices[0] - matrices[1]) / (2 * epsilon)

    return derivative.reshape(mol.natm * 3, mol.nao, mol.nao)


def test__append_flat_gaussian(molecule_medium):
    """Check dimensions and new env entries."""
    mol = molecule_medium
    bas = mol._bas
    env = mol._env
    bas_mod, env_mod = _append_flat_gaussian(bas, env)
    assert len(bas_mod) == mol.nbas + 1  # one new basis
    assert len(env_mod) == len(mol._env) + 2  # two new coefficients
    assert env_mod[bas_mod[-1, 5]] == 0  # exponent zero
    assert env_mod[bas_mod[-1, 6]] == 1  # coefficient one


def test_get_potential_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare numeric integrals using random potential.

    The integral over the atomic orbitals is computed without einsum and checked. The value of the
    random seed is logged to allow for reproducing the test.
    """
    mol, ao, grid = molecule_ao_grid_level_5
    bit_generator = np.random.PCG64()
    logger.info(f"bit generator state: {bit_generator.state}")
    generator = np.random.Generator(bit_generator)
    potential = generator.standard_normal(grid.weights.shape)

    if ao.ndim > 2:
        ao = ao[0]
    num_integral = get_potential_vector(ao, potential, grid)
    num_integral_2 = (potential * grid.weights) @ ao
    assert num_integral.shape == (mol.nao,)
    assert np.allclose(num_integral_2, num_integral, **INTEGRATION_THRESHOLD)


def test_get_gga_potential_matrix(molecule_ao_grid_level_5):
    """Check dimensions and correct type.

    The value of the random seed is logged to allow for reproducing the test.
    """
    mol, ao, grid = molecule_ao_grid_level_5
    bit_generator = np.random.PCG64()
    logger.info(f"bit generator state: {bit_generator.state}")
    generator = np.random.Generator(bit_generator)
    potential = generator.standard_normal(grid.weights.shape)

    matrix = get_gga_potential_matrix(ao, potential, grid)

    assert matrix.shape == (mol.nao, mol.nao)
    assert np.issubdtype(matrix.dtype, np.floating)
    assert np.all(np.isfinite(matrix))


def test_get_normalization_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_normalization_vector(mol)
    num_integral = get_potential_vector(ao, np.ones_like(grid.weights), grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test__get_one_center_integral(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = _get_one_center_integral(mol, "int1e_ovlp")
    num_integral = get_potential_vector(ao, np.ones_like(grid.weights), grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_attraction_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_attraction_vector(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_potential_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    num_integral = get_potential_vector(ao, atom_pot, grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_gradient_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_gradient_vector(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_gradient_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    tmp_num_integral = np.einsum("ni,znx,n->zxi", ao, atom_pot, grid.weights, optimize=True)

    num_integral = tmp_num_integral.reshape(3 * mol.natm, mol.nao, order="C")

    assert integral.shape == (3 * mol.natm, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_gradient_basisfunction_vector(molecule_ao_grid_level_5):
    mol, ao, grid = molecule_ao_grid_level_5

    atom_pot = get_point_charge_potential_on_grid(mol.atom_charges(), mol.atom_coords(), grid)
    gradient_atom_pot = get_point_charge_gradient_on_grid(
        mol.atom_charges(), mol.atom_coords(), grid
    )

    slices_by_atoms = mol.aoslice_by_atom()
    atom_pot_gradient_basisfunction = np.zeros((3 * mol.natm, mol.nao))
    for atom_id in range(mol.natm):
        _, _, ao_start, ao_end = slices_by_atoms[atom_id]

        gradient_basisfunctions_atom = np.zeros((3, mol.nao), dtype=float)
        gradient_basisfunctions_atom[:, ao_start:ao_end] = np.einsum(
            "xni,n,n->xi",
            -1 * ao[1:, :, ao_start:ao_end],
            atom_pot,
            grid.weights,
            optimize=True,
        )

        atom_pot_gradient_basisfunction[3 * atom_id : 3 * atom_id + 3, :] = (
            gradient_basisfunctions_atom
        )

    analytical_atom_pot_gradient_basisfunction = (
        get_gradient_nuclear_attraction_and_basisfunction_vector(mol)
    )

    assert np.allclose(
        analytical_atom_pot_gradient_basisfunction,
        atom_pot_gradient_basisfunction,
        **INTEGRATION_THRESHOLD,
    )


def test_get_nuclear_attraction_matrix(molecule_ao_grid_level_5):
    """Check dimensions, symmetry and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_attraction_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_potential_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    num_integral = np.einsum("ni,nj,n,n->ij", ao, ao, atom_pot, grid.weights, optimize=True)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)
    assert np.allclose(
        num_integral, integral, **INTEGRATION_THRESHOLD
    ), f"{np.abs(num_integral - integral).max()}"


def test_get_nuclear_gradient_matrix(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_gradient_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_gradient_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    tmp_num_integral = np.einsum(
        "ni,znx,nj,n->zxij", ao, atom_pot, ao, grid.weights, optimize=True
    )

    num_integral = tmp_num_integral.reshape(3 * mol.natm, mol.nao, mol.nao, order="C")

    assert integral.shape == (3 * mol.natm, mol.nao, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_overlap_matrix(molecule_ao_grid_level_5):
    """Check dimensions, symmetry and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_overlap_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]
    num_integral = np.einsum("ni,nj,n->ij", ao, ao, grid.weights, optimize=True)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_generate_overlap_nuclear_gradient_tensor(molecule_medium):
    """Check dimensions, symmetry and compare to numeric integral."""
    mol = molecule_medium

    numerical_tensor = numerical_overlap_gradient(mol)

    for i, analytical_tensor_partial in enumerate(generate_overlap_nuclear_gradient_tensor(mol)):
        assert (
            pytest.approx(numerical_tensor[i : i + 1], **INTEGRATION_THRESHOLD_PYTEST)
            == analytical_tensor_partial
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.skipif(
    GPU_MEMORY < REQUIRED_GPU_MEMORY,
    reason=f"Not enough GPU memory available (at least {REQUIRED_GPU_MEMORY/1e9:.1f}GB required,"
    f" {GPU_MEMORY/1e9:.1f}GB available).",
)
def test_get_coulomb_matrix(molecule_medium):
    """Check dimensions, symmetry and compare to numeric integral.

    A comparison to numerical integrals using cuda. Test is only executed if cuda is available with
    at enough memory.
    """
    mol = molecule_medium
    integral = get_coulomb_matrix(mol)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.cuda.empty_cache()
    # convert numpy arrays to tensor and move to device
    integral = torch.from_numpy(integral).float().to(device)
    ao = torch.from_numpy(ao).float().to(device)
    coords = torch.from_numpy(grid.coords).float().to(device)
    weights = torch.from_numpy(grid.weights).float().to(device)

    # compute potential using inplace operations to save memory
    potential = torch.linalg.vector_norm(coords[:, None] - coords, dim=2)
    # divide safely
    nonzero_mask = potential != 0
    torch.div(1, potential, out=potential)
    potential[~nonzero_mask] = 0

    potential = torch.einsum("n,nm,m->nm", weights, potential, weights)
    # num_integral = torch.einsum("ni,mj,nm->ij", ao, ao, potential)
    left_contraction = torch.einsum("ni,nm->im", ao, potential)
    num_integral = torch.einsum("im,mj->ij", left_contraction, ao)

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert torch.allclose(num_integral, integral, **INTEGRATION_THRESHOLD_R)


# This requires relatively large grids to be even remotely accurate, thus we need to
# restrict ourselves to small molecules for the standard tests. We could discuss adding
# an additional test for medium sized molecules and larger grids which is no run as standard
@pytest.mark.slow
def test_get_coulomb_gradient_basisfunction_matrix_generator(molecule_small):
    mol = molecule_small

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol)
    grid.level = GRID_LEVEL_R_GRADIENT
    grid.prune = PRUNING_SCHEME_R
    grid.build()
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.cuda.empty_cache()
    # convert numpy arrays to tensor and move to device
    ao = torch.from_numpy(ao).float().to(device)
    coords = torch.from_numpy(grid.coords).float().to(device)
    weights = torch.from_numpy(grid.weights).float().to(device)

    # compute potential using inplace operations to save memory
    potential = torch.linalg.vector_norm(coords[:, None] - coords, dim=2)
    # divide safely
    nonzero_mask = potential != 0
    torch.div(1, potential, out=potential)
    potential[~nonzero_mask] = 0

    potential = torch.einsum("n,nm,m->nm", weights, potential, weights)
    # num_integral = torch.einsum("ni,mj,nm->ij", ao, ao, potential)
    ao_slices = mol.aoslice_by_atom()

    full_num_integral = torch.zeros((3 * mol.natm, mol.nao, mol.nao), device=device)
    for atom_id in range(mol.natm):
        shell_start, shell_end, ao_start, ao_end = ao_slices[atom_id]
        left_contraction = torch.einsum("xni,nm->xim", -1 * ao[1:, :, ao_start:ao_end], potential)
        num_integral = torch.einsum("xim,mj->xij", left_contraction, ao[0])

        left_contraction_2 = torch.einsum("ni,nm->im", ao[0], potential)
        num_integral_2 = torch.einsum(
            "im,xmj->xij", left_contraction_2, -1 * ao[1:, :, ao_start:ao_end]
        )

        full_num_integral[3 * atom_id : 3 * atom_id + 3, ao_start:ao_end, :] += num_integral
        # full_num_integral[
        #     3 * atom_id : 3 * atom_id + 3, :, ao_start:ao_end
        # ] += num_integral.swapaxes(1, 2)
        full_num_integral[3 * atom_id : 3 * atom_id + 3, :, ao_start:ao_end] += num_integral_2

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    for i, analytical_matrix in enumerate(generate_coulomb_gradient_basisfunction_matrix(mol)):
        assert np.allclose(
            full_num_integral[i : i + 1],
            analytical_matrix,
            **INTEGRATION_THRESHOLD_R_GRADIENT,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.skipif(
    GPU_MEMORY < REQUIRED_GPU_MEMORY,
    reason=f"Not enough GPU memory available (at least {REQUIRED_GPU_MEMORY/1e9:.1f}GB required),"
    f" {GPU_MEMORY/1e9:.1f}GB available).",
)
@pytest.mark.parametrize("mol_orbital, mol_rho", test_molecules_rho_orbital)
def test_get_coulomb_tensor(mol_orbital: gto.Mole, mol_rho: gto.Mole):
    """Check dimensions, symmetry and compare to numerical integral.

    A comparison to numerical integrals using cuda. Test is only executed if cuda is available with
    at enough memory.
    """
    integral = get_coulomb_tensor(mol_rho, mol_orbital)
    assert integral.shape == (mol_rho.nao, mol_orbital.nao, mol_orbital.nao)
    assert np.allclose(integral.transpose(0, 2, 1), integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol_rho)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()

    ao_rho = dft.numint.eval_ao(mol_rho, grid.coords, deriv=0)
    ao_orbital = dft.numint.eval_ao(mol_orbital, grid.coords, deriv=0)
    ao_orbital_matrix = np.einsum("nj,nk->njk", ao_orbital, ao_orbital)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # convert numpy arrays to tensor and move to device
    integral = torch.from_numpy(integral).float().to(device)
    ao_rho = torch.from_numpy(ao_rho).float().to(device)
    ao_orbital_matrix = torch.from_numpy(ao_orbital_matrix).float().to(device)
    coords = torch.from_numpy(grid.coords).float().to(device)
    weights = torch.from_numpy(grid.weights).float().to(device)

    # compute potential using inplace operations to save memory
    potential = torch.linalg.vector_norm(coords[:, None] - coords, dim=2)
    # divide safely
    nonzero_mask = potential != 0
    torch.div(1, potential, out=potential)
    potential[~nonzero_mask] = 0

    potential = torch.einsum("n,nm,m->nm", weights, potential, weights)
    # num_integral = torch.einsum("ni,mj,nm->ij", ao, ao, potential)
    left_contraction = torch.einsum("ni,nm->im", ao_rho, potential)
    num_integral = torch.einsum("im,mjk->ijk", left_contraction, ao_orbital_matrix)

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert torch.allclose(num_integral, integral, **INTEGRATION_THRESHOLD_R_GRADIENT)


def test_get_overlap_tensor(molecule_small):
    """Check dimensions, symmetry and compare to numerical integral."""
    mol_orbital = molecule_small
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital)
    integral = get_overlap_tensor(mol_rho, mol_orbital)
    assert integral.shape == (mol_rho.nao, mol_orbital.nao, mol_orbital.nao)
    assert np.allclose(integral.transpose(0, 2, 1), integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol_rho)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()

    ao_rho = dft.numint.eval_ao(mol_rho, grid.coords, deriv=0)
    ao_orbital = dft.numint.eval_ao(mol_orbital, grid.coords, deriv=0)

    num_integral = np.einsum(
        "ni,nj,nk,n->ijk", ao_rho, ao_orbital, ao_orbital, grid.weights, optimize=True
    )

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert (
        pytest.approx(
            num_integral,
            abs=INTEGRATION_THRESHOLD_R["atol"],
            rel=INTEGRATION_THRESHOLD_R["rtol"],
        )
        == integral
    )


def test_generate_overlap_tensor_blocks_reconstructs_full_tensor(molecule_small):
    """Blocked overlap-tensor generation should reproduce the full tensor exactly."""
    mol_orbital = molecule_small
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital)
    integral = get_overlap_tensor(mol_rho, mol_orbital)
    reconstructed = np.zeros_like(integral)

    for ao_slice, block in generate_overlap_tensor_blocks(mol_rho, mol_orbital, block_size=1):
        reconstructed[:, ao_slice, :] = block

    assert np.array_equal(reconstructed, integral)


def test_generate_overlap_tensor_blocks_respects_target_bytes(molecule_small):
    """The byte target should influence heuristic shell block splitting."""
    mol_orbital = molecule_small
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital)

    tiny_target_blocks = list(generate_overlap_tensor_blocks(mol_rho, mol_orbital, target_bytes=1))
    large_target_blocks = list(
        generate_overlap_tensor_blocks(mol_rho, mol_orbital, target_bytes=10**12)
    )

    assert len(tiny_target_blocks) == mol_orbital.nbas
    assert len(large_target_blocks) == 1


def test_get_overlap_tensor_contraction_blocked_matches_full_contraction(molecule_small):
    """Blocked self-contraction should match materializing the full overlap tensor."""
    mol_orbital = molecule_small
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital)
    overlap_tensor = torch.from_numpy(get_overlap_tensor(mol_rho, mol_orbital))
    expected = torch.einsum("iab,jab->ij", overlap_tensor, overlap_tensor)

    actual = get_overlap_tensor_contraction_blocked(
        mol_rho,
        mol_orbital,
        device="cpu",
        block_size=1,
    )

    torch.testing.assert_close(actual.cpu(), expected)


def test_get_basis_dipole(molecule_ao_grid_level_5):
    """Check dimensions and compare to numerical integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_basis_dipole(mol)
    if ao.ndim > 2:
        ao = ao[0]

    num_integral = np.einsum("gi,gd,g->di", ao, grid.coords, grid.weights, optimize=True)

    assert integral.shape == (3, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)
