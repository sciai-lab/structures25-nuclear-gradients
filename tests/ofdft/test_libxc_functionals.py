from functools import partial

import numpy as np
import pytest
import torch
from pyscf import dft
from pyscf.dft import numint

from mldft.ofdft import basis_integrals, libxc_functionals
from mldft.ofdft.libxc_functionals import eval_libxc_functionals, required_derivative
from mldft.utils.coeffs_to_grid import coeffs_to_rho_and_derivatives
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import build_mol_with_even_tempered_basis

INTEGRATION_THRESHOLD = {"rel": 2e-6, "abs": 2e-6}
MATRIX_EQUIVALENCE_THRESHOLD = {"rtol": 1e-9, "atol": 1e-11}


def _matrix_based_energy_and_potential(
    coeffs: np.ndarray,
    functionals: list[str],
    ao_block: np.ndarray,
    grid_weights_block: np.ndarray,
    max_derivative: int,
):
    """Reference implementation using the old matrix-based _eval_libxc_block path."""
    num_grid_points = len(grid_weights_block)

    derivative_wrt_rho = np.zeros(num_grid_points, dtype=np.float64)
    derivative_wrt_sigma = np.zeros(num_grid_points, dtype=np.float64)
    potential_vector = np.zeros_like(coeffs)
    rho_and_needed_derivatives = coeffs_to_rho_and_derivatives(
        coeffs, ao_block, max_derivative_order=max_derivative
    )

    energies = []
    for libxc_functional_name in functionals:
        functional_energy, functional_derivatives = (
            libxc_functionals._get_energy_and_functional_derivatives(
                libxc_functional_name,
                rho_and_needed_derivatives,
                grid_weights=grid_weights_block,
            )
        )
        energies.append(functional_energy)
        derivative_wrt_rho += functional_derivatives[0]
        if len(functional_derivatives) > 1:
            derivative_wrt_sigma += functional_derivatives[1]

    potential_vector += basis_integrals.get_potential_vector(
        ao_block, derivative_wrt_rho, grid_weights=grid_weights_block
    )
    if max_derivative >= 1:
        gga_potential = basis_integrals.get_gga_potential_matrix(
            ao_block, derivative_wrt_sigma, grid_weights=grid_weights_block
        )
        potential_vector += 2 * coeffs @ gga_potential
    return energies, potential_vector


@pytest.mark.parametrize("functional_string", ["GGA_X_PBE", "PBE,PBE"])
def test_matrix_free_matches_matrix_based(functional_string, molecule_small):
    grid = dft.Grids(molecule_small)
    grid.level = 3
    grid.build()

    max_derivative = required_derivative([functional_string])
    ao = dft.numint.eval_ao(molecule_small, grid.coords, deriv=max_derivative)

    np.random.seed(0)
    coeffs = np.random.random(molecule_small.nao)

    energies, potential_vector = eval_libxc_functionals(
        coeffs=coeffs,
        functionals=[functional_string],
        grid=grid,
        ao=ao,
        max_derivative=max_derivative,
    )
    energies_ref, potential_vector_ref = _matrix_based_energy_and_potential(
        coeffs=coeffs,
        functionals=[functional_string],
        ao_block=ao,
        grid_weights_block=grid.weights,
        max_derivative=max_derivative,
    )

    np.testing.assert_allclose(energies, energies_ref, **MATRIX_EQUIVALENCE_THRESHOLD)
    np.testing.assert_allclose(
        potential_vector,
        potential_vector_ref,
        **MATRIX_EQUIVALENCE_THRESHOLD,
    )


@pytest.mark.parametrize(
    "ao,max_derivative_order,should_raise",
    [
        (
            np.random.rand(1, 10, 10),
            0,
            False,
        ),  # Valid 3D array with max_derivative_order = 0
        (
            np.random.rand(4, 10, 10),
            1,
            False,
        ),  # Valid 3D array with max_derivative_order = 1
        (
            np.random.rand(3, 10, 10),
            1,
            True,
        ),  # Invalid 3D array with max_derivative_order = 0
        (
            np.random.rand(10, 10),
            0,
            False,
        ),  # Valid 2D array with max_derivative_order = 0
        (
            np.random.rand(10, 10),
            1,
            True,
        ),  # Invalid 2D array with max_derivative_order = 1
        (np.random.rand(5, 5, 5, 5), 0, True),  # Invalid dimensionality
        (
            np.random.rand(10, 10),
            -1,
            True,
        ),  # Invalid derivative order
    ],
)
def test_check_ao_dimensions_wrt_derivatives(ao, max_derivative_order, should_raise):
    """Checks whether _check_ao_dimensions_wrt_derivatives raises correctly."""
    if should_raise:
        with pytest.raises((AssertionError, ValueError)):
            libxc_functionals._check_ao_dimensions_wrt_derivatives(ao, max_derivative_order)
    else:
        libxc_functionals._check_ao_dimensions_wrt_derivatives(ao, max_derivative_order)


@pytest.mark.parametrize(
    "kinetic_functional, xc_functional",
    [
        ("LDA_K_TF", "PBE,PBE"),
        (None, "PBE,PBE"),
    ],
)
def test_eval_libxc_functionals(molecule_medium, kinetic_functional, xc_functional):
    """Test whether a libxc functional evaluation returns the correct types.

    No exact checks on the correct evaluation of the returned functional
    """
    grid = dft.Grids(molecule_medium)
    grid.level = 1
    grid.build()

    functionals = [kinetic_functional, xc_functional]
    max_deriv = libxc_functionals.required_derivative(functionals)

    ao = dft.numint.eval_ao(molecule_medium, grid.coords, deriv=max_deriv)

    coeffs = np.random.random(molecule_medium.nao)
    energy, potential_vector = libxc_functionals.eval_libxc_functionals(
        coeffs, functionals, grid, ao
    )
    assert isinstance(energy, list)
    assert isinstance(potential_vector, np.ndarray)
    assert np.issubdtype(potential_vector.dtype, np.floating)


def gradient_finite_differences(functional, coeffs, delta=1e-6):
    """Numerically computes the gradient of a functional.

    Args:
        functional: A functional.
        coeffs: The coefficients of the Kohn-Sham orbitals.
        delta: The step size for the numerical derivative.

    Returns:
        The gradient of the functional.
    """
    grad = np.zeros_like(coeffs)
    for i in range(len(coeffs)):
        energy_p, _ = functional(coeffs + delta * np.eye(len(coeffs))[i])
        energy_m, _ = functional(coeffs - delta * np.eye(len(coeffs))[i])
        grad[i] = (energy_p - energy_m) / (2 * delta)

    return grad


@pytest.mark.parametrize("libxc_key", ["LDA_X", "GGA_X_PBE", "GGA_C_PBE"])
def test_functional_potential(molecule_small, libxc_key):
    """Tests the functional potential against a numerical gradient."""
    grid = dft.Grids(molecule_small)
    grid.level = 3
    grid.build()

    max_deriv = libxc_functionals.required_derivative([libxc_key])

    ao = dft.numint.eval_ao(molecule_small, grid.coords, deriv=max_deriv)
    functional = partial(
        libxc_functionals.eval_libxc_functionals, functionals=libxc_key, grid=grid, ao=ao
    )

    np.random.seed(0)
    coeffs = np.random.random(molecule_small.nao)
    _, potential_vector = functional(coeffs)
    numerical_gradient = gradient_finite_differences(functional, coeffs, delta=1e-8)
    assert pytest.approx(potential_vector, **INTEGRATION_THRESHOLD) == numerical_gradient


def test_functional_energies_instances(molecule_medium):
    """Tests whether the functional returns new instances of Energies.

    If no instance of Eneries has been supplied at the creation of the functional.
    """
    libxc_key = "LDA_X"
    grid = dft.Grids(molecule_medium)
    grid.level = 3
    grid.build()

    max_deriv = libxc_functionals.required_derivative([libxc_key])

    ao = dft.numint.eval_ao(molecule_medium, grid.coords, deriv=max_deriv)

    coeffs = np.random.random(molecule_medium.nao)
    energies_1, _ = libxc_functionals.eval_libxc_functionals(coeffs, libxc_key, grid, ao)
    coeffs = np.random.random(molecule_medium.nao)
    energies_2, _ = libxc_functionals.eval_libxc_functionals(coeffs, libxc_key, grid, ao)

    assert energies_1 != energies_2


@pytest.mark.parametrize(
    "libxc_code, expected",
    [
        ("LDA_K_TF", 0),
        ("GGA_C_LYP,LDA_X", 1),
        ("PBE,PBE", 1),
        ("MGGA_X_BR89,GGA_C_LYP", 2),
    ],
)
def test_xc_type_to_deriv(libxc_code, expected):
    """Tests the _xc_type_to_deriv function to produce the correct return."""
    deriv = libxc_functionals._xc_type_to_deriv(libxc_code)
    assert deriv == expected


@pytest.mark.parametrize(
    "libxc_code, expected",
    [
        ("bla", KeyError),
    ],
)
def test_xc_type_to_deriv_raises(libxc_code, expected):
    """Tests the _xc_type_to_deriv to give correct error raises."""
    with pytest.raises(expected):
        libxc_functionals._xc_type_to_deriv(libxc_code)


@pytest.mark.parametrize(
    "libxc_codes, expected",
    [
        ("GGA_C_LYP,MGGA_X_BR89", NotImplementedError),
        ("HYB_GGA_XC_B3LYP", NotImplementedError),
    ],
)
def test_check_kxc_implementation_raises(libxc_codes, expected):
    """Tests the check_kxc_implementation to give correct error raises if not in OFDFT implemented
    functionals are requested."""
    with pytest.raises(expected):
        libxc_functionals.check_kxc_implementation(libxc_codes)


@pytest.mark.parametrize(
    "libxc_codes, expected",
    [
        (["LDA_K_TF"], 0),
        (["GGA_K_VW"], 1),
        (["GGA_K_VW", "LDA_X"], 1),
        (["GGA_K_VW", "LDA_X,GGA_C_LYP"], 1),
        (["GGA_K_VW", "MGGA_X_BR89,GGA_C_LYP"], 2),
    ],
)
def test_required_derivative(libxc_codes, expected):
    """Tests the _xc_type_to_deriv function to produce the correct derivative."""
    deriv = libxc_functionals.required_derivative(libxc_codes)
    assert deriv == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ("BLYP", "B88,LYP"),
        ("PBE", "PBE,PBE"),
        ("LDA_X", "LDA_X"),
    ],
)
def test_translate_xc_code(code, expected):
    libxc_code = libxc_functionals.translate_xc_code(code)
    assert libxc_code == expected


@pytest.mark.parametrize(
    "libxc_key", ["LDA_X", "GGA_X_PBE", "GGA_C_LYP", "PBE,PBE", "LDA_X,GGA_C_LYP"]
)
def test_get_energy_and_potential(molecule_small, libxc_key):
    """Only barebone test for the get_energy_and_potential function.

    Ensures that it produces some output without raising errors. The pyscf function against which
    it is tested is effectivley the same that is used inside the function.
    """
    mol = molecule_small
    nao = mol.nao

    grid = dft.Grids(mol)
    grid.level = 5
    grid.build()

    dm = np.random.random((nao, nao))
    ni = dft.numint.NumInt()

    deriv = libxc_functionals.required_derivative(libxc_key)
    xctype = dft.numint.libxc.xc_type(libxc_key)
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=deriv)
    rho = dft.numint.eval_rho1(mol, ao, dm, xctype=xctype)

    exc_expected, vxc_expected, _, _ = ni.eval_xc(libxc_key, rho, deriv=1, spin=0)
    den = rho * grid.weights if xctype == "LDA" else rho[0] * grid.weights
    exc_expected = np.dot(exc_expected, den)
    exc, vxc = libxc_functionals._get_energy_and_functional_derivatives(libxc_key, rho, grid=grid)
    assert np.isclose(exc_expected, exc)
    assert np.allclose(vxc, vxc_expected)


gga_list = [("GGA_K_APBE", "cpu"), ("GGA_C_PBE", "cpu"), ("GGA_X_PBE", "cpu")]


@pytest.mark.parametrize("functional_string, device", gga_list)
def test_block_functional(functional_string, device, molecule_small):
    """ "Test the blockwise evaluation of libxc functionals."""
    mol = molecule_small
    grid = dft.Grids(mol)
    grid_level = 3
    grid.build()

    ao = numint.eval_ao(mol, grid.coords, deriv=1)

    mol_with_density_basis = build_mol_with_even_tempered_basis(mol)
    prune_method = "nwchem_prune"
    grid_density_basis = grid_setup(
        mol_with_density_basis, grid_level=grid_level, prune=prune_method
    )
    p = np.random.random((mol_with_density_basis.nao_nr(),))
    p = p / np.linalg.norm(p)
    max_derivative = required_derivative(functional_string)
    ao = dft.numint.eval_ao(
        mol_with_density_basis, grid_density_basis.coords, deriv=max_derivative
    )

    coeffs = p
    # Evaluate full
    energy, potential_vector_full = eval_libxc_functionals(
        coeffs=coeffs,
        functionals=[functional_string],
        grid=grid_density_basis,
        ao=ao,
        max_derivative=1,
        blockwise=False,
        mol_density_basis=mol_with_density_basis,
    )
    # Evaluate blockwise
    energy_blockwise, potential_vector_blockwise = eval_libxc_functionals(
        coeffs=coeffs,
        functionals=[functional_string],
        grid=grid_density_basis,
        ao=ao,
        max_derivative=1,
        blockwise=True,
        mol_density_basis=mol_with_density_basis,
    )
    # Compare the two results
    torch.testing.assert_close(
        torch.tensor(potential_vector_blockwise),
        torch.tensor(potential_vector_full),
        rtol=1e-10,
        atol=1e-12,
        msg=f"Mismatch for potential vector {functional_string}",
    )
    torch.testing.assert_close(
        torch.tensor(energy_blockwise),
        torch.tensor(energy),
        rtol=1e-10,
        atol=1e-12,
        msg=f"Mismatch for energy {functional_string}, energy difference {(torch.tensor(energy_blockwise, dtype=torch.float64) - torch.tensor(energy, dtype=torch.float64))[0].item()}",
    )
