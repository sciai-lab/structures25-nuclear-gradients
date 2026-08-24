r"""This module deals with libxc kinetic, exchange and correlation functionals. It implements the
evaluation of libxc functionals to take a coefficient vector as input and return the energy and
gradient/potential vector. Also contains related utility functions.

For classical functionals, it relies on the libxc library and pyscf interface for
the calculation of those quantities. The classical functionals :math:`E_{xc}[\rho]`
are evaluated on a spatial grid with the electron density :math:`\rho(r)` and its
derivatives as input. The libxc library then provides the energy density
:math:`\epsilon_{xc}(r)` and the :math:`E_{xc}` energy is then given by:

.. math::

    E_{xc}[\rho] = \int \epsilon_{xc}(r) \rho(r) \mathrm{d} r


Additionally, for the minimization of the energy the gradient of the energy functional
is needed. This is given by:

.. math::

    \nabla_p E_{xc}[\rho] = \underbrace{\int \frac{\delta E_{xc}[\rho]}{\delta
    \rho(r)} \nabla_p \rho(r) \mathrm{d}r}_{\nabla_p E_{LDA}} +
    \underbrace{\int \frac{\delta E_{xc}[\rho]}{\delta \sigma(r)}
    \nabla_p \sigma(r) \mathrm{d}r}_{\nabla_p E_{GGA}}

with :math:`\nabla_p E_{LDA}` given by:

.. math::
    (\nabla_p E_{LDA})_\mu = \int v_\rho (r) \omega_\mu(r) \mathrm{d}r

where :math:`\nabla_p E_{GGA}` can then be rewritten as:

.. math::

        (\nabla_p E_{GGA})_\mu =& \sum_\nu 2 p_\nu \underbrace{\int v_\sigma(r)
        \nabla\omega_\mu(r)\nabla\omega_\nu(r)\mathrm{d}r}_{(v_\sigma)_{\mu,\nu}}\\
        \nabla_p E_{GGA} =& 2\mathbf{v_\sigma} \mathbf{p}


ML functionals are currently not implemented but the interface is designed to
facilitate their implementation in the future.
"""

from typing import Callable

import numpy as np
import torch
from loguru import logger
from pyscf import dft, lib

from mldft.ofdft import basis_integrals
from mldft.utils.coeffs_to_grid import coeffs_to_rho_and_derivatives
from mldft.utils.grids import compute_max_block_size, get_grid_blocks

# XC Alias from standard name to libxc convention
# Taken from pyscf
XC_ALIAS = {
    # Conventional name : name in XC_CODES
    "BLYP": "B88,LYP",
    "BP86": "B88,P86",
    "PW91": "PW91,PW91",
    "PBE": "PBE,PBE",
    "REVPBE": "PBE_R,PBE",
    "PBESOL": "PBE_SOL,PBE_SOL",
    "PKZB": "PKZB,PKZB",
    "TPSS": "TPSS,TPSS",
    "REVTPSS": "REVTPSS,REVTPSS",
    "SCAN": "SCAN,SCAN",
    "RSCAN": "RSCAN,RSCAN",
    "R2SCAN": "R2SCAN,R2SCAN",
    "SCANL": "SCANL,SCANL",
    "R2SCANL": "R2SCANL,R2SCANL",
    "SOGGA": "SOGGA,PBE",
    "BLOC": "BLOC,TPSSLOC",
    "OLYP": "OPTX,LYP",
    "OPBE": "OPTX,PBE",
    "RPBE": "RPBE,PBE",
    "BPBE": "B88,PBE",
    "MPW91": "MPW91,PW91",
    "HFLYP": "HF,LYP",
    "HFPW92": "HF,PW_MOD",
    "SPW92": "SLATER,PW_MOD",
    "SVWN": "SLATER,VWN",
    "MS0": "MS0,REGTPSS",
    "MS1": "MS1,REGTPSS",
    "MS2": "MS2,REGTPSS",
    "MS2H": "MS2H,REGTPSS",
    "MVS": "MVS,REGTPSS",
    "MVSH": "MVSH,REGTPSS",
    "SOGGA11": "SOGGA11,SOGGA11",
    "SOGGA11_X": "SOGGA11_X,SOGGA11_X",
    "KT1": "KT1,VWN",
    "KT2": "GGA_XC_KT2",
    "KT3": "GGA_XC_KT3",
    "DLDF": "DLDF,DLDF",
    "GAM": "GAM,GAM",
    "M06_L": "M06_L,M06_L",
    "M06_SX": "M06_SX,M06_SX",
    "M11_L": "M11_L,M11_L",
    "MN12_L": "MN12_L,MN12_L",
    "MN15_L": "MN15_L,MN15_L",
    "N12": "N12,N12",
    "N12_SX": "N12_SX,N12_SX",
    "MN12_SX": "MN12_SX,MN12_SX",
    "MN15": "MN15,MN15",
    "MBEEF": "MBEEF,PBE_SOL",
    "SCAN0": "SCAN0,SCAN",
    "PBEOP": "PBE,OP_PBE",
    "BOP": "B88,OP_B88",
    # new in libxc-4.2.3
    "REVSCAN": "MGGA_X_REVSCAN,MGGA_C_REVSCAN",
    "REVSCAN_VV10": "MGGA_X_REVSCAN,MGGA_C_REVSCAN_VV10",
    "SCAN_VV10": "MGGA_X_SCAN,MGGA_C_SCAN_VV10",
    "SCAN_RVV10": "MGGA_X_SCAN,MGGA_C_SCAN_RVV10",
    "M05": "HYB_MGGA_X_M05,MGGA_C_M05",
    "M06": "HYB_MGGA_X_M06,MGGA_C_M06",
    "M05_2X": "HYB_MGGA_X_M05_2X,MGGA_C_M05_2X",
    "M06_2X": "HYB_MGGA_X_M06_2X,MGGA_C_M06_2X",
    # extra aliases
    "SOGGA11X": "SOGGA11_X",
    "M06L": "M06_L",
    "M11L": "M11_L",
    "MN12L": "MN12_L",
    "MN15L": "MN15_L",
    "N12SX": "N12_SX",
    "MN12SX": "MN12_SX",
    "M052X": "M05_2X",
    "M062X": "M06_2X",
}  # noqa: E122


def _check_ao_dimensions_wrt_derivatives(ao: np.ndarray, max_derivative_order: int):
    """Check the dimensions of the atomic orbital coefficients (ao) with respect to the specified
    maximum derivative order.

    Args:
        ao (np.ndarray): Atomic orbital coefficients.
        max_derivative_order (int): The maximum derivative order used in calculations.

    Raises:
        AssertionError: If the dimensions of the atomic orbital coefficients are not
            consistent with the specified maximum derivative order.
        ValueError: If the dimensionality of the atomic orbital coefficients is neither
            2 nor 3.
    """
    assert max_derivative_order >= 0, "no negative derivatives are defined"
    if ao.ndim == 3:
        derivative_shape = ao.shape[0]
        if max_derivative_order == 0:
            assert (
                derivative_shape >= 1
            ), f"encountered ao derivative shape: {derivative_shape} but expected >= 1"
        elif max_derivative_order == 1:
            assert (
                derivative_shape >= 4
            ), f"encountered ao derivative shape: {derivative_shape} but expected >= 4"
        elif max_derivative_order == 2:
            assert (
                derivative_shape >= 10
            ), f"encountered ao derivative shape: {derivative_shape} but expected >= 10"
        elif max_derivative_order == 3:
            assert (
                derivative_shape >= 20
            ), f"encountered ao derivative shape: {derivative_shape} but expected >= 20"
        else:
            raise NotImplementedError(
                f"max_derivative_order: {max_derivative_order} not implemented"
            )

    elif ao.ndim == 2:
        assert (
            max_derivative_order == 0
        ), f"ao is 2-dimensional, no derivatives given but {max_derivative_order} needed"
    else:
        raise ValueError(f"ao has dimension {ao.ndim} but only 2 or 3 are expected")


def _get_energy_and_functional_derivatives(
    libxc_key: str,
    rho: np.ndarray,
    grid: np.ndarray | None = None,
    grid_weights: np.ndarray | None = None,
):
    r"""Calculates the energy and functional derivatives of the given functional.

    The xc energy functional value is given by :math:`\int \epsilon_{xc} \rho(r) dr`.

    Args:
        libxc_key: libxc style key for the functional.
        rho: Shape of ((,N)) for electron density (and derivatives); where N is
             the number of grids. rho (,N) are ordered as (den, grad_x, grad_y, grad_z,
             laplacian, tau) where grad_x = d/dx den, laplacian = nabla^2 den,
             tau = 1/2(nabla f)^2.
        grid: Integration grid for functionals.

    Returns:
        tuple: The energy functional value and the functional derivatives with respect
        to (rho, sigma, laplacian, tau). See `libxc documentation
        <https://libxc.gitlab.io/>`_ for more details.
    """
    # e_libxc: functional values at each grid point != energy density
    # e_xc = dot(e_libxc * rho, weights)
    # v_libxc: functional derivatives w.r.t. rho, sigma (related to \nabla rho)
    # f_libxc: second order derivatives
    # k_libxc: third order derivatives
    e_libxc, v_libxc, f_libxc, k_libxc = dft.libxc.eval_xc(libxc_key, rho, spin=0)
    # drop density derivatives
    if grid_weights is None:
        if grid is not None and hasattr(grid, "weights"):
            grid_weights = grid.weights
        else:
            raise ValueError("Either grid or grid_weights must be provided.")

    if rho.ndim >= 2:
        rho = rho[0]
    e_libxc = np.dot(e_libxc * rho, grid_weights)

    return e_libxc, v_libxc


def _xc_type_to_deriv(code: str):
    """Gets the required derivative of a libxc_code.

    uses pyscf.dft.libxc functions to get the functional types.

    Args:
        code: libxc style code

    Raises:
        NotImplementedError: if the functional is neither LDA, GGA nor metaGGA

    Note:
        In case of MGGAs it will only work if both an exchange and correlation functional are supplied
        but not in case of MGGA_XC or single MGGA_X or MGGA_C functionals.
    """
    # get the xc_type from pyscf
    # this will raise a KeyError of the code is not part of libxc
    functional_type = dft.libxc.xc_type(code)
    functional_type = functional_type.upper()
    # Local density approximation only requires the electron density at each
    # point
    if "LDA" in functional_type:
        return 0
    # meta GGAs sometimes require the laplacian
    elif "MGGA" in functional_type:
        # NOTE: This pyscf function only works if both an X and C functional
        # are supplied, e.g. it will fail in case of MGGA_XC like functionals
        # or in case of a single MGGA_X or C functional ... though that last
        # case is not too bad as you should not calculate energies only using an
        # X or C functional
        libxc_id = dft.numint.libxc.parse_xc_name(xc_name=code)
        with_laplace = any([dft.libxc.needs_laplacian(id_) for id_ in libxc_id])
        return 2 if with_laplace else 1
    # General Gradient approximation requires the gradient of the electron
    # density at each point
    elif "GGA" in functional_type:
        return 1
    else:
        # just for safety, will likely never be called ... as all currently
        # implemented functionals in libxc are either LDA, GGA or MGGA (and
        # hybrid combinations)
        raise NotImplementedError(f"{functional_type} not supported.")


def required_derivative(libxc_codes: list[str | Callable] | str | Callable) -> int:
    """Gets the highest required derivatives for the given functionals.

    LDA functionals require no derivatives. GGAs require first derivatives, some
    meta-GGAs also require the laplacian.

    Args:
        libxc_codes: libxc functional codes or callables (Not Implemented yet)

    Returns:
        highest required derivatives

    Raises:
        NotImplementedError: Callable are not yet Implemented
        TypeError: if libxc_codes contains neither str nor Callable
    """
    if isinstance(libxc_codes, str) or callable(libxc_codes):
        libxc_codes = [libxc_codes]
    deriv = 0
    for code in libxc_codes:
        if isinstance(code, str):
            # add prefix to APBE functional to determine the required derivative
            if code == "APBE":
                code = "GGA_K_APBE"
            # map xc_type to derivatives
            required_deriv = _xc_type_to_deriv(code)
            deriv = max(deriv, required_deriv)
        elif callable(code):
            raise NotImplementedError("callable functionals are not (yet) implemented")
        elif code is None:
            pass
        else:
            raise TypeError("Only str or Callable functionals are supported")

    return deriv


def requires_grid(functionals: list[str | Callable]):
    """Check if any of the given functionals require a spatial grid for calculation.

    Args:
        functionals (List[str or Callable]): A list of functionals, where each element
            can be either a string representing a libxc functional or a callable ML
            functional.

    Returns:
        bool: True if at least one functional requires a spatial grid, False otherwise.
    """
    requires_grid_ = False
    for functional in functionals:
        if isinstance(functional, str):
            requires_grid_ = True
        elif isinstance(functional, torch.nn.Module):
            pass
        elif isinstance(functional, Callable):
            pass
        else:
            raise NotImplementedError

    return requires_grid_


def check_kxc_implementation(
    code: str | Callable,
) -> None:
    """Checks whether the XC or kinetic functional is implemented in the OFDFT framework.

    Certain functionals are not available in an OFDFT framework. Hybrid
    functionals are impossible as exact exchange is not available. Furthermore,
    meta-GGA functionals using KS kinetic energy density are also unavailable.

    Args:
        code: libxc_code or callable

    Raises:
        NotImplementedError: If libxc_codes contains a hybrid functional
        NotImplementedError: If libxc_codes contains a nlc functional
        NotImplementedError: If libxc_codes contains a meta-GGA functional
        NotImplementedError: If libxc_codes is a Callable
        TypeError: If an element in libxc_codes is neither str nor callable
    """
    if isinstance(code, str):
        # Hybrid functionals can not be supported via OFDFT
        if dft.numint.libxc.is_hybrid_xc(code):
            raise NotImplementedError(
                f"hybrid functionals such as {code} are not supported in OFDFT"
            )
        # Don't know which functionals fall into this category
        # but does not seem to be standard ones ...
        # might be only for backwards compatibility with older versions of
        # libxc
        elif dft.numint.libxc.is_nlc(code):
            raise NotImplementedError(
                f"nlc functionals such as {code} are currently not supported"
            )
        # metaGGAs require the KS kinetic energy density, so they can not be
        # used at the moment but maybe some day with the right ML model ...
        elif dft.numint.libxc.is_meta_gga(code):
            raise NotImplementedError(
                f"mGGA functionals such as {code} are currently not supported"
            )
        else:
            pass
    elif callable(code):
        raise NotImplementedError("Callable K or XC functionals are not yet supported")
    else:
        raise TypeError(
            f"Functionals need to be either a (libxc) string or Callable not {type(code)}"
        )


def translate_xc_code(code: str) -> str:
    """Translates some standard XC Functional names into their libxc codes.

    Args:
        code: XC Functional name (such as BLYP, PBE, ...)
    Returns:
        libxc code if translation is available, else the input code
    """
    try:
        return XC_ALIAS[code]
    except KeyError:
        return code


def translate_check_xc(kxc_code: str | Callable):
    """Translates (if applicable) a kxc_code to libxc, check its implementation.

    Args:
        kxc_code: the kinetic or xc energy functional

    Returns:
        translated energy functional

    Raises:
        NotImplementedError: If kxc_code contains a hybrid functional
        NotImplementedError: If kxc_code contains a nlc functional
        NotImplementedError: If kxc_code contains a meta-GGA functional
        NotImplementedError: If kxc_code is a Callable
        TypeError: If an element in kxc_code is neither str nor callable
    """
    kxc_code = translate_xc_code(kxc_code)
    check_kxc_implementation(kxc_code)
    return kxc_code


def eval_libxc_functionals(
    coeffs: np.ndarray,
    functionals: str | list[str],
    grid: np.ndarray | None,
    ao: np.ndarray | None,
    max_derivative: int | None = None,
    blockwise: bool = False,
    max_memory: float = 4000.0,
    mol_density_basis=None,
) -> tuple[float | list[float], np.ndarray]:
    """Evaluate libxc functionals.

    Args:
        coeffs: Coefficients used in the functional calculation.
        functionals (list[str]): A list of strings representing libxc functionals.
        ao (np.ndarray or None): Atomic orbital values on the grid. Set to None if
            blockwise=True.
        grid (np.ndarray or None): Spatial grid for the calculation. Set to None if
            not applicable.
        max_derivative (int): The maximum derivative order used in calculations. Will
            be calculated if not given.
        blockwise (bool): If True, uses blockwise computation for the functional.
        max_memory (float): The maximum amount of memory to use for the calculation.
    Returns:
        Tuple of energies and the sum of the gradients. If only one functional is
        given, the energies are returned as a float, otherwise as a list of floats.
    """
    if isinstance(functionals, str):
        energies, grad = eval_libxc_functionals(
            coeffs,
            [functionals],
            grid,
            ao,
            max_derivative,
            blockwise,
            max_memory,
            mol_density_basis,
        )
        return energies[0], grad
    if max_derivative is None:
        max_derivative = required_derivative(functionals)

    if blockwise:
        assert (
            grid is not None and mol_density_basis is not None
        ), "Grid and mol_density_basis must be provided for blockwise evaluation."
        nao = mol_density_basis.nao_nr()
        max_block_size = compute_max_block_size(max_memory, nao, heuristic_multiplier=8)
        grid_chunks = get_grid_blocks(grid.size, max_block_size)
        logger.trace(
            f"Computing functionals in {len(grid_chunks)} grid blocks of size {max_block_size}"
        )
        energies_total = [0.0] * len(functionals)
        potential_vector_total = np.zeros_like(coeffs)
        for block_start, block_end in grid_chunks:
            ao_block = dft.numint.eval_ao(
                mol_density_basis, grid.coords[block_start:block_end], deriv=1
            )
            grid_weights_block = grid.weights[block_start:block_end]
            block_slice = slice(block_start, block_end)
            energies, potential_vector = _eval_libxc_block(
                coeffs, functionals, grid_weights_block, ao_block, max_derivative
            )
            for i, energy in enumerate(energies):
                energies_total[i] += energy
            potential_vector_total += potential_vector
        return energies_total, potential_vector_total
    else:
        assert (
            ao is not None and grid is not None
        ), "AO and Grid must be provided for non-blockwise evaluation."
        _check_ao_dimensions_wrt_derivatives(ao, max_derivative)
        energies, potential_vector = _eval_libxc_block(
            coeffs, functionals, grid.weights, ao, max_derivative
        )
        return energies, potential_vector


def _eval_libxc_block(
    coeffs: np.ndarray,
    functionals: str | list[str],
    grid_weights_block: np.ndarray | None,
    ao_block: np.ndarray | None,
    max_derivative: int | None = None,
) -> tuple[float | list[float], np.ndarray]:
    """Evaluate libxc functionals for a single block of the grid. If blockwise=False then
    block_slice=slice(None) and the entire grid is selected.

    Args:
        coeffs: Coefficients used in the functional calculation.
        functionals: List of libxc functionals to evaluate.
        grid: The full grid object.
        ao_block: Atomic orbital values for the current block.
        block_slice: Slice object for the current block.
        max_derivative: Maximum derivative order required.
    Returns:
        Tuple of energies and potential vector(s) for the block.
    """

    num_grid_points = len(grid_weights_block)

    derivative_wrt_rho = np.zeros(num_grid_points, dtype=np.float64)
    derivative_wrt_sigma = np.zeros(num_grid_points, dtype=np.float64)
    potential_vector = np.zeros_like(coeffs)
    rho_and_needed_derivatives = coeffs_to_rho_and_derivatives(
        coeffs, ao_block, max_derivative_order=max_derivative
    )

    energies = []
    # Loop over functionals and compute energy and derivatives
    for libxc_functional_name in functionals:
        ret = _get_energy_and_functional_derivatives(
            libxc_functional_name,
            rho_and_needed_derivatives,
            grid_weights=grid_weights_block,
        )
        functional_energy, functional_derivatives = ret
        energies.append(functional_energy)

        # Update derivatives
        # functional_derivatives is a list containing up to 4 functional derivatives
        # depending on the functional type. The first element is always the derivative
        # w.r.t. the electron density rho. The second element is the derivative w.r.t.
        # the sigma function (sigma = nabla rho nabla rho) and the third and fourth
        # element are the derivatives w.r.t. the laplacian and kinetic energy density.
        # LDA functionals only have the first element,
        # GGA functionals have the first two elements
        # meta-GGA functionals have either three or all four elements
        derivative_wrt_rho += functional_derivatives[0]

        # The gradient of GGA functionals additionally contains a term
        # depending on the gradient w.r.t to the coefficients of the sigma
        # function (sigma = nabla rho nabla rho)
        # this part of the gradient is then given by:
        # \int \delta E[\rho] / \delta sigma(r) \nabla_p \sigma(r) dr
        # where p is the index of the coefficient
        # After some math this can be written as:
        # p @ \int \delta E[\rho] / \delta sigma(r) \nabla \omega_mu(r) \nabla \omega_nu(r) \ dr
        # with p the coefficient index vector and \nabla \omega_mu(r) the gradient of the
        # atomic orbital w.r.t. the spatial coordinates
        if len(functional_derivatives) > 1:
            derivative_wrt_sigma += functional_derivatives[1]

    potential_vector += basis_integrals.get_potential_vector(
        ao_block, derivative_wrt_rho, grid_weights=grid_weights_block
    )
    if max_derivative >= 1:
        potential_vector += basis_integrals.get_gga_potential_vector(
            ao_block, coeffs, derivative_wrt_sigma, grid_weights_block
        )
    return energies, potential_vector


NBINS = 100


def nr_rks(ni, mol, grids, xc_code, dms, hermi=1, max_memory=2000):
    """Calculate RKS XC functional on given meshgrids for a set of density matrices. Adapted from
    pyscf to only compute the energy and not the gradient.

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array or a list of 2D arrays
            Density matrix or multiple density matrices
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        excsum: the XC functional value.
    """
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    excsum = np.zeros(nset)
    ao_deriv = 0 if xctype == "LDA" else 1
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        for i in range(nset):
            rho = make_rho(i, ao, mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
            if xctype == "LDA":
                den = rho * weight
            else:
                den = rho[0] * weight
            excsum[i] += np.dot(den, exc)
    if nset == 1:
        excsum = excsum[0]
    return excsum


def nr_uks(ni, mol, grids, xc_code, dms, hermi=1, max_memory=20000):
    """Calculate UKS XC functional and potential matrix on given meshgrids for a set of density
    matrices.

    Adapted from pyscf to only compute the energy and not the gradient.

    Args:
        ni: PySCF numerical integrator instance.
        mol: Molecule, an instance of :class:`Mole`.
        grids: Grid specification, an instance of :class:`Grids`. ``grids.coords`` and
            ``grids.weights`` are used for the mesh points and integration weights.
        xc_code: XC functional description. See :func:`parse_xc` of ``pyscf/dft/libxc.py`` for
            more details.
        dms: Density matrices stored as ``(alpha, alpha, ..., beta, beta, ...)``.
        hermi: Whether the input density matrices are symmetric. This also indicates whether the
            returned potential matrices are symmetric.
        max_memory: Maximum cache size in MB.

    Returns:
        The XC functional value ``excsum``.
    """
    xctype = ni._xc_type(xc_code)
    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    excsum = np.zeros(nset)

    ao_deriv = 0 if xctype == "LDA" else 1
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory
    ):
        for i in range(nset):
            rho_a = make_rhoa(i, ao, mask, xctype)
            rho_b = make_rhob(i, ao, mask, xctype)
            rho = (rho_a, rho_b)
            exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
            if xctype == "LDA":
                den_a = rho_a * weight
                den_b = rho_b * weight
            else:
                den_a = rho_a[0] * weight
                den_b = rho_b[0] * weight
            excsum[i] += np.dot(den_a, exc)
            excsum[i] += np.dot(den_b, exc)

    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        excsum = excsum[0]

    return excsum


def _format_uks_dm(dms):
    if isinstance(dms, np.ndarray) and dms.ndim == 2:  # RHF DM
        dma = dmb = dms * 0.5
    else:
        dma, dmb = dms
    if getattr(dms, "mo_coeff", None) is not None:
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if mo_coeff[0].ndim < dma.ndim:  # handle ROKS
            mo_occa = np.array(mo_occ > 0, dtype=np.double)
            mo_occb = np.array(mo_occ == 2, dtype=np.double)
            dma = lib.tag_array(dma, mo_coeff=mo_coeff, mo_occ=mo_occa)
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff, mo_occ=mo_occb)
        else:
            dma = lib.tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    return dma, dmb
