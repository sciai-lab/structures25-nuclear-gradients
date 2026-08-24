"""Compute analytic integrals over contracted Gaussian-type orbitals (GTO).

This module uses pyscf which in turn uses libcint to compute analytic
integrals. Basis functions are usually separated into radial and
spherical parts,

.. math::

        \\omega_\\mu(r, \\theta, \\phi) = f_{l_\\mu}(r) Y_{l_\\mu}^{m_\\mu} (\\theta, \\phi),

where :math:`Y_l^m` are spherical harmonics. The radial part of contracted
Gaussian-type orbitals are given by a monomial times a linear combination of
Gaussians,

.. math::

        f_l(r) = r^l \\sum_{i=1}^N c_i \\exp(-\\zeta_i r^2).

The :math:`c_i` are called contraction coefficients and the :math:`\\zeta_i` scale the
width of the Gaussians. :math:`N` is the number of primitives. Note that also the
nucleus can be modeled using GTOs (:code:`nuc_model=1` is point particle).

`pyscf` molecules have three layers of basis data (see
https://pyscf.org/develop/gto_developer.html):

*   input, e.g. :code:`.atom`, :code:`.basis`
*   internal, e.g. :code:`._atom`, :code:`._base`
*   for :code:`libcint`, e.g. :code:`._atm`, :code:`._bas`, :code:`._env`

The ._env array stores all relevant information about the basis set,
including the atom positions, the exponents and contraction coefficients.
The properties :code:`._atm` and :code:`._bas` are used to store the indices of the :code:`._env`
array. They contain the following information (0 are entries not relevant for
us):

*   :code:`_atm`: charge, ind_of_coord, nuc_model, ind_nuc_zeta, 0, 0
*   :code:`_bas`: atom_id, angular_momentum, num_primitives, num_contracted_gtos, 0, ind_exp, ind_contraction_coeff, 0
"""

from functools import lru_cache
from typing import Generator, Tuple

import numpy as np
import torch
from pyscf import gto
from tqdm import tqdm

from mldft.ofdft import OFDFT_DTYPE

# correction factor for integrals containing the flat Gaussian (empirical)
FLAT_GAUSSIAN_CORRECTION = 2 * np.sqrt(np.pi)


def _append_flat_gaussian(bas, env) -> Tuple[np.ndarray, np.ndarray]:
    """Append a flat Gaussian to the basis set.

    The added flat Gaussian (constant function with value 1) is needed to
    compute integrals using pyscf.

    Args:
        bas: The basis array as used by libcint.
        env: The environment array as used by libcint.

    Returns:
        bas_mod: The modified basis array.
        env_mod: The modified environment array.
    """
    # add exponent 0 and contraction coefficient 1
    env_mod = np.append(env, [0, 1])

    ind_exp = env_mod.size - 2
    ind_ctrc_coef = env_mod.size - 1

    bas_mod = np.zeros((bas.shape[0] + 1, bas.shape[1]), dtype=np.int32)
    bas_mod[:-1, :] = bas
    bas_mod[-1] = [0, 0, 1, 1, 0, ind_exp, ind_ctrc_coef, 0]

    return bas_mod, env_mod


def _get_one_center_integral(mol: gto.Mole, intor_name: str, components: int = 1) -> np.ndarray:
    """Compute 1-center integrals.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.
        intor_name: The name of the integral to compute. Partial list at
            `pyscf <https://pyscf.org/pyscf_api_docs/pyscf.gto.html#module-pyscf.gto.moleintor>`_.
            More options in the c++ library at
            `libcint <https://github.com/sunqm/libcint/blob/master/scripts/auto_intor.cl>`_.
        components: The number of components of the integral. For example, the
            dipole moment has three components.

    Returns:
        Vector of integrals.
    """

    atm = [mol._atm]
    bas = [mol._bas]
    env = [mol._env]

    denominators = [
        1,
    ]

    if components > 1:
        total_integral = np.zeros((components, mol.nao), dtype=np.float64)
    else:
        total_integral = np.zeros(mol.nao, dtype=np.float64)

    for i, denominator in enumerate(denominators):
        local_atm = atm[i]
        local_bas = bas[i]
        local_env = env[i]

        local_bas, local_env = _append_flat_gaussian(local_bas, local_env)
        # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
        intor_name = mol._add_suffix(intor_name)

        nbas = local_bas.shape[0]

        # shell slice: choose indices of the resulting tensor,
        # in this case of the matrix <\omega_i | [operator] | \omega_j>
        # shls_slice = (ish_start, ish_end, jsh_start, jsh_end)
        # we want the true basis functions as bra and the flat gaussian as ket
        shls_slice = (0, nbas - 1, nbas - 1, nbas)

        integral = gto.moleintor.getints2c(
            intor_name,
            local_atm,
            local_bas,
            local_env,
            shls_slice=shls_slice,
            comp=components,
        )
        integral = np.squeeze(integral, axis=-1)
        integral /= denominator

        total_integral += integral

    return total_integral * FLAT_GAUSSIAN_CORRECTION


def get_normalization_vector(mol: gto.Mole, clip_minimum: float = 1.6e-15) -> np.ndarray:
    r"""Compute the integral over the basis functions.

    Compute

    .. math::

        w_\mu = \int \omega_\mu(r) \mathrm{d}r

    for all basis functions :math:`\omega_\mu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.
        clip_minimum: If not None, the integral is clipped from below at this threshold.
            This should be used to avoid numerical instabilities which can lead to larger
            errors later on.

    Returns:
        Vector of integrals.
    """
    integral = _get_one_center_integral(mol, "int1e_ovlp")
    if clip_minimum is not None:
        integral[np.abs(integral) < clip_minimum] = 0
    return integral


@lru_cache
def get_basis_dipole(mol: gto.Mole) -> np.ndarray:
    r"""Compute the basis dipole vector.

    Compute

    .. math::

        d_{i \mu} = \int \omega_\mu(r) r_i \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and coordinates :math:`r_i`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals. With shape (3, n_basis).
    """
    return _get_one_center_integral(mol, "int1e_r", components=3)


def get_potential_vector(
    ao: np.ndarray,
    potential: np.ndarray,
    grid: np.ndarray | None = None,
    grid_weights: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute the potential vector for given potential on the grid.

    For a given potential :math:`v_\mathrm{pot}` and for all atomic orbitals
    :math:`\omega_\mu` compute

    .. math::
        v_\mu = \sum_{i \in \mathrm{grid}} (\omega_\mu)_i (v_\mathrm{pot})_i w_i
        \approx \int \omega_\mu(r) v_\mathrm{pot}(r) \mathrm{d}r

    on the grid. The weights :math:`w_i` are taken from ``grids``.

    Args:
        ao: The atomic orbitals on the grid with size (n_grid, n_ao). If the
            atomic orbitals also contain derivatives in an array of size (:,
            n_grid, n_ao), they are ignored.
        potential: The exchange-correlation potential on the grid.
        grid: The grid on which the atomic orbitals and the
            exchange-correlation potential are defined.

    Returns:
        Vector of integrals.
    """
    if ao.ndim > 2:
        ao = ao[0]  # throw away derivatives
    if grid_weights is None:
        if grid is not None and hasattr(grid, "weights"):
            grid_weights = grid.weights
        else:
            raise ValueError("Either grid or grid_weights must be provided.")
    integral = np.einsum("ni,n,n->i", ao, potential, grid_weights)
    return integral


def get_gga_potential_matrix(
    ao: np.ndarray,
    vsigma: np.ndarray,
    grid: None | np.ndarray = None,
    grid_weights: None | np.ndarray = None,
) -> np.ndarray:
    r"""Compute the potential matrix for given GGA potential on the grid.

    Computes the potential matrix :math:`V^{(\sigma)}` from the functional derivative w.r.t.
    :math:`\sigma:=\nabla\rho(r)\nabla\rho(r)`. This is necessary for the calculation of the
    correct gradient of the K/XC functionals. Their gradient can be separated into the usual
    LDA like gradient and a GGA gradient contribution.

    .. math::
        \nabla_p E_\mathrm{XC} = \nabla_p E_\mathrm{LDA} + \nabla_p E_\mathrm{GGA}


    The GGA gradient contribution is given by

    .. math::
        \nabla_p E_\mathrm{GGA} = \int\underbrace{\frac{\delta E[\rho]}{\delta\sigma(r)}}_{
        v_\sigma(r)}\nabla_p \sigma(r)\mathrm{d}r,

    which can be rewritten as

    .. math::
        (\nabla_p E_\mathrm{GGA})_\mu &= 2 \sum_\nu p_\nu \underbrace{\int v_\sigma(
        r)\nabla\omega_\mu(r)\nabla\omega_\nu(r)\mathrm{d}r}_{V^{(\sigma)}_{\mu\nu}}, \\
        \nabla_p E_\mathrm{GGA} &= 2 V^{(\sigma)} p.

    This function computes the matrix

    .. math::

        V^{(\sigma)}_{\mu\nu} = \int v_\sigma(r)\nabla\omega_\mu(r)\nabla\omega_\nu(r)\mathrm{d}r.

    Args:
        ao: The atomic orbitals on the grid with size (n_derivative, n_grid, n_ao).
        vsigma: The functional derivative w.r.t `sigma:=\nabla rho \nabla \rho` on the grid.
        grid: The grid on which the atomic orbitals and the functional are evaluated.

    Returns:
        Matrix of integrals.
    """
    assert ao.ndim > 2, "AO derivatives are required, ao dimension must be greter 2"
    assert ao.shape[0] >= 4, "x,y,z derivatives required; ao shape must be at least 4"

    if grid_weights is not None:
        weights = grid_weights
    elif grid is not None and hasattr(grid, "weights"):
        weights = grid.weights
    else:
        raise ValueError("Either grid or grid_weights must be provided.")
    grad_ao = ao[1:4]
    mat = np.einsum("pni,n,pnj", grad_ao, vsigma * weights, grad_ao)
    return mat


def get_gga_potential_vector(
    ao: np.ndarray,
    coeffs: np.ndarray,
    vsigma: np.ndarray,
    grid_weights: np.ndarray,
) -> np.ndarray:
    r"""Compute the GGA potential vector for given density coefficients.

    This function computes the product :math:`2 V^{(\sigma)} p` directly without forming
    the full potential matrix :math:`V^{(\sigma)}`, which is more efficient for large basis sets.

    The GGA gradient contribution is given by

    .. math::
        \nabla_p E_\mathrm{GGA} = \int\frac{\delta E[\rho]}{\delta\sigma(r)}\nabla_p \sigma(r)\mathrm{d}r,

    which can be computed in two equivalent ways:

    **Matrix formulation** (see :func:`get_gga_potential_matrix`):

    .. math::
        (\nabla_p E_\mathrm{GGA})_\mu &= 2 \sum_\nu p_\nu V^{(\sigma)}_{\mu\nu}, \\
        V^{(\sigma)}_{\mu\nu} &= \int v_\sigma(r)\nabla\omega_\mu(r)\cdot\nabla\omega_\nu(r)\mathrm{d}r \\
        &= \sum_{x \in \{x,y,z\}} \int v_\sigma(r)\nabla_x\omega_\mu(r)\nabla_x\omega_\nu(r)\mathrm{d}r.

    **Vector formulation** (this function):

    .. math::
        (\nabla_p E_\mathrm{GGA})_\mu &= 2 \int v_\sigma(r) \nabla\omega_\mu(r)\cdot\nabla\rho(r) \mathrm{d}r, \\
        \nabla\rho(r) &= \sum_\nu p_\nu \nabla\omega_\nu(r).

    This matrix-free formulation is mathematically equivalent to forming
    :math:`V^{(\sigma)}` and multiplying by the coefficient vector, but it avoids
    constructing the full :math:`(n_\mathrm{ao}, n_\mathrm{ao})` intermediate.
    As a result, the storage cost drops from :math:`O(n_\mathrm{ao}^2)` to
    :math:`O(n_\mathrm{ao})`, and the dominant work scales as
    :math:`O(n_\mathrm{grid} n_\mathrm{ao})` instead of
    :math:`O(n_\mathrm{grid} n_\mathrm{ao}^2)`.

    Args:
        ao: The atomic orbitals on the grid with size (n_derivative, n_grid, n_ao).
        coeffs: The density coefficients :math:`p`, shape (n_ao,).
        vsigma: The functional derivative w.r.t. :math:`\sigma:=\nabla\rho\cdot\nabla\rho` on the grid.
            Shape (n_grid,).
        grid_weights: The integration weights on the grid.
    Returns:
        The GGA potential vector :math:`v`, shape (n_ao,).
    """

    assert ao.ndim > 2, "AO derivatives are required"
    assert ao.shape[0] >= 4, "x,y,z derivatives required"
    grad_ao = ao[1:4]  # (3, ngrids, nao)

    assert vsigma.ndim == 1, "vsigma must be a vector with shape (n_grid,)"
    grad_rho = np.einsum("xni,i->xn", grad_ao, coeffs)
    weighted_vsigma = vsigma * grid_weights
    return sum(
        2 * np.einsum("n,nj->j", grad_rho[axis] * weighted_vsigma, grad_ao[axis])
        for axis in range(3)
    )


def get_nuclear_attraction_vector(mol: gto.Mole) -> np.ndarray:
    r"""Compute the nuclear attraction integral vector.

    Compute the integral over the potential of the nuclei :math:`a`

    .. math::

        (v_\mathrm{nuc})_\mu = -\int \omega_\mu(r) \sum_{a=1}^A \frac{Z_a}{\|r - r_a\|} \mathrm{d}r

    for all basis functions :math:`\omega_\mu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Vector of integrals.
    """
    return _get_one_center_integral(mol, "int1e_nuc")


def get_nuclear_gradient_vector(mol: gto.Mole) -> np.ndarray:
    r"""Compute the nuclear gradient integral vector.

    Compute the integral over the gradient of the potential of the nuclei :math:`A`

    .. math::

        \left(\nabla_{R_A}v_\mathrm{nuc}\right)_{ A_{x/y/z}\mu} = -\int \omega_\mu(r) \nabla_{R_A} \frac{Z_A}{\|r - R_A\|} \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and nuclei :math:`A`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Vector of integrals, shape (3 * natms, nao), ordered according to atom id.
    """
    intor_name = "int1e_iprinv"
    components = 3

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nuclear_gradient = np.zeros((3 * mol.natm, mol.nao), dtype=np.float64)

    for atm_id in range(mol.natm):
        with mol.with_rinv_as_nucleus(atm_id):
            atm = mol._atm
            bas = mol._bas
            env = mol._env

            # potential. Check this, the MOFDFT people seem to assume this is correct
            bas, env = _append_flat_gaussian(bas, env)

            nbas = bas.shape[0]

            # shell slice: choose indices of the resulting tensor,
            # in this case of the matrix <\omega_i | [operator] | \omega_j>
            # shls_slice = (ish_start, ish_end, jsh_start, jsh_end)
            # we want the true basis functions as bra and the flat gaussian as ket
            shls_slice = (0, nbas - 1, nbas - 1, nbas)
            integral = gto.moleintor.getints2c(
                intor_name, atm, bas, env, shls_slice=shls_slice, comp=components
            )
            integral = np.squeeze(integral, axis=-1)
            integral *= mol.atom_charge(atm_id)
            integral *= FLAT_GAUSSIAN_CORRECTION
            # we calculate the gradient w.r.t to the electron coordinates which
            # is the negative of the gradient w.r.t to the nuclear coordinates
            integral *= -1

            nuclear_gradient[3 * atm_id : 3 * atm_id + 3, :] = integral

    return nuclear_gradient


def get_gradient_nuclear_attraction_and_basisfunction_vector(
    mol: gto.Mole,
) -> np.ndarray:
    r"""Compute the gradient of the nuclear attraction integral vector.

    .. math::
        \begin{pmatrix}
            \int \pdv{\omega_1(r)}{R_1} v_{ne}(r) dr &  \int \pdv{\omega_2(r)}{R_1} v_{ne}(r) dr & \dots & \int \pdv{\omega_n(r)}{R_1} v_{ne}(r) dr \\
            \int \pdv{\omega_1(r)}{R_2} v_{ne}(r) dr &  \int \pdv{\omega_2(r)}{R_2} v_{ne}(r) dr & \dots & \int \pdv{\omega_n(r)}{R_2} v_{ne}(r) dr \\
            \vdots & \vdots & \ddots & \vdots \\
            \int \pdv{\omega_1(r)}{R_N} v_{ne}(r) dr &  \int \pdv{\omega_2(r)}{R_N} v_{ne}(r) dr & \dots & \int \pdv{\omega_n(r)}{R_N} v_{ne}(r) dr \\
        \end{pmatrix}

    It should be noted that the gradient of basis function w.r.t. the nuclear coordinates is the negative of the gradient of the basis functions w.r.t electron coordinates. Also most of these blocks will be zero as only the basisfunctions located on an atom will contribute here.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Vector of integrals, shape (3 * natms, nao), ordered according to atom id.
    """
    # < \nabla \omega_i | v_{ne} >
    intor_name = "int1e_ipnuc"
    components = 3

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nuclear_gradient = np.zeros((3 * mol.natm, mol.nao), dtype=np.float64)

    slices = mol.aoslice_by_atom()

    atm = mol._atm
    bas = mol._bas
    env = mol._env

    # potential. Check this, the MOFDFT people seem to assume this is correct
    bas, env = _append_flat_gaussian(bas, env)

    nbas = bas.shape[0]

    # shell slice: choose indices of the resulting tensor,
    # in this case of the matrix <\omega_i | [operator] | \omega_j>
    # shls_slice = (ish_start, ish_end, jsh_start, jsh_end)
    # we want the true basis functions as bra and the flat gaussian as ket
    shls_slice = (0, nbas - 1, nbas - 1, nbas)
    integral = gto.moleintor.getints2c(
        intor_name, atm, bas, env, shls_slice=shls_slice, comp=components
    )
    integral = np.squeeze(integral, axis=-1)
    # integral *= mol.atom_charge(atm_id)
    integral *= FLAT_GAUSSIAN_CORRECTION
    # we calculate the gradient w.r.t the electron coordinates which
    # is the negative of the gradient w.r.t the nuclear coordinates
    integral *= -1

    # reshape them to the correct output shape
    for atom_id in range(mol.natm):
        shell_start, shell_end, ao_start, ao_end = slices[atom_id]
        nuclear_gradient[atom_id * 3 : atom_id * 3 + 3, ao_start:ao_end] = integral[
            :, ao_start:ao_end
        ]

    return nuclear_gradient


def get_nuclear_attraction_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center nuclear attraction integral matrix.

    Compute

    .. math::

        (V_\mathrm{nuc})_{\mu\nu} = \int \eta_\mu(r) \sum_{a=1}^A  \frac{Z_a}{\|r - r_a\|}
        \eta_\nu(r) \mathrm{d} r

    for all basis functions :math:`\eta_\mu` and :math:`\eta_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int1e_nuc")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env, hermi=1)


def get_nuclear_gradient_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center nuclear gradient integral matrix.

    Compute

    .. math::

        \left(\nabla_{R_A}V_\mathrm{nuc}\right)_{A_{x/y/z}\mu\nu} = \int \omega_\mu(r) \left(\nabla_{R_A}\frac{Z_A}{\|r_1 - A_A\|} \right) \omega_\nu(r)
        \mathrm{d} r

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu` and all nuclei :math:`A`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals shape (3 * natoms, nao, nao), ordered according to atom id.
    """
    intor_name = "int1e_iprinv"
    components = 3

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nuclear_gradient = np.zeros((3 * mol.natm, mol.nao, mol.nao), dtype=np.float64)

    for atm_id in range(mol.natm):
        with mol.with_rinv_as_nucleus(atm_id):
            integral = mol.intor(intor_name, comp=components)
            integral *= -mol.atom_charge(atm_id)
            nuclear_gradient[3 * atm_id : 3 * atm_id + 3, :, :] = integral + np.swapaxes(
                integral, 1, 2
            )

    return nuclear_gradient


def get_overlap_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center overlap matrix.

    Compute

    .. math::

        W_{\mu\nu} = \int \omega_\mu(r) \omega_\nu(r) \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int1e_ovlp")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env)


def get_overlap_tensor(mol_rho: gto.Mole, mol_orbital: gto.Mole) -> np.ndarray:
    r"""Compute the 3-center overlap integral tensor between different bases.

    Compute the overlap integral over one atomic orbital :math:`\omega_\mu` of the density basis
    given by :code:`mol_rho` and the orbitals :math:`\eta_\nu` of the orbital basis given by
    :code:`mol_orbital`,

    .. math::

        \tilde L_{\mu,\alpha\beta} = \langle\omega_\mu | \eta_\alpha\eta_\beta \rangle
        = \int \omega_\mu(r) \eta_\alpha(r) \eta_\beta(r) \mathrm{d} r.

    Args:
        mol_rho: The molecule containing information about the geometry and the density basis set.
        mol_orbital: The molecule containing information about the geometry and the orbital
            basis set.

    Returns:
        Tensor of integrals with shape (n_ao_rho, n_ao_orbital, n_ao_orbital).
    """
    intor = mol_rho._add_suffix("int3c1e")
    # concatenate the environment arrays of both molecules (pyscf wants double indices first)
    atm, bas, env = gto.mole.conc_env(
        mol_orbital._atm,
        mol_orbital._bas,
        mol_orbital._env,
        mol_rho._atm,
        mol_rho._bas,
        mol_rho._env,
    )

    # first and second index run over orbital basis, third over density basis
    shls_slice = (
        0,
        mol_orbital.nbas,
        0,
        mol_orbital.nbas,
        mol_orbital.nbas,
        mol_orbital.nbas + mol_rho.nbas,
    )

    # integral = df.r_incore.aux_e2(mol_orbital, mol_rho, intor="int3c2e")
    integral = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice)
    return integral.transpose(2, 0, 1)


def _resolve_overlap_tensor_block_size(
    mol_rho: gto.Mole,
    mol_orbital: gto.Mole,
    block_size: int | None,
    target_bytes: int = 1_000_000_000,
) -> int:
    """Resolve a shell-block size for blocked overlap-tensor routines."""
    if block_size is not None:
        return max(1, min(int(block_size), mol_orbital.nbas))

    bytes_per_block_ao = np.dtype(np.float64).itemsize * mol_rho.nao * mol_orbital.nao
    estimated_block_ao = max(1, target_bytes // max(bytes_per_block_ao, 1))
    avg_ao_per_shell = max(1, int(np.ceil(mol_orbital.nao / max(mol_orbital.nbas, 1))))
    estimated_shells = max(1, estimated_block_ao // avg_ao_per_shell)
    return min(estimated_shells, mol_orbital.nbas)


def generate_overlap_tensor_blocks(
    mol_rho: gto.Mole,
    mol_orbital: gto.Mole,
    block_size: int | None = None,
    target_bytes: int = 1_000_000_000,
) -> Generator[tuple[slice, np.ndarray], None, None]:
    r"""Yield shell-blocks of the 3-center overlap tensor.

    Each yielded block contains the full second orbital index and a shell-batched
    slice of the first orbital index, matching the tensor layout returned by
    :func:`get_overlap_tensor`.

    Args:
        mol_rho: Molecule with the density basis set.
        mol_orbital: Molecule with the orbital basis set.
        block_size: Number of orbital shells to process per block. If ``None``,
            a heuristic is used.
        target_bytes: Approximate maximum block tensor size used by the
            heuristic when ``block_size`` is ``None``.

    Yields:
        Tuples ``(ao_slice, block)`` where ``block`` has shape
        ``(n_ao_rho, ao_slice.stop - ao_slice.start, n_ao_orbital)``.
    """
    intor = mol_rho._add_suffix("int3c1e")
    atm, bas, env = gto.mole.conc_env(
        mol_orbital._atm,
        mol_orbital._bas,
        mol_orbital._env,
        mol_rho._atm,
        mol_rho._bas,
        mol_rho._env,
    )

    n_shells_orbital = mol_orbital.nbas
    ao_loc = mol_orbital.ao_loc_nr()
    block_size = _resolve_overlap_tensor_block_size(
        mol_rho, mol_orbital, block_size, target_bytes=target_bytes
    )
    shell_starts = range(0, n_shells_orbital, block_size)
    iterator = (
        tqdm(shell_starts, desc="Generating overlap tensor blocks", colour="blue", leave=False)
        if len(shell_starts) > 1
        else shell_starts
    )

    for shell_start in iterator:
        shell_end = min(shell_start + block_size, n_shells_orbital)
        ao_start = int(ao_loc[shell_start])
        ao_end = int(ao_loc[shell_end])
        shls_slice = (
            shell_start,
            shell_end,
            0,
            n_shells_orbital,
            n_shells_orbital,
            n_shells_orbital + mol_rho.nbas,
        )
        block = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice)
        yield slice(ao_start, ao_end), block.transpose(2, 0, 1)


def get_overlap_tensor_contraction_blocked(
    mol_rho: gto.Mole,
    mol_orbital: gto.Mole,
    device: str | torch.device = "cpu",
    block_size: int | None = None,
    target_bytes: int = 1_000_000_000,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Compute the self-contraction of the overlap tensor block by block.

    This returns

    .. math::
        A_{ij} = \sum_{\alpha,\beta}
        \tilde L_{i,\alpha\beta}\tilde L_{j,\alpha\beta},

    without materializing the full 3-center tensor at once.

    Args:
        mol_rho: Molecule with the density basis set.
        mol_orbital: Molecule with the orbital basis set.
        device: Torch device on which to accumulate the contraction.
        block_size: Number of orbital shells to process per block. If ``None``,
            a heuristic is used.
        target_bytes: Approximate maximum block tensor size used by the
            heuristic when ``block_size`` is ``None``.
        dtype: Torch dtype used for the accumulation. Defaults to ``torch.float64``.

    Returns:
        Tensor with shape ``(n_ao_rho, n_ao_rho)`` on ``device``.
    """
    if dtype is None:
        dtype = torch.float64

    contraction = torch.zeros((mol_rho.nao, mol_rho.nao), device=device, dtype=dtype)
    for _, block in generate_overlap_tensor_blocks(
        mol_rho, mol_orbital, block_size=block_size, target_bytes=target_bytes
    ):
        block_tensor = torch.as_tensor(block, device=device, dtype=dtype)
        block_flat = block_tensor.reshape(block_tensor.shape[0], -1)
        contraction += block_flat @ block_flat.T
    return contraction


def get_L2_coulomb_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the two-electron repulsion integral matrix.

    Compute

    .. math::

        D_{\alpha,\beta,\gamma,\delta}  = (\eta^\alpha\eta^\beta|\eta^\gamma\eta^\delta)

    for all basis functions :math:`\eta^\alpha`, :math:`\eta^\beta`, :math:`\eta^\gamma`, and :math:`\eta^\delta`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.(n_basis, n_basis, n_basis, n_basis)
    """
    intor_name = mol._add_suffix("int2e")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env)


def get_coulomb_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center Coulomb integral matrix.

    Compute

    .. math::
        \tilde W_{\mu\nu} = \int\int \frac{\omega_\mu(r_1) \omega_\nu(r_2)}{r_{12}} \mathrm{d} r_1
        \mathrm{d} r_2

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int2c2e")
    # use that the matrix of integrals is symmetric (hermitian)
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env, hermi=1)


def get_coulomb_gradient_basisfunction_contraction(
    mol: gto.Mole, coeffs: np.ndarray
) -> np.ndarray:
    r"""Compute the gradient of the Hartree energy w.r.t. the nuclear coordinates.

    Contracts the 2-center Coulomb gradient integrals with the density coefficients on both
    indices, atom block by atom block, without materializing the full ``(3 * natm, nao, nao)``
    tensor:

    .. math::
        \pdv{J[\rho]}{\vb{R}_A} = \frac{1}{2} \sum_{\mu\nu} p_\mu p_\nu
        \left(\nabla_{R_A}W\right)_{A_{x/y/z}\mu\nu}

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.
        coeffs: The density coefficients p, shape ``(nao,)``.

    Returns:
        Gradient vector of shape ``(3 * natm,)``, ordered according to atom id.
    """
    intor_name = mol._add_suffix("int2c2e_ip1")
    slices = mol.aoslice_by_atom()
    gradient = np.empty(3 * mol.natm, dtype=np.float64)
    for atm_id in range(mol.natm):
        shell_start, shell_end, ao_start, ao_end = slices[atm_id]
        tensor = gto.moleintor.getints(
            intor_name,
            mol._atm,
            mol._bas,
            mol._env,
            shls_slice=(shell_start, shell_end, 0, mol.nbas),
        )
        # Contracting both indices with coeffs directly accounts for the symmetrized
        # nu-in-atom contribution, while the factor of two cancels the usual 1/2 prefactor.
        # The sign flips because the integral derivative is w.r.t. the electron coordinates.
        gradient[3 * atm_id : 3 * atm_id + 3] = -((tensor @ coeffs) @ coeffs[ao_start:ao_end])
    return gradient


def generate_overlap_nuclear_gradient_tensor(
    mol: gto.Mole, device: str | torch.device = "cpu"
) -> Generator[torch.Tensor, None, None]:
    r"""Generator which computes the nuclear gradient of the overlap matrix.

    Each of the Matrices in the first dimension (one for each atom and its x, y, z component) is the gradient of the overlap matrix with respect to the nuclear coordinates of the corresponding atom.

    .. math::
        \left( \pdv{W_{\mu\nu}}{R_{A_{x/y/z}}} \right) = \int \pdv{\omega_\mu(r) \omega_\nu(r)}{R_{A_{x/y/z}}} \mathrm{d}r

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Yields:
        Tensor of integrals with shape (1, nao, nao), yielded 3 * natoms times, ordered according to atom id.
    """
    ip_ovlp = torch.as_tensor(mol.intor("int1e_ipovlp"), dtype=OFDFT_DTYPE, device=device)

    ao_slices = mol.aoslice_by_atom()
    gradient = torch.zeros((1, mol.nao, mol.nao), dtype=ip_ovlp.dtype, device=device)

    for i in range(mol.natm):
        _, _, ao_start, ao_end = ao_slices[i]

        for k in range(3):
            # atom-i, atoms-j (i!=j) block
            gradient[0, ao_start:ao_end, :] = -1 * ip_ovlp[k, ao_start:ao_end, :]

            # atom-j, atom-i (i!=j) block
            gradient[0, :, ao_start:ao_end] = -1 * torch.movedim(
                ip_ovlp[k, ao_start:ao_end, :], -2, -1
            )
            # atom-i , atom-i block
            gradient[0, ao_start:ao_end, ao_start:ao_end] = -1 * (
                ip_ovlp[k, ao_start:ao_end, ao_start:ao_end]
                + torch.movedim(ip_ovlp[k, ao_start:ao_end, ao_start:ao_end], -2, -1)
            )

            # the rest is zero anyway ...
            yield gradient

            gradient[0, :, :] = 0


def generate_coulomb_gradient_basisfunction_matrix(
    mol: gto.Mole, device: str | torch.device = "cpu"
) -> Generator[torch.Tensor, None, None]:
    r"""Generator which computes the 2-center Coulomb gradient integral vector.

    Compute

    .. math::

        \left(\nabla_{R_A}W\right)_{A_{x/y/z}\mu\nu} = \int\int \frac{\nabla_{R_A}\omega_\mu(r_1) \omega_\nu(r_2)}{r_{12}} \mathrm{d} r_1
        \mathrm{d} r_2

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Yields:
        Tensor of integrals (1, nao, nao), which is yielded 3 * natoms times and ordered according to atomid.
    """
    intor_name = mol._add_suffix("int2c2e_ip1")

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    slices = mol.aoslice_by_atom()

    coulomb_gradient = torch.zeros((1, mol.nao, mol.nao), dtype=OFDFT_DTYPE, device=device)

    for atm_id in range(mol.natm):
        shell_start, shell_end, ao_start, ao_end = slices[atm_id]
        tensor = gto.moleintor.getints(
            intor_name,
            mol._atm,
            mol._bas,
            mol._env,
            shls_slice=(shell_start, shell_end, 0, mol.nbas),
        )
        # we calculate the gradient w.r.t to the electron coordinates which
        # is the negative of the gradient w.r.t to the nuclear coordinates
        tensor *= -1
        tensor = torch.as_tensor(tensor, dtype=OFDFT_DTYPE, device=device)

        for k in range(3):

            coulomb_gradient[0, ao_start:ao_end, :] += tensor[k]
            coulomb_gradient[0, :, ao_start:ao_end] += torch.movedim(tensor[k], 0, 1)

            yield coulomb_gradient

            coulomb_gradient[0, :, :] = 0


def get_coulomb_tensor(
    mol_rho: gto.Mole, mol_orbital: gto.Mole, shls_slice: tuple | None = None, aosym: str = "s1"
) -> np.ndarray:
    r"""Compute the 3-center Coulomb integral tensor between different bases.

    Compute the Coulomb integral over one atomic orbital :math:`\omega_\mu` of the density basis
    given by :code:`mol_rho` and the orbitals :math:`\omega_\nu` of the orbital basis given by
    :code:`mol_orbital`,

    .. math::

        \tilde L_{\mu,\alpha\beta} = (\omega_\mu | \eta_\alpha\eta_\beta)
        = \int\int \frac{\omega_\mu(r_1) \eta_\alpha(r_2) \eta_\beta(r_2)}{r_{12}}
        \mathrm{d} r_1 \mathrm{d} r_2.

    Args:
        mol_rho: The molecule containing information about the geometry and the density basis set.
        mol_orbital: The molecule containing information about the geometry and the orbital
            basis set.

    Returns:
        Tensor of integrals with shape (n_ao_rho, n_ao_orbital, n_ao_orbital).
    """
    intor = mol_rho._add_suffix("int3c2e")
    # concatenate the environment arrays of both molecules (pyscf wants double indices first)
    atm, bas, env = gto.mole.conc_env(
        mol_orbital._atm,
        mol_orbital._bas,
        mol_orbital._env,
        mol_rho._atm,
        mol_rho._bas,
        mol_rho._env,
    )

    # first and second index run over orbital basis, third over density basis
    shls_slice = (
        (
            0,
            mol_orbital.nbas,
            0,
            mol_orbital.nbas,
            mol_orbital.nbas,
            mol_orbital.nbas + mol_rho.nbas,
        )
        if shls_slice is None
        else shls_slice
    )

    # integral = df.r_incore.aux_e2(mol_orbital, mol_rho, intor="int3c2e")
    integral = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice, aosym=aosym)
    return np.moveaxis(integral, -1, 0)


@lru_cache(maxsize=1)  # Use lru_cache to avoid recomputation during label generation
def get_coulomb_tensor_cached(
    mol_rho: gto.Mole, mol_orbital: gto.Mole, shls_slice: tuple | None = None
) -> np.ndarray:
    return get_coulomb_tensor(mol_rho, mol_orbital, shls_slice)
