r"""Implements the nuclear gradients for OFDFT, required for geometry optimization.

The analytical nuclear gradients (the negative of the forces) are calculated as follows for a functional using a ML :math:`E_{TXC}` model:

.. math::

    \grad_{\mathbf{R}_A} E_\text{tot} = \mathbf{{R}_A} E_{en} + \grad_{\mathbf{R}_A} E_{nn} + \grad_{\mathbf{R}_A} E_{TXC} + \grad_{\mathbf{R}_A} E_{J}

As we are using an atom centered basis set the nuclear gradients also include derivatives of the basis functions w.r.t. the nuclear coordinates.
Furthermore both our basis transformations (global natural reparametrization and local frames) depend on the nuclear coordinates either directly in the case of local frames or indirectly via the basis functions in the case of the global natural reparametrization.
With this the gradient of the ML energy functional w.r.t. the nuclear coordinates is given by:

.. math::
    \pdv{E}{\mathbf{R}_A} = \left.\pdv{E}{\mathbf{R}_A}\right\vert_{\mathbf{p}_0} + \left.\pdv{E}{\tilde{\mathbf{p}}}\right\vert_{\mathbf{R}} \pdv{\mathbf{X}}{\mathbf{R}_A} \mathbf{p}_0 \\

where :math:`\mathbf{p}_0` are the density coefficients in the AO basis and :math:`\tilde{\mathbf{p}}` are the density coefficients in the transformed basis and :math:`\mathbf{X}` is the transformation matrix.

.. math::
    \tilde{\mathbf{p}} &= \mathbf{X} \mathbf{p}_0 \\
    \pdv{\tilde{\mathbf{p}}}{\mathbf{R}_A} &= \pdv{\mathbf{X}}{\mathbf{R}_A} \mathbf{p}_0

The Hellman-Feynman gradient is the part of the derivative that is not dependent on the basis functions meaning the contribution from the nuclear-nuclear repulsion and the gradient of the nuclear-electron potential.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Generator, Sequence

import numpy as np
import torch
from e3nn.o3 import Irreps
from hydra.utils import instantiate
from loguru import logger
from pyscf import gto
from pyscf.grad.rhf import grad_nuc

from mldft.ml.data.components.basis_transforms import (
    ApplyBasisTransformation,
    ToGlobalNatRep,
    ToLocalFrames,
    transform_tensor,
    transform_tensor_with_sample,
)
from mldft.ml.data.components.convert_transforms import add_atom_coo_indices, to_torch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.local_frames_module import (
    LocalBasisModule,
    LocalFramesTransformMatrixSparse,
)
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import (
    generate_overlap_nuclear_gradient_tensor,
    get_coulomb_gradient_basisfunction_contraction,
    get_gradient_nuclear_attraction_and_basisfunction_vector,
    get_nuclear_gradient_vector,
    get_overlap_matrix,
)
from mldft.ofdft.density_optimization import density_optimization
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.utils import local_frames
from mldft.utils.sample_generator import SampleGenerator


def _global_natrep_gradient_sylvester(
    ovlp_nuclear_gradient_tensor: torch.Tensor,
    ovlp_eigenvalues: torch.Tensor,
    ovlp_eigenvectors: torch.Tensor,
    eps_eig: float = 1e-12,
) -> torch.Tensor:
    r"""Calculate the gradient of the global symmetric natrep matrix using the Sylvester approach.

    This method is numerically stable even for degenerate or near-degenerate eigenvalues.
    Instead of using eigenvalue differences :math:`(1/(\lambda_i - \lambda_j))`, it uses eigenvalue sums
    :math:`(1/(\sqrt{\lambda_i} + \sqrt{\lambda_j}))` which are always positive for SPD matrices.

    The derivative of :math:`\mathbf{S}^{1/2}` satisfies the Sylvester equation:

    .. math::
        \mathbf{S}^{1/2} · d\mathbf{S}^{1/2} + d\mathbf{S}^{1/2} · \mathbf{S}^{1/2} = d\mathbf{S}

    In the eigenbasis this becomes:

    .. math::
        d\tilde{\mathbf{S}}^{1/2}_{ij} = d\tilde{\mathbf{S}}_{ij} / (√λ_i + √λ_j)

    And then backtransformed with the eigenvectors :math:`\mathbf{U}` to the original basis.

    .. math::
        d\mathbf{S}^{1/2} = \mathbf{U} d\tilde{\mathbf{S}}_{ij} / (√λ_i + √λ_j) \mathbf{U}^T


    Args:
        ovlp_nuclear_gradient_tensor: Derivatives of overlap matrix w.r.t nuclear coords.
            Shape is (m, n, n) where m = 3 * n_atoms.
        ovlp_eigenvalues: Eigenvalues of the overlap matrix. Shape is (n,).
        ovlp_eigenvectors: Eigenvectors of the overlap matrix. Shape is (n, n).
        eps_eig: Clamp eigenvalues below this to avoid division issues.

    Returns:
        The gradient of :math:`\mathbf{S}^{1/2}` w.r.t. nuclear coordinates. Shape is (m, n, n).
    """
    dS = ovlp_nuclear_gradient_tensor

    # Handle 2D input (single gradient matrix) by adding batch dimension
    squeeze_output = False
    if dS.ndim == 2:
        dS = dS.unsqueeze(0)
        squeeze_output = True

    m = dS.shape[0]
    n = dS.shape[1]
    device = dS.device
    dtype = dS.dtype

    # Compute sqrt of eigenvalues with clamping for numerical stability
    lam_safe = torch.clamp(ovlp_eigenvalues, min=eps_eig)
    sqrtlam = torch.sqrt(lam_safe)

    # Denominator: √λ_i + √λ_j (always positive for SPD matrices)
    denom = sqrtlam.unsqueeze(1) + sqrtlam.unsqueeze(0)  # (n, n)

    V = ovlp_eigenvectors
    Vt = V.T

    # Transform to eigenbasis: G = V^T @ dS @ V
    # Use matmul broadcasting: (m, n, n)
    G = Vt @ dS @ V

    # Solve Sylvester equation in eigenbasis
    X_tilde = G / denom.unsqueeze(0)

    # Transform back: dA = V @ X_tilde @ V^T
    dA = V @ X_tilde @ Vt

    if squeeze_output:
        dA = dA.squeeze(0)

    return dA


def _wigner_from_pos(irreps, pos0, pos1, pos2):
    basis = local_frames.local_frames_from_positions(pos0, pos1, pos2)
    transformation_matrix = local_frames.pyscf_to_e3nn_local_frames_matrix(basis, irreps)
    return transformation_matrix


def no_basis_transforms_gradient_generator(
    sample: OFData, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """Generator which yields a zero tensor of the correct shape.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Yields:
        A zero tensor of the correct shape. Shape is (1, n_basis, n_basis) and is yielded 3 * natoms times.
    """
    for i in range(3 * sample.mol.natm):
        yield torch.zeros((1, sample.mol.nao, sample.mol.nao), dtype=OFDFT_DTYPE, device=device)


def global_symmetric_natrep_gradient_generator(
    sample: OFData, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """Generator to calculate the gradient of the global symmetric natrep matrix w.r.t. the nuclear
    coordinates.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Yields:
        The gradient of the global symmetric natrep matrix w.r.t. the nuclear coordinates. Shape
        is (1, n_basis, n_basis) and is yielded natoms times.
    """
    ovlp_matrix = torch.as_tensor(get_overlap_matrix(sample.mol), dtype=OFDFT_DTYPE, device=device)

    ovlp_eigenvalues, ovlp_eigenvectors = torch.linalg.eigh(ovlp_matrix)

    for ovlp_gradient_tensor in generate_overlap_nuclear_gradient_tensor(
        sample.mol, device=device
    ):
        yield _global_natrep_gradient_sylvester(
            ovlp_gradient_tensor, ovlp_eigenvalues, ovlp_eigenvectors
        )


def local_frames_gradient_sparse(sample: OFData, device: torch.device) -> torch.Tensor:
    """Calculate the gradient of the local frames w.r.t. the nuclear coordinates.

    The derivative matrix is build up by only calculating the non-zero blocks.
    The derivative of the Wigner D matrix w.r.t. the nuclear coordinates is calculated by using the
    :py:func:`torch.autograd.functional.jacobian` function.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The final device for the return

    Returns:
        The gradient of the local frames w.r.t. the nuclear coordinates. Shape is (3 * n_atoms, n_basis, n_basis).

    Note:
        This will only work for local frames that ignore hydrogen atoms as neighbours. In this function the device will only affect the final returned tensor and not temporary tensors.
    """
    pos = sample.pos
    atomic_numbers = sample.atomic_numbers
    irreps_per_atom = sample.irreps_per_atom
    n_basis_per_atom = sample.n_basis_per_atom
    n_basis = n_basis_per_atom.sum()
    n_atm = atomic_numbers.shape[0]

    size = (3 * n_atm, n_basis, n_basis)
    indices = []
    values = []

    # This will make problems in case of not ignoring the hydrogen atoms
    heavy_atom_mask = atomic_numbers > 1
    heavy_atom_ind = torch.argwhere(heavy_atom_mask)

    dist_mat = torch.cdist(pos, pos[heavy_atom_mask])
    dist_order = heavy_atom_ind[torch.argsort(dist_mat, dim=1, descending=False)]
    dist_order = dist_order.squeeze()

    basis_start = 0

    pos.requires_grad = True

    for atom_index, atomic_number in enumerate(atomic_numbers):
        n_basis_atom = n_basis_per_atom[atom_index]
        basis_stop = basis_start + n_basis_atom
        basis_indices = torch.arange(basis_start, basis_stop)

        # filters out the neighbours that we need to account for
        # in case of hydrogen atoms the first two entries in the dist_order are the (heavy atom) neighbours
        if atomic_number == 1:
            neighbour_indices = dist_order[atom_index, :2]
        # in case of a heavy atom the first entry is the atom itself and the second and third are the neighbours
        else:
            neighbour_indices = dist_order[atom_index, 1:3]

        pos0 = pos[atom_index]
        pos1, pos2 = pos[neighbour_indices]

        blocks = []

        for rep in irreps_per_atom[atom_index]:
            # calculate only the block for a single occurrence of each irrep
            irrep = Irreps("1x" + str(rep.ir))

            # at least l = 1 or 3 does not work with `forward-mode`
            if (irrep.lmax % 2) == 1:
                # NOTE: this is the slower strategy, whenever possible the other should be used
                # at the current state though this is  the only one working for l = 1 and l = 3
                # I think the reason lies in e3nn, should larger changes to the code be made
                # one could try to fix this
                strategy = "reverse-mode"
            else:
                strategy = "forward-mode"

            multiplicity = rep.mul
            func = partial(_wigner_from_pos, irrep)

            jac0, jac1, jac2 = torch.autograd.functional.jacobian(
                func, (pos0, pos1, pos2), vectorize=True, strategy=strategy
            )
            jac_pos_combined = torch.cat([jac0, jac1, jac2], dim=2).movedim(2, 0)
            # append the block as often as the irrep occurs in the basis
            blocks += [jac_pos_combined for i in range(multiplicity)]

        derivative_matrices = []
        # any atom has itself and two neighbours with each 3 spatial dimensions thus 9
        for j in range(9):
            derivative_matrices.append(torch.block_diag(*[block[j] for block in blocks]))

        derivative_matrices = torch.stack(derivative_matrices)

        atom_indices = [atom_index, *neighbour_indices[:2]]
        atom_coords_indices = torch.tensor(
            [x for i in atom_indices for x in range(3 * i, 3 * i + 3)]
        )

        # if we deal with the indices earlier and smarter this might be avoided all together
        derivative_matrices = derivative_matrices.flatten()
        non_zero_mask = derivative_matrices != 0
        local_indices = torch.stack(
            [
                grid.flatten()
                for grid in torch.meshgrid(
                    atom_coords_indices, basis_indices, basis_indices, indexing="ij"
                )
            ]
        )

        local_indices = local_indices.to(non_zero_mask.device)
        indices.append(local_indices[:, non_zero_mask])
        values.append(derivative_matrices[non_zero_mask])

        basis_start += n_basis_atom

    indices = torch.concatenate(indices, axis=1)
    values = torch.concatenate(values)

    return torch.sparse_coo_tensor(
        indices, values, size=size, dtype=OFDFT_DTYPE, device=device
    ).coalesce()


def local_frames_gradient_generator(
    sample: OFData, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """Generator to calculate the gradient of the local frames w.r.t. the nuclear coordinates and
    yields a dense tensor.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Yields:
        A dense tensor containing the gradient of the local frames w.r.t. the nuclear coordinates. Shape is (1, n_basis, n_basis), it is yielded natoms times.
    """
    sparse_gradient = local_frames_gradient_sparse(sample, device)
    for i in range(sample.mol.natm):
        for k in range(3 * i, 3 * (i + 1)):
            tensor = sparse_gradient[k].to_dense().unsqueeze(0)
            yield tensor


def local_frames_gradient_generator_sparse(
    sample: OFData, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    """Generator to calculate the gradient of the local frames w.r.t. the nuclear coordinates and
    yields a sparse matrix.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Yields:
        A sparse matrix containing the gradient of the local frames w.r.t. the nuclear coordinates.
        Shape is (n_basis, n_basis), it is yielded 3 * natoms times.
    """
    sparse_gradient = local_frames_gradient_sparse(sample, device)
    for i in range(3 * sample.mol.natm):
        yield sparse_gradient[i]


def local_frames_global_symmetric_natrep_gradient_lf_sparse_generator(
    sample: OFData, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    r"""Calculate the gradient of the local frames global symmetric natrep matrix w.r.t. the nuclear
    coordinates using sparse matrix algebra for local frames and their derivatives.

    Args:
        sample: The sample for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Yields:
        The gradient of the local frames global symmetric natrep matrix w.r.t. the nuclear
        coordinates. Shape is (n_basis, n_basis) and is yielded 3 * natoms times.
        At the moment it can not take any further configuration of the basis transformations into account.
    """
    # This might also not work depending upon the exact configuration of the basis transforms
    logger.trace("Calculating local frames matrices")

    with torch.no_grad():
        sample = add_atom_coo_indices(sample)
        local_frames_transformation_matrix = (
            LocalFramesTransformMatrixSparse().sample_forward(sample).to(device)
        )

        local_frames_transformation_matrix_inv = local_frames_transformation_matrix.T

        untransformed_ovlp = torch.as_tensor(
            get_overlap_matrix(sample.mol), dtype=OFDFT_DTYPE, device=device
        )

        logger.trace("Transforming ovlp matrix and gradient tensor")
        # transform the overlap matrix
        local_frame_ovlp = transform_tensor(
            untransformed_ovlp,
            local_frames_transformation_matrix,
            local_frames_transformation_matrix_inv,
            representation=Representation.BILINEAR_FORM,
        )

        # for further speed up this should not be recalculated but passed
        logger.trace("Solving transformed eigenvalue problem")
        # solve the eigenvalue problem for the transformed overlap matrix
        local_frame_ovlp_eigenvalues, local_frame_ovlp_eigenvectors = torch.linalg.eigh(
            local_frame_ovlp
        )

        logger.trace("calculating natrep matrix")
        local_frame_eigval_sqrt = torch.sqrt(local_frame_ovlp_eigenvalues)

        # Global symmetric natrep matrix by hand as we already have the eigenvalues and eigenvectors
        global_natrep_transformation_matrix = torch.einsum(
            "...ij, ...j, ...kj -> ...ik",
            local_frame_ovlp_eigenvectors,
            local_frame_eigval_sqrt,
            local_frame_ovlp_eigenvectors,
        )

        for i, (
            untransformed_ovlp_nuclear_gradient,
            local_frames_gradient_tensor,
        ) in enumerate(
            zip(
                generate_overlap_nuclear_gradient_tensor(sample.mol, device=device),
                local_frames_gradient_generator_sparse(sample, device),
            )
        ):
            tmp = torch.sparse.mm(
                local_frames_gradient_tensor,
                (untransformed_ovlp @ local_frames_transformation_matrix_inv),
            )

            local_frame_ovlp_nuclear_gradient = torch.sparse.mm(
                local_frames_transformation_matrix,
                torch.sparse.mm(
                    untransformed_ovlp_nuclear_gradient.squeeze(0),
                    local_frames_transformation_matrix_inv,
                ),
            )

            torch.add(
                local_frame_ovlp_nuclear_gradient,
                (tmp + tmp.T),
                out=local_frame_ovlp_nuclear_gradient,
            )

            logger.trace("Calculating Natrep gradient")
            # calculate the global symmetric natrep gradient in the local frame
            # Add a batch dimension for the sylvester solver (1, nao, nao)
            global_natrep_gradient = _global_natrep_gradient_sylvester(
                local_frame_ovlp_nuclear_gradient.unsqueeze(0),
                local_frame_ovlp_eigenvalues,
                local_frame_ovlp_eigenvectors,
            )

            # combine the local frame and global symmetric natrep gradients
            logger.trace("calculating combined gradient tensor")
            global_natrep_gradient = global_natrep_gradient.squeeze(0)
            yield (
                global_natrep_gradient @ local_frames_transformation_matrix
                + global_natrep_transformation_matrix @ local_frames_gradient_tensor
            )


def hellmann_gradient(mol: gto.Mole, coeffs: np.ndarray | torch.Tensor) -> np.ndarray:
    r"""Calculate the negative of Hellman-Feynman forces for the given molecule and ofdft
    coefficients (see [M-OFDFT]_, Section `A.5 <https://arxiv.org/html/2309.16578v2#A1.SS5>`_).

    .. math::
        -\mathbf{F}_{HF} = \int \rho(r) \grad_{\mathbf{R}_A} V_{en}(r, \mathbf{R}_A)dr + \grad_{\mathbf{R}_A} E_{nn}

    Args:
        mol: The molecule for which the forces are to be calculated.
        coeffs: The density coefficients for which the forces are to be calculated.

    Returns:
        The Hellman-Feynman forces. The shape of the forces is (3 * n_atoms,) ordered according to
        atom id.
    """
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.detach().cpu().numpy()

    nuclear_gradient_repulsion = grad_nuc(mol).reshape(-1, order="C")
    nuclear_gradient_basis_vector = get_nuclear_gradient_vector(mol)

    return nuclear_gradient_repulsion + nuclear_gradient_basis_vector @ coeffs


def hellmann_gradient_torch(sample, coeffs: torch.Tensor, device: torch.device) -> torch.Tensor:
    r"""Calculate negative of Hellman-Feynman forces for the given molecule and ofdft coefficients.

    .. math::
        -\mathbf{F}_{HF} = \int \rho(r) \grad_{\mathbf{R}_A} V_{en}(r, \mathbf{R}_A)dr + \grad_{\mathbf{R}_A} E_{nn}

    Args:
        mol: The molecule for which the forces are to be calculated.
        coeffs: The density coefficients for which the forces are to be calculated.
        device: The device on which the calculation is to be performed.

    Returns:
        The Hellman-Feynman forces. The shape of the forces is (3 * n_atoms,) ordered according to
        atom id.
    """
    mol = sample.mol
    grad = hellmann_gradient(mol, coeffs)
    return torch.as_tensor(grad, dtype=OFDFT_DTYPE, device=device)


def hartree_basisfunction_gradient(
    sample: OFData, coeffs: torch.Tensor, device: torch.device
) -> torch.Tensor:
    r"""Calculate the gradient of the Hartree energy w.r.t. the nuclear coordinates.

    .. math::
        \pdv{J[\rho]}{\vb{R}_A} &= \frac{1}{2} \sum_{i\in A} \sum_j p_i p_j \int \int - \pdv{\omega_i(r)}{\vb{r}}\frac{\omega_j(r')}{\abs{r - r'}} \dd{r} \dd{r'} \\
        &+ \frac{1}{2} \sum_{i} \sum_{j \in A} p_i p_j  \int \int \frac{\omega_i(r')}{\abs{r - r'}} \left(- \pdv{\omega_j(r)}{\vb{r}} \right)\dd{r} \dd{r'}

    Note that the derivative w.r.t the nuclear coordinates of a basis functions is just the negative of the gradient of the basis function w.r.t. the electron coordinates.

    Args:
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Returns:
        The gradient of the Hartree energy w.r.t. the nuclear coordinates.
    """

    mol = sample.mol
    hartree_gradient = get_coulomb_gradient_basisfunction_contraction(
        mol, coeffs.detach().cpu().numpy()
    )
    return torch.as_tensor(hartree_gradient, dtype=OFDFT_DTYPE, device=device)


def nuclear_attraction_basisfunction_gradient(
    sample: OFData, coeffs: torch.Tensor, device: torch.device
) -> torch.Tensor:
    r"""Calculate the contributions of the basis functions to the analytical nuclear gradient of the
    nuclear attraction energy.

    .. math::
        \pdv{V_{ne}[\rho]}{\vb{R}_A} = \sum_{i\in A} p_i   \int - \pdv{\omega_i(r)}{\vb{r}} v_{ne}(r) \dd{r}


    Args:
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Returns:
        The gradient of the nuclear attraction energy w.r.t. the nuclear coordinates
    """
    mol = sample.mol
    nuclear_attraction_gradient_basisfunction_vector = (
        get_gradient_nuclear_attraction_and_basisfunction_vector(mol)
    )
    nuclear_attraction_gradient_basisfunction_vector = torch.as_tensor(
        nuclear_attraction_gradient_basisfunction_vector,
        dtype=OFDFT_DTYPE,
        device=device,
    )
    return nuclear_attraction_gradient_basisfunction_vector @ coeffs


def fully_numerical_gradient(
    sample_generator: SampleGenerator,
    functional_factory: FunctionalFactory,
    optimizer: Optimizer,
    sample: OFData,
    coeffs: torch.Tensor,
    device: torch.device,
    *,
    epsilon: float = 1e-4,
    max_xc_memory: int = 4000,
    normalize_initial_guess: bool = True,
    ks_basis: str = "6-31G(2df,p)",
    proj_minao_module=None,
    sad_guess_kwargs: dict | None = None,
    strict: bool = True,
) -> torch.Tensor:
    r"""Finite-difference the density-optimized total energy w.r.t. nuclear coordinates.

    Unlike :func:`numerical_model_gradient`, this runs a full density optimization for every
    displaced geometry before evaluating the central finite difference.
    """
    mol = sample.mol
    base_coords = mol.atom_coords(unit="Bohr")
    gradient = torch.zeros(3 * mol.natm, dtype=OFDFT_DTYPE, device=device)
    depsilon = 2 * epsilon

    for flat_coord in range(3 * mol.natm):
        atom_index = flat_coord // 3
        coord_index = flat_coord % 3
        energies = torch.zeros(2, dtype=OFDFT_DTYPE, device=device)

        for energy_index, sign in enumerate([1, -1]):
            perturbed_coords = base_coords.copy()
            perturbed_coords[atom_index, coord_index] += sign * epsilon
            perturbed_mol = mol.set_geom_(perturbed_coords, unit="Bohr", inplace=False)
            perturbed_sample = sample_generator.get_sample_from_mol(perturbed_mol)

            transform_device = getattr(
                getattr(perturbed_sample, "transformation_matrix", None),
                "device",
                device,
            )
            transformed_initial_guess = transform_tensor_with_sample(
                perturbed_sample,
                coeffs.detach().to(transform_device),
                representation=Representation.VECTOR,
            )
            energy, _, converged, _ = density_optimization(
                perturbed_sample,
                perturbed_mol,
                optimizer,
                functional_factory,
                initialization=transformed_initial_guess,
                max_xc_memory=max_xc_memory,
                normalize_initial_guess=normalize_initial_guess,
                ks_basis=ks_basis,
                proj_minao_module=proj_minao_module,
                sad_guess_kwargs=sad_guess_kwargs,
                disable_pbar=True,
                disable_printing=True,
            )
            if not converged:
                message = (
                    "Density optimization did not converge for displacement "
                    f"atom={atom_index} coord={coord_index} sign={sign}"
                )
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
            energies[energy_index] = torch.as_tensor(
                energy.total_energy,
                dtype=OFDFT_DTYPE,
                device=device,
            )

        gradient[flat_coord] = (energies[0] - energies[1]) / depsilon

    return gradient


def numerical_model_gradient(
    sample_generator: SampleGenerator,
    functional_factory: FunctionalFactory,
    sample: OFData,
    coeffs: torch.Tensor,
    *args,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    r"""Calculate the numerical gradient of the energy functional (including the model) w.r.t. the
    nuclear coordinates.

    .. math::
        \pdv{E_{\text{tot}}}{\mathbf{R}_A}

    Args:
        sample_generator: The model configuration.
        functional_factory: The functional factory.
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients in the AO basis for which the gradient is to be calculated.
        epsilon: The step size for the numerical gradient.

    Returns:
        The numerical gradient of the energy functional w.r.t. the nuclear coordinates. Shape is (3 * n_atoms,).
    """
    mol = sample.mol
    gradient_nuclear_coords_ml = torch.zeros((mol.natm, 3))

    base_coords = mol.atom_coords(unit="Bohr")

    depsilon = 2 * epsilon
    for i in range(mol.natm):
        for k in range(3):
            energies = torch.zeros(2)
            for j, sign in enumerate([1, -1]):
                new_coords = base_coords.copy()
                new_coords[i, k] += sign * epsilon

                new_mol = mol.set_geom_(new_coords, unit="Bohr", inplace=False)
                new_sample = sample_generator.get_sample_from_mol(new_mol)
                new_transformed_coeffs = transform_tensor_with_sample(
                    new_sample, coeffs, representation=Representation.VECTOR
                )
                new_sample.coeffs = new_transformed_coeffs
                new_functional = functional_factory.construct(
                    new_sample.mol,
                    coulomb_matrix=new_sample.coulomb_matrix,
                    nuclear_attraction_vector=new_sample.nuclear_attraction_vector,
                )

                new_energy, _ = new_functional(new_sample)
                energies[j] = new_energy.total_energy

            denergies = energies[0] - energies[1]
            gradient_nuclear_coords_ml[i, k] = denergies / depsilon

    gradient_nuclear_coords_ml = gradient_nuclear_coords_ml.reshape(mol.natm * 3)
    return gradient_nuclear_coords_ml


def _convert_sample_and_coeffs_for_model(model, sample, coeffs):
    if model.dtype != sample.coeffs.dtype or model.device != coeffs.device:
        converted_sample = to_torch(
            sample.detach().clone(),
            float_dtype=model.dtype,
            device=model.device,
        )
        converted_coeffs = coeffs.detach().clone().to(model.device, model.dtype)
    else:
        converted_sample = sample
        converted_coeffs = coeffs

    transformed_coeffs = transform_tensor_with_sample(
        converted_sample, converted_coeffs, representation=Representation.VECTOR
    )
    converted_sample.coeffs = transformed_coeffs

    return converted_sample, converted_coeffs


def model_gradient_wrt_nuclear_coordinates(
    model: torch.nn.Module, sample: OFData, coeffs: torch.Tensor
) -> torch.Tensor:
    r"""Calculate the (analytical) gradient of the model w.r.t. the nuclear coordinates.

    .. math::
        \left.\pdv{E}{\mathbf{R}_A}\right\vert_{\mathbf{p}_0}

    Args:
        model: The model for which the gradient is to be calculated.
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients in the AO basis for which the gradient is to be calculated.

    Returns:
        The gradient of the model w.r.t. the nuclear coordinates. Shape is (3 * n_atoms,).
    """

    converted_sample, converted_coeffs = _convert_sample_and_coeffs_for_model(
        model, sample, coeffs
    )

    converted_sample.pos.requires_grad = True

    if hasattr(converted_sample, "lframes"):
        converted_sample.lframes = LocalBasisModule().forward(
            converted_sample.pos, atomic_numbers=sample.atomic_numbers
        )

    model_output = model.net(converted_sample)
    model_nuclear_gradient, *_ = torch.autograd.grad(
        model_output.pred_energy,
        converted_sample.pos,
        # necessary for models that do not use the nuclear positions (as in some of our tests) to
        # then automatically return a zero gradient
        materialize_grads=True,
    )

    return model_nuclear_gradient.detach().to(OFDFT_DTYPE).reshape(-1)


def model_gradient_wrt_coeffs(
    model: torch.nn.Module, sample: OFData, coeffs: torch.Tensor
) -> torch.Tensor:
    r"""Calculate the (analytical) gradient of the model w.r.t. the density coefficients.

    As is required in case of basis transformations that depend on the nuclear coordinates.

    .. math::
        \left.\pdv{E}{\tilde{\mathbf{p}}}\right\vert_{\mathbf{R}}

    Args:
        model: The model for which the gradient is to be calculated.
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients in the AO basis for which the gradient is to be calculated.

    Returns:
        The gradient of the model w.r.t. the density coefficients. Shape is (n_basis,).
    """
    converted_sample, converted_coeffs = _convert_sample_and_coeffs_for_model(
        model, sample, coeffs
    )

    converted_sample.coeffs.requires_grad = True
    # we have usually already predicted the energy so this might not be necessary
    # how to do it better?

    model_output = model.net(converted_sample)
    model_coeffs_gradient, *_ = torch.autograd.grad(
        model_output.pred_energy, converted_sample.coeffs
    )

    return model_coeffs_gradient.detach().to(OFDFT_DTYPE)


def model_gradient(
    model_gradient_wrt_nuclear_coordinates: Callable,
    model_gradient_wrt_coeffs: Callable,
    basis_transform_gradient_wrt_nuclear_coordinates: Callable,
    sample: OFData,
    coeffs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    r"""Calculate the gradient of the model w.r.t. the nuclear coordinates and account for the
    possible dependence of the basis transformations on the nuclear coordinates.

    .. math::
        \pdv{E}{\mathbf{R}_A} = \left.\pdv{E}{\mathbf{R}_A}\right\vert_{\mathbf{p}_0} + \left.\pdv{E}{\tilde{\mathbf{p}}}\right\vert_{\mathbf{R}} \pdv{\mathbf{X}}{\mathbf{R}_A} \mathbf{p}_0

    Args:
        model_gradient_wrt_nuclear_coordinates: Callable that calculates the gradient of the model w.r.t. the nuclear coordinates.
        model_gradient_wrt_coeffs: Callable that calculates the gradient of the model w.r.t. the density coefficients.
        basis_transform_gradient_wrt_nuclear_coordinates: Callable that calculates the gradient of the basis transformation w.r.t. the nuclear coordinates.
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients in the AO basis for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Returns:
        The gradient of the model w.r.t. the nuclear coordinates. Shape is (3 * n_atoms,).
    """
    logger.trace("Calculating model gradient w.r.t nuclear coordinates")
    grad_wrt_coords = model_gradient_wrt_nuclear_coordinates(sample, coeffs).detach()
    logger.trace("Calculating model gradient w.r.t coeffs")
    grad_wrt_coeffs = model_gradient_wrt_coeffs(sample, coeffs).detach()
    logger.trace("Calculating basis transform gradient w.r.t nuclear coordinates")
    grad_basis_transform = basis_transform_gradient_wrt_nuclear_coordinates(
        sample, device
    ).detach()

    logger.trace("Combing model gradient contributions")
    nuclear_gradient = grad_wrt_coords
    nuclear_gradient += torch.einsum(
        "i,xij,j->x", grad_wrt_coeffs, grad_basis_transform, coeffs.detach()
    )

    return nuclear_gradient.detach()


def model_gradient_from_generator(
    model_gradient_wrt_nuclear_coordinates: Callable,
    model_gradient_wrt_coeffs: Callable,
    basis_transform_gradient_wrt_nuclear_coordinates: Callable,
    sample: OFData,
    coeffs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    r"""Calculate the gradient of the model w.r.t. the nuclear coordinates and account for the
    possible dependence of the basis transformations on the nuclear coordinates.

    Is more memory efficient than the non-generator version by never building all the intermediate tensors fully in memory, though slightly slower for small molecules (break even point seems to be around 10 atoms).

    .. math::
        \pdv{E}{\mathbf{R}_A} = \left.\pdv{E}{\mathbf{R}_A}\right\vert_{\mathbf{p}_0} + \left.\pdv{E}{\tilde{\mathbf{p}}}\right\vert_{\mathbf{R}} \pdv{\mathbf{X}}{\mathbf{R}_A} \mathbf{p}_0

    Args:
        model_gradient_wrt_nuclear_coordinates: Callable that calculates the gradient of the model w.r.t. the nuclear coordinates.
        model_gradient_wrt_coeffs: Callable that calculates the gradient of the model w.r.t. the density coefficients.
        basis_transform_gradient_wrt_nuclear_coordinates: Callable that calculates the gradient of the basis transformation w.r.t. the nuclear coordinates.
        sample: The sample for which the gradient is to be calculated.
        coeffs: The density coefficients in the AO basis for which the gradient is to be calculated.
        device: The device on which the calculation is to be performed.

    Returns:
        The gradient of the model w.r.t. the nuclear coordinates. Shape is (3 * n_atoms,).
    """

    logger.trace("Calculating model gradient w.r.t nuclear coordinates")
    grad_wrt_coords = model_gradient_wrt_nuclear_coordinates(sample, coeffs).detach().to(device)
    logger.trace("Calculating model gradient w.r.t coeffs")
    grad_wrt_coeffs = model_gradient_wrt_coeffs(sample, coeffs).detach().to(device)
    logger.trace("Setting up generator for basis transform gradient w.r.t nuclear coordinates")
    grad_basis_transform_generator = basis_transform_gradient_wrt_nuclear_coordinates(
        sample, device
    )
    logger.trace("Combining model gradient contributions")
    nuclear_gradient = grad_wrt_coords

    for i, grad_basis_transform_partial in enumerate(grad_basis_transform_generator):
        logger.trace(f"Basis transform gradient at Atom Id: {i // 3} Coord: {i % 3}")
        if grad_basis_transform_partial.ndim == 2:
            nuclear_gradient_dim = 1
            grad_basis_transform_partial = grad_basis_transform_partial.unsqueeze(0)
        else:
            nuclear_gradient_dim = grad_basis_transform_partial.shape[0]
        nuclear_gradient[
            i * nuclear_gradient_dim : (i + 1) * nuclear_gradient_dim
        ] += torch.einsum(
            "i,xij,j->x",
            grad_wrt_coeffs,
            grad_basis_transform_partial.detach(),
            coeffs.detach(),
        )

    return nuclear_gradient


BASIS_TRANSFORM_GENERATOR_FUNCTION = {
    "no_basis_transforms": no_basis_transforms_gradient_generator,
    "global_symmetric_natrep": global_symmetric_natrep_gradient_generator,
    "local_frames": local_frames_gradient_generator,
    # for most molecular systems the sparse function seems to be faster
    "local_frames_global_symmetric_natrep": local_frames_global_symmetric_natrep_gradient_lf_sparse_generator,
}


def _get_constant_post_basis_transform_matrix(
    sample: OFData, post_basis_transforms: list[ApplyBasisTransformation]
) -> torch.Tensor:
    """Get the combined constant post-basis transformation matrix for a sample.

    Post basis transforms such as ``ToE3NNConvention`` act on the transformed coefficients seen by
    the model, but do not depend on nuclear coordinates themselves. Their constant left-multiplying
    action must still be applied to ``dX / dR`` before contracting with ``dE / d\tilde{p}``.
    """
    sample_post = sample.clone()
    eye = torch.eye(
        sample.coeffs.shape[0],
        dtype=sample.coeffs.dtype,
        device=sample.coeffs.device,
    )
    sample_post.add_item("transformation_matrix", eye, Representation.VECTOR)
    sample_post.add_item("inv_transformation_matrix", eye.clone(), Representation.DUAL_VECTOR)

    for transform in post_basis_transforms:
        sample_post = transform(sample_post)

    return sample_post.transformation_matrix


def _compose_post_basis_transform_gradient(
    basis_transform_gradient: Callable,
    post_basis_transforms: list[ApplyBasisTransformation],
) -> Callable:
    """Compose constant post basis transforms onto a geometry-dependent basis gradient."""
    if len(post_basis_transforms) == 0:
        return basis_transform_gradient

    def wrapped(sample: OFData, device: torch.device):
        post_transform_matrix = _get_constant_post_basis_transform_matrix(
            sample, post_basis_transforms
        )
        post_transform_matrix = post_transform_matrix.to(device)
        for grad_partial in basis_transform_gradient(sample, device):
            if grad_partial.ndim == 2:
                yield post_transform_matrix @ grad_partial
            else:
                yield torch.einsum("ij,xjk->xik", post_transform_matrix, grad_partial)

    return wrapped


def get_basis_transform_gradient_from_config(config) -> Callable:
    """Get the function that calculates the correct basis transform gradient w.r.t. the nuclear
    coordinates from the model configuration.

    Args:
        config: The configuration.

    Returns:
        The function that calculates the correct basis transform gradient w.r.t. the nuclear coordinates
    """
    transforms = instantiate(config.data.transforms)
    basis_transforms = list(transforms.basis_transforms) + list(transforms.cached_basis_transforms)
    post_basis_transforms = [
        transform
        for transform in transforms.post_transforms
        if isinstance(transform, ApplyBasisTransformation)
    ]
    function_dict = BASIS_TRANSFORM_GENERATOR_FUNCTION
    if config.data.transforms.name == "no_basis_transforms":
        assert len(basis_transforms) == 0
        logger.trace("Using no basis transform gradient contribution")
        basis_transform_gradient = function_dict["no_basis_transforms"]

    elif config.data.transforms.name == "global_symmetric_natrep":
        assert len(basis_transforms) == 1
        assert isinstance(basis_transforms[0], ToGlobalNatRep)
        logger.trace("Using Global Symmetric Natrep Gradient")
        basis_transform_gradient = function_dict["global_symmetric_natrep"]

    elif config.data.transforms.name == "local_frames":
        assert len(basis_transforms) == 1
        assert isinstance(basis_transforms[0], ToLocalFrames)
        assert basis_transforms[0].local_frames.local_basis_module.ignore_hydrogen
        logger.trace("Using local frames gradient")
        basis_transform_gradient = function_dict["local_frames"]

    elif config.data.transforms.name == "local_frames_global_symmetric_natrep":
        assert len(basis_transforms) == 2
        assert isinstance(basis_transforms[0], ToLocalFrames)
        assert basis_transforms[0].local_frames.local_basis_module.ignore_hydrogen
        assert isinstance(basis_transforms[1], ToGlobalNatRep)
        logger.trace("Using local frames global (symmetric) natrep gradient")
        basis_transform_gradient = function_dict["local_frames_global_symmetric_natrep"]

    else:
        raise ValueError("Invalid basis transform")

    return _compose_post_basis_transform_gradient(
        basis_transform_gradient,
        post_basis_transforms,
    )


def get_optimal_device(mol: gto.Mole) -> torch.device:
    """Estimate optimal device for nuclear gradient calculations.

    If CUDA is available and has enough free memory to handle the estimated memory requirements of the nuclear gradient calculations for a single coordinate, return CUDA device. Otherwise, return CPU device.

    Args:
        mol: Molecule object

    Returns:
        Device to use
    """

    safety_factor = 0.8  # Use only 80% of available memory to be safe
    nao = mol.nao
    # Estimate memory per consumption (local_frames_global_natrep)
    # others tend to require less memory
    # the most expensive step is likely the calculation of the local frame ovlp nuclear gradient which requires storing
    # two transformation matrices, ovlp matrix, transformed ovlp matrix,
    # eigenvectors, global natrep transformation matrix, ovlp nuclear gradient matrix (and the generator keeps another intermediate tensor, adding another three matrices)
    # local frames gradient (extremely sparse, but fully build as a sparse tensor)
    # three intermediary matrices, and the global natrep gradient matrix itself
    # this makes 14 matrices without the local frames gradient tensor
    # the final yield statement likely generates another two intermediate matrices
    # thus depending upon deallocation and garbage collection we likely have around 16 matrices of size (nao, nao) in memory at the same time for a single coordinate
    # on top of that comes overhead, possible temporary memory requirements for certain operations that I am unaware of and the local frames gradient tensor whose size is not as easy to estiamte as it is sparse. Empirically assuming this additional contributions to be in the order of magitude of 4 further dense matrices seems to work out
    bytes_per_coord = 20 * nao * nao * 8

    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            available_mem = free_mem * safety_factor

            # If a single coordinate doesn't fit, fall back to CPU
            if bytes_per_coord > available_mem:
                logger.warning(
                    f"Not enough GPU memory for single coordinate gradient (req: {bytes_per_coord / 1e6:.2f} MB, avail: {available_mem / 1e6:.2f} MB). Falling back to CPU."
                )
                return torch.device("cpu")
            logger.debug(
                f"Setting device to {device} with estimated memory requirement of {bytes_per_coord / 1e6:.2f} MB and available memory of {available_mem / 1e6:.2f} MB."
            )
            return torch.device("cuda")
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}. Falling back to CPU.")
            return torch.device("cpu")

    logger.warning("CUDA not available or not suitable. Falling back to CPU.")
    return torch.device("cpu")


def _contains_total_energy_model(contributions: Sequence[object]) -> bool:
    """Check whether a list of contributions contains a total-energy ML model.

    Total-energy models are identified by ``target_key == "tot"``.

    Args:
        contributions: Functional contributions to check.

    Returns:
        ``True`` if a total-energy model contribution is present.
    """
    return any(
        isinstance(contribution, torch.nn.Module)
        and getattr(contribution, "target_key", None) == "tot"
        for contribution in contributions
    )


def _should_add_hellmann_contribution(functional_factory: FunctionalFactory) -> bool:
    """Determine whether the analytical Hellmann contribution should be added.

    If a model predicts total energies directly (``target_key == "tot"``), adding the
    analytical Hellmann-Feynman part would double-count it.

    Args:
        functional_factory: Factory defining energy contributions.

    Returns:
        ``True`` when Hellmann-Feynman should be added analytically.
    """
    return not _contains_total_energy_model(functional_factory.contributions)


class NuclearGradients(ABC):
    """Abstract class for calculating the nuclear gradients of the energy functional.

    Attributes:
        func_factory: The functional factory.
        sample_generator: The model configuration.
        basis_transform_gradient_wrt_nuclear_coordinates: The function that calculates the basis transform gradient w.r.t. the nuclear coordinates.
        contributions: The contributions to the nuclear gradients.
        device: The device to run calculations on.
    """

    def __init__(
        self,
        sample_generator: SampleGenerator,
        functional_factory: FunctionalFactory,
        strict: bool = True,
        device: str = "auto",
    ):
        """Initialize the NuclearGradients object.

        Args:
            sample_generator: The model configuration.
            functional_factory: The functional factory.
            strict: Whether to raise an error for unimplemented contributions.
            device: Device to run on ("auto", "cpu", "cuda").
            include_hellmann: Whether to include the analytical Hellmann-Feynman contribution.
        """
        if (
            len(functional_factory.torch_functionals) > 0
            or len(functional_factory.libxc_functionals) > 0
        ):
            raise NotImplementedError("Analytical forces (grid response) are not yet implemented")

        self.func_factory = functional_factory
        self.sample_generator = sample_generator
        self.device_str = device
        self.device = None if device == "auto" else torch.device(device)
        self.contributions = self.get_contributions(strict=strict)

    @abstractmethod
    def get_contributions(self, strict: bool = True) -> list[Callable]:
        raise NotImplementedError

    def __call__(self, sample: OFData, coeffs: np.ndarray | torch.Tensor):
        """Calculate the nuclear gradients of the energy functional.

        Args:
            sample: The sample for which the nuclear gradients are to be calculated.
            coeffs: The density coefficients in the AO basis for which the nuclear gradients are to be calculated.

        Returns:
            The nuclear gradients of the energy functional. The shape of the forces is (3 * n_atoms,) ordered according to
            atom id.
        """
        if isinstance(coeffs, np.ndarray):
            coeffs = torch.as_tensor(coeffs, dtype=OFDFT_DTYPE)

        # Resolve the device once and reuse it for subsequent iterations.
        if self.device is None:
            self.device = get_optimal_device(sample.mol)
        device = self.device

        # Move data to device
        orig_device = coeffs.device
        if device != orig_device:
            coeffs = coeffs.to(device)

        forces = torch.zeros(3 * sample.mol.natm, dtype=OFDFT_DTYPE, device=device)

        for contribution in self.contributions:
            res = contribution(sample, coeffs, device).to(device=device, dtype=OFDFT_DTYPE)
            forces += res

        return forces.to(orig_device)


class NumericalNuclearGradients(NuclearGradients):
    """Class for calculating the numerical nuclear gradients of the energy functional."""

    def __init__(
        self,
        sample_generator,
        functional_factory,
        epsilon=1e-4,
        strict=True,
        device="auto",
    ):
        """Initialize the NumericalNuclearGradients object.

        Args:
            sample_generator: The model configuration.
            functional_factory: The functional factory.
            epsilon: The step size for the numerical gradient.
            device: Device to run on.
        """
        self.epsilon = epsilon
        super().__init__(
            sample_generator,
            functional_factory,
            strict=strict,
            device=device,
        )

    def get_contributions(self, strict=True) -> list[Callable]:
        contributions = [
            partial(
                numerical_model_gradient,
                self.sample_generator,
                self.func_factory,
                epsilon=self.epsilon,
            )
        ]

        return contributions


class FullyNumericalNuclearGradients(NumericalNuclearGradients):
    """Class for finite-difference gradients from density-optimized total energies."""

    def __init__(
        self,
        sample_generator,
        functional_factory,
        optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
        epsilon=1e-4,
        strict=True,
        device="auto",
        max_xc_memory: int = 4000,
        normalize_initial_guess: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        proj_minao_module=None,
        sad_guess_kwargs: dict | None = None,
    ):
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.max_xc_memory = max_xc_memory
        self.normalize_initial_guess = normalize_initial_guess
        self.ks_basis = ks_basis
        self.proj_minao_module = proj_minao_module
        self.sad_guess_kwargs = sad_guess_kwargs
        super().__init__(
            sample_generator,
            functional_factory,
            strict=strict,
            device=device,
        )

    def get_contributions(self, strict=True) -> list[Callable]:
        return [
            partial(
                fully_numerical_gradient,
                self.sample_generator,
                self.func_factory,
                self.optimizer,
                epsilon=self.epsilon,
                max_xc_memory=self.max_xc_memory,
                normalize_initial_guess=self.normalize_initial_guess,
                ks_basis=self.ks_basis,
                proj_minao_module=self.proj_minao_module,
                sad_guess_kwargs=self.sad_guess_kwargs,
                strict=strict,
            )
        ]


class AnalyticalNuclearGradients(NuclearGradients):
    """Class for calculating the analytical nuclear gradients of the energy."""

    def __init__(
        self,
        sample_generator,
        functional_factory,
        strict=True,
        device="auto",
        include_hellmann: bool = True,
    ):
        """Initialize the AnalyticalNuclearGradients object.

        Args:
            sample_generator: The model configuration.
            functional_factory: The functional factory.
            strict: Whether to raise an error for unimplemented contributions.
            device: Device to run on.
            include_hellmann: Whether to include the analytical Hellmann-Feynman contribution.
        """
        self.include_hellmann = include_hellmann
        self.basis_transform_gradient_wrt_nuclear_coordinates = (
            get_basis_transform_gradient_from_config(sample_generator.model_config)
        )
        super().__init__(
            sample_generator,
            functional_factory,
            strict=strict,
            device=device,
        )

    def get_contributions(self, strict=True):
        contributions = self._get_hellmann_initial_contributions()

        for contribution in self.func_factory.contributions:
            if contribution == "hartree":
                contributions.append(hartree_basisfunction_gradient)

            elif contribution == "nuclear_attraction":
                contributions.append(nuclear_attraction_basisfunction_gradient)
            elif isinstance(contribution, torch.nn.Module):
                ml_gradient_wrt_coeffs = partial(model_gradient_wrt_coeffs, contribution)
                ml_gradient_wrt_nuclear_coordinates = partial(
                    model_gradient_wrt_nuclear_coordinates, contribution
                )
                ml_gradient = partial(
                    model_gradient_from_generator,
                    ml_gradient_wrt_nuclear_coordinates,
                    ml_gradient_wrt_coeffs,
                    self.basis_transform_gradient_wrt_nuclear_coordinates,
                )
                contributions.append(ml_gradient)
            else:
                if strict:
                    raise NotImplementedError(f"Contribution {contribution} not implemented")
                else:
                    logger.warning(f"Contribution {contribution} not implemented")

        return contributions

    def _get_hellmann_initial_contributions(self) -> list[Callable]:
        """Build the shared initial contribution list for analytical gradients.

        Returns:
            A list containing the Hellmann-Feynman contribution unless it is skipped for
            total-energy ML models.
        """
        contributions: list[Callable] = []
        if not self.include_hellmann:
            logger.trace("Skipping Hellmann-Feynman contribution by request")
            return contributions
        if _should_add_hellmann_contribution(self.func_factory):
            contributions.append(hellmann_gradient_torch)
        else:
            logger.trace("Skipping Hellmann-Feynman contribution for total-energy model")
        return contributions


DEFAULT_GRADIENTS = "analytic"


def get_gradient_method(
    gradients: str | Callable,
    sample_generator: SampleGenerator,
    func_factory: FunctionalFactory,
    device: str = "auto",
    **kwargs,
):
    """Get the gradient method based on the input string or callable.

    Args:
        gradients: The gradient method as a string or callable.
        sample_generator: The model configuration to use.
        func_factory: The functional factory to use.
        device: Device to run on. If "auto", the device will determined based on the available hardware.

    Returns:
        The gradient method as a callable.
    """
    if callable(gradients):
        return gradients
    elif gradients.lower() == "numerical":
        return NumericalNuclearGradients(sample_generator, func_factory, device=device, **kwargs)
    elif gradients.lower() == "fully_numerical":
        return FullyNumericalNuclearGradients(
            sample_generator,
            func_factory,
            device=device,
            **kwargs,
        )
    elif (
        gradients.lower() == "analytic"
        or gradients.lower() == "analytical"
        or gradients.lower() == "analytical_generator"  # for backward compatibility
    ):
        return AnalyticalNuclearGradients(
            sample_generator,
            func_factory,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown gradient method {gradients}")
