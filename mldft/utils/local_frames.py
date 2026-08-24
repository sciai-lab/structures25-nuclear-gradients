from typing import Optional

import e3nn.o3._wigner as o3_wigner
import numpy as np
import torch
from e3nn.o3 import Irreps, _rotation
from torch import Tensor

from mldft.ml.models.components.equiformerv2_components.wigner import wigner_D

# monkey patch the wigner_D function from e3nn to the faster version
o3_wigner.wigner_D = wigner_D


def pyscf_to_e3nn_matrix(irreps: Irreps, dtype: torch.dtype = torch.float64) -> Tensor:
    """Calculates the transformation matrix which can be used to transform a coefficient vector in
    the pyscf convention to a coefficient vector in the e3nn convention.

    For details on the conventions, compare https://docs.e3nn.org/en/stable/guide/change_of_basis.html and
    https://pyscf.org/user/gto.html#ordering-of-basis-functions. The difference lies in the orientation
    of the spherical harmonics for l>1.

    A coefficient vector can be transformed by multiplying it with this matrix **on the left**.

    .. warning:
        As of now, the equivariance of the vector in the e3nn convention under e3nn transformations has not been
        checked explicitly

    Args:
        irreps: The irreps of the coefficient vector in the pyscf convention.
        dtype: The dtype of the matrix.

    Returns:
        torch.Tensor: The transformation matrix. Shape (irreps.dim, irreps.dim).
    """
    # pyscf uses a different convention than e3nn for l>1 (the wikipedia convention).
    # To alleviate the difference, we transform the l>1 vectors into the e3nn convention first.
    # To keep the l=1 vectors fixed, we replace them with scalars for this transformation:
    # Construct an irreps object where l=1 vectors are replaced by three l=0 scalars
    irreps_l_greater_1 = []
    for irrep in irreps:
        if irrep.ir.l == 1:
            irreps_l_greater_1.append(f"{3 * irrep.mul}x0e")
        else:
            irreps_l_greater_1.append(str(irrep))
    irreps_l_greater_1 = Irreps("+".join(irreps_l_greater_1))

    change_of_coord = torch.tensor(
        [
            # this specifies the change of basis yzx -> xyz
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )

    return irreps_l_greater_1.D_from_matrix(change_of_coord)


def pyscf_to_e3nn_local_frames_matrix(basis: Tensor, irreps: Irreps) -> Tensor:
    """Calculates transformation matrix which can be used to transform a coefficient vector in the
    pyscf convention to a coefficient vector in local frames in the e3nn convention.

    A coefficient vector can be transformed by multiplying it with this matrix **on the left**.

    Args:
        basis: The basis vectors of the local frame. Shape (3, 3).
        irreps: The irreps of the coefficient vector in the pyscf convention.

    Returns:
        torch.Tensor: The transformation matrix. Shape (irreps.dim, irreps.dim).
    """

    # Ignore the four lines below, we do not need to do this here, as we refactored the local basis module:
    # Now, the second axis points towards the closest neighbor, alleviating the need for this permutation.
    # # Permute the basis to the order zyx.
    # # The reason is that e3nn uses y (the second axis) as the distinguished one, and we want this to correspond to the
    # # first axis in the local basis, which points towards the closest neighbor.
    # # basis = basis[[2, 0, 1], :]

    # Calculate the transformation matrix for transforming from the pyscf to the e3nn convention
    pyscf_to_e3nn_mat = pyscf_to_e3nn_matrix(irreps, dtype=basis.dtype)
    pyscf_to_e3nn_mat = pyscf_to_e3nn_mat.to(basis.device)

    # Calculate the rotation matrix from the basis.
    rot_mat = get_rotation_matrix_from_basis(basis)

    # Calculate the transformation matrix for transforming to the local frames inside the e3nn convention.
    wigner_D_mat = irreps.D_from_matrix(rot_mat)

    return wigner_D_mat @ pyscf_to_e3nn_mat


def compute_local_frames_matrix(bases: Tensor, irreps_per_atom: np.ndarray[Irreps]) -> Tensor:
    r"""Compute a transformation matrix that converts a coefficient vector in the global basis to
    one in local frames defined by per–atom rotation matrices, while accounting for the irreps
    structure (each block is a Wigner D matrix of size \(2l+1\)).

    The final output is block–diagonal over atoms, with each block constructed by
    concatenating the Wigner \(D\) matrices for each irreps block (repeated according
    to multiplicity).

    Parameters
    ----------
    bases : Tensor
        The per–atom rotation matrices. Tensor of shape \((N,3,3)\).
    irreps_per_atom : np.ndarray[Irreps]
        An array of irreps for each atom. Each irreps is assumed to be iterable over tuples
        `(mul, ir)` where `ir.l` is the angular momentum for that block.

    Returns
    -------
    M : torch.Tensor
        The block–diagonal transformation matrix of shape \((D, D)\) where
        \(D = \sum_{\mathrm{atom}} \sum_{\mathrm{block \in irreps}} (2l+1)\).
    """

    blocks = []
    for irreps, basis in zip(irreps_per_atom, bases):
        blocks.append(pyscf_to_e3nn_local_frames_matrix(basis, irreps).to(bases.device))

    return torch.block_diag(*blocks)


def batch_compute_local_frames_matrix(
    bases: list[Tensor] | Tensor,
    irreps_per_atom: np.ndarray[Irreps],
) -> Tensor:
    r"""Compute a transformation matrix that converts a coefficient vector in the global basis to
    one in local frames defined by per–atom rotation matrices, while accounting for the irreps
    structure (each block is a Wigner D matrix of size \(2l+1\)).

    The final output is block–diagonal over atoms, with each block constructed by
    concatenating the Wigner \(D\) matrices for each irreps block (repeated according
    to multiplicity).

    This implementation is batched over different rotation matrices (e.g. at different atoms)
    and significantly faster than the naive for loops used in e3nn, as implemented above.

    Parameters
    ----------
    bases : Tensor
        The per–atom rotation matrices. Tensor of shape \((N,3,3)\).
    irreps_per_atom : np.ndarray[Irreps]
        An array of irreps for each atom. Each irreps is assumed to be iterable over tuples
        `(mul, ir)` where `ir.l` is the angular momentum for that block.

    Returns
    -------
    M : torch.Tensor
        The block–diagonal transformation matrix of shape \((D, D)\) where
        \(D = \sum_{\mathrm{atom}} \sum_{\mathrm{block \in irreps}} (2l+1)\).
    """

    device = bases.device

    N = bases.shape[0]
    # --- 1. Ensure bases is a (N,3,3) tensor and get Euler angles in batch.
    # Convert the rotation matrices to Euler angles.
    alpha, beta, gamma = _rotation.matrix_to_angles(bases)  # each of shape (N,)

    # Compute a second set of euler angles derived from the rotation matrix R * C where C is the change of basis from yzx -> xyz
    # This is needed to compute the Wigner D matrices for l>1 in the e3nn convention.
    # Compute the change of basis matrix C
    change_of_basis = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=bases.dtype
    )

    # Compute the new rotation matrices R * C
    bases_e3nn = torch.matmul(bases, change_of_basis[None, :, :])

    # Convert the new rotation matrices to Euler angles.
    alpha_e3nn, beta_e3nn, gamma_e3nn = _rotation.matrix_to_angles(
        bases_e3nn
    )  # each of shape (N,)

    # --- 2. Precompute each block’s size and global offset.
    # For each atom, extract:
    #    - block_sizes: list of (2*l+1) for each block (repeated mul times)
    #    - ls: list of the corresponding l values.
    block_sizes_list = []
    ls_list = []
    for irr in irreps_per_atom:
        sizes = [2 * ir.l + 1 for (mul, ir) in irr for _ in range(mul)]
        ls = [ir.l for (mul, ir) in irr for _ in range(mul)]
        # Convert to tensors (dtype long) on the proper device.
        block_sizes_list.append(torch.tensor(sizes, device=device, dtype=torch.long))
        ls_list.append(torch.tensor(ls, device=device, dtype=torch.long))

    # For each atom, the total number of coefficients is the sum of its block sizes.
    atom_dims = torch.tensor(
        [int(bs.sum().item()) for bs in block_sizes_list], device=device, dtype=torch.long
    )
    # Compute per–atom starting offsets in the global vector.
    # atom_offsets: tensor of shape (N+1,), where atom_offsets[i] is the starting index for atom i.
    atom_offsets = torch.cat(
        (torch.tensor([0], device=device, dtype=torch.long), torch.cumsum(atom_dims, dim=0))
    )
    total_dim = atom_offsets[-1].item()

    # For each atom, compute local offsets for each block (i.e. within that atom).
    local_offsets_list = [
        torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(bs, dim=0)[:-1]))
        for bs in block_sizes_list
    ]
    # Compute global offsets for every block and record the atom index.
    block_atom_list = []  # which atom each block belongs to
    block_offsets_list = []  # starting index (global) for each block
    for i in range(N):
        n_blocks = block_sizes_list[i].shape[0]
        # The global offsets for the blocks of atom i:
        global_offsets = atom_offsets[i] + local_offsets_list[i]
        block_atom_list.append(torch.full((n_blocks,), i, device=device, dtype=torch.long))
        block_offsets_list.append(global_offsets)
    block_atom = torch.cat(block_atom_list)  # shape (n_total_blocks,)
    block_offsets = torch.cat(block_offsets_list)  # shape (n_total_blocks,)
    block_ls = torch.cat(ls_list)  # shape (n_total_blocks,)

    # --- 3. Preallocate the overall transformation matrix.
    M = torch.zeros(total_dim, total_dim, device=device, dtype=bases.dtype)

    # --- 4. For each unique l value, batch-compute the D matrices and scatter them.
    # Group blocks by their l value.
    unique_ls = torch.unique(block_ls, sorted=True, return_inverse=False)
    for l_val in unique_ls:
        l_val_int = int(l_val.item())
        block_size = 2 * l_val_int + 1  # all blocks with this l have the same size

        # Select all blocks corresponding to this l.
        mask = block_ls == l_val
        group_block_offsets = block_offsets[mask]  # shape (n_group,)
        group_block_atoms = block_atom[mask]  # shape (n_group,)

        # Group the blocks by their atom so that we can compute each atom’s D matrix only once.
        unique_atoms, atom_inv = torch.unique(group_block_atoms, sorted=True, return_inverse=True)
        # Batch-compute the Wigner D matrices for these unique atoms.
        if l_val > 1:
            D_unique = wigner_D(
                l_val_int,
                alpha_e3nn[unique_atoms],
                beta_e3nn[unique_atoms],
                gamma_e3nn[unique_atoms],
            )
        else:
            D_unique = wigner_D(
                l_val_int, alpha[unique_atoms], beta[unique_atoms], gamma[unique_atoms]
            )  # shape (n_unique, block_size, block_size)
        # For each block in the group, gather the correct D matrix.
        D_blocks = D_unique[atom_inv]  # shape (n_group, block_size, block_size)

        # Now scatter these blocks into the overall matrix M.
        # For each block, the row (and column) indices are a contiguous range:
        #   rows = block_offset + [0,1,...,block_size-1]
        # We compute these indices in a vectorized fashion.
        # n_group = group_block_offsets.shape[0]
        arange_bs = torch.arange(
            block_size, device=device, dtype=torch.long
        )  # shape (block_size,)
        # Compute an index matrix for the rows and columns of each block.
        row_idx = group_block_offsets.unsqueeze(1) + arange_bs.unsqueeze(
            0
        )  # shape (n_group, block_size)
        col_idx = group_block_offsets.unsqueeze(1) + arange_bs.unsqueeze(
            0
        )  # shape (n_group, block_size)
        # Use advanced indexing to assign all blocks in one go.
        # row_idx.unsqueeze(-1): (n_group, block_size, 1)
        # col_idx.unsqueeze(-2): (n_group, 1, block_size)
        # Their broadcasted sum gives indices of shape (n_group, block_size, block_size)
        M[row_idx.unsqueeze(-1), col_idx.unsqueeze(-2)] = D_blocks

    return M


def transform_coeffs_to_local(coeffs: Tensor, irreps: Irreps, basis: Tensor) -> Tensor:
    """Transforms the coefficients into the local frame.

    Args:
        coeffs (Tensor): Coefficients of the atom.
        irreps (Irreps): Irreps of the atom.
        basis (Tensor): Local basis of the atom.

    Returns:
        Tensor: The transformed coefficients in the local frame.
    """
    # multiply the coefficients with the Wigner matrices to transform them into the local frame
    return (
        pyscf_to_e3nn_local_frames_matrix(basis, irreps).to(coeffs.device, coeffs.dtype) @ coeffs
    )


def transform_local_coeffs_to_global(coeffs: Tensor, irreps: Irreps, basis: Tensor) -> Tensor:
    """Transforms the coefficients (in the e3nn convention) back into the standard basis.

    .. warning:
        The output of this function follows the pyscf convention,
        and is hence not equivariant under the e3nn convention.

    Args:
        coeffs (Tensor): Coefficients of the atom.
        irreps (Irreps): Irreps of the atom.
        basis (Tensor): Local basis of the atom.

    Returns:
        Tensor: The transformed coefficients into the local frame.
    """

    # multiply the coefficients with the Wigner matrices to transform them into the standard basis
    # here we use that the back rotation is the non transposed rotation matrix
    return (
        pyscf_to_e3nn_local_frames_matrix(basis, irreps).T.to(coeffs.device, coeffs.dtype) @ coeffs
    )


def get_rotation_matrix_from_basis(basis: Tensor) -> Tensor:
    """Returns the rotation matrix from the basis.

    Args:
        basis (Tensor): The basis of the local frame.

    Returns:
        Tensor: The rotation matrix to the new basis from the standard basis. Assumed to be multiplied as
        'vector @ rot_mat.T' or 'torch.matmul(rot_mat, vector)'.
    """

    # the rotation matrix is just the basis matrix
    return basis


def get_nonparallel_vector(v: Tensor) -> Tensor:
    """Returns a vector that is not parallel to the input vector.

    For each vector, select a canonical axis (e.g., x=[1,0,0], y=[0,1,0], z=[0,0,1]) that is least
    aligned with v.

    Args:
        v (Tensor): The input vector. Shape (..., 3) where ... can be any number of dimensions.

    Returns:
        Tensor: A vector that is not parallel to the input vector. Shape (..., 3).
    """
    abs_v = torch.abs(v)
    min_idx = torch.argmin(abs_v, dim=-1)  # (...,)

    # Create fallback vectors along that axis
    eye = torch.eye(3, device=v.device)
    fallback = eye[min_idx]  # (..., 3)
    return fallback


def local_frames_from_rel_positions(
    diff_vec_1: Tensor, diff_vec_2: Tensor, diff_vec_3: Optional[Tensor] = None
) -> Tensor:
    """Returns the local frames of the atoms in the molecule.

    In case the vectors are parallel, the second vector is changed to a random vector.

    .. warning:
        Different to how it is described in [M-OFDFT]_, the axis pointing towards the
        closest neighbor is the second axis. This is because the second axis is the
        distinguished axis in e3nn.
        In case of only wanting to put 2 pos in, make the third vector a 0 vector. Then the third axis is just chosen
        s.t. the local frame is right-handed.

    Args:
        diff_vec_1 (Tensor): The relative position of the first neighbor atom. Shape (..., 3)
        diff_vec_2 (Tensor): The relative position of the second neighbor atom. Shape (..., 3)
        diff_vec_3 (Optional[Tensor]): The relative position of the third neighbor atom. Shape (..., 3). If given, the basis system will be chosen such that the third axis is oriented towards the third neighbor.


    Returns:
        torch.Tensor: The local frames of the atoms in the molecule.

    Notes:
        In case the vectors are parallel, the second vector is changed to a random vector.
    """

    # calculate the basis vectors of the local frame with the cross products

    lengths = torch.norm(diff_vec_1, dim=-1)
    assert not torch.any(lengths == 0), f"zero distance, {lengths=}"

    out_shape = list(diff_vec_1.shape) + [3]

    # unsqueeze first two vectors to make batched operations possible
    basis_x = torch.div(diff_vec_1, torch.norm(diff_vec_1, dim=-1).unsqueeze(-1)).view(-1, 3)
    diff_vec_2 = diff_vec_2.view(-1, 3)

    assert not torch.any(torch.isnan(basis_x)), f"NaN in basis_x: {basis_x=}"

    # calculate the z basis vector with the cross product between x and the second direction
    # and normalize it
    cross_z = torch.cross(basis_x, diff_vec_2, dim=-1)
    # NOTE: if the vectors are parallel the norm of the cross product will be zero
    # thus the second vector then needs to be changed.
    cross_z_norm = torch.norm(cross_z, dim=-1)
    if torch.any(cross_z_norm == 0):
        nonparallel_vector = get_nonparallel_vector(basis_x)
        diff_vec_2 = torch.where(cross_z_norm[:, None] == 0, nonparallel_vector, diff_vec_2)
        cross_z = torch.cross(basis_x, diff_vec_2, dim=-1)

    basis_z = cross_z / torch.norm(cross_z, dim=-1).unsqueeze(-1)

    # calculate the y basis vector with the cross product between z and x
    # no need to normalize because x and z are normalized
    basis_y = torch.cross(basis_z, basis_x, dim=-1)

    # if the third vector is given, we can use it to orient the z axis in its direction
    if diff_vec_3 is not None:
        diff_vec_3 = diff_vec_3.view(-1, 3)
        basis_z = torch.where(
            torch.einsum("ij,ij->i", basis_z, diff_vec_3)[:, None] < 0, -basis_z, basis_z
        )

    # stack the vectors into one tensor and return it
    # return torch.stack((basis_x, basis_y, basis_z), dim=-2)
    return torch.stack((basis_z, basis_x, basis_y), dim=-2).view(out_shape)


def local_frames_from_positions(
    pos: Tensor,
    neighbor1_pos: Tensor,
    neighbor2_pos: Tensor,
    neighbor3_pos: Optional[Tensor] = None,
) -> Tensor:
    """Returns the local frames of the atoms in the molecule.
    Warning: This function does not use the third neighbor position.

    Args:
        pos (Tensor): The position of the atom. Shape (..., 3)
        neighbor1_pos (Tensor): The position of the first neighbor atom. Shape (..., 3)
        neighbor2_pos (Tensor): The position of the second neighbor atom. Shape (..., 3)
        neighbor3_pos (Optional[Tensor]): The position of the third neighbor atom. Shape (..., 3). If given, the basis system will be chosen such that the third axis is oriented towards the third neighbor.

    Returns:
        torch.Tensor: The local frames of the atoms in the molecule.
    """

    # calculate vectors which point from the start to the neighbors
    diff_vec_1 = neighbor1_pos - pos
    diff_vec_2 = neighbor2_pos - pos
    diff_vec_3 = neighbor3_pos - pos if neighbor3_pos is not None else None

    # calculate basis and return it
    return local_frames_from_rel_positions(diff_vec_1, diff_vec_2, diff_vec_3)
