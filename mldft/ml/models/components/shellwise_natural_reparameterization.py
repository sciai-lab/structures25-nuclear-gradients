r"""This submodule implements the shellwise natural reparameterization. It is a wrapper around
:code:`compute_single_atom_shellwise_reparameterization_matrices`. Please note that the intended
use is to compute the reparameterization matrices for *all* atoms of a given basis (by initializing
a :code:`ShellwiseNaturalReparameterization` instance) and then assemble the matrices for any given
molecule.

.. note::
    This method presumes that the radial
    part of the basis function may depend on the quantum number :math:`l` but *not* :math:`m`.

.. note::
    The default constructor is not intended to be used. Use the class methods :code:`from_basis_info`
    (before training) and :code:`from_file` (for training and inference).

Method
======
for a given overlap matrix :math:`\mathbf{W}` of a molecule

1. split up
:math:`\mathbf{W} = \bigoplus_{Z \in \mathscr{\mathscr{M}}} \mathbf{W}_Z`

2. for every
:math:`Z \in \mathscr{M}`, split up :math:`\mathbf{W}_Z = \bigoplus_{l} \mathbf{W}_l`

3. for every :math:`l`, permute
:math:`\tilde{\mathbf{W}}_l = \mathbf{P} \mathbf{W}_l \mathbf{P}^\mathsf{T}
= \bigoplus_{m=-l}^l \mathbf{W}_{l,m}` s.t. it has a block structure

4. for every :math:`l` and :math:`m`, compute the reparameterization matrices,
:math:`\tilde{\mathbf{W}}_{l,m} = \mathbf{M}_{l,m} \mathbf{M}_{l,m}^\mathsf{T}`.

5. for every :math:`l`, assemble the reparameterization matrices and permute them back,
:math:`\mathbf{M}_l = \mathbf{P}^\mathsf{T} \bigoplus_m \mathbf{M}_{l,m}`

6. assemble :math:`\mathbf{M}_Z = \bigoplus_l \mathbf{M}_l`

The steps 3 through 5 are done in :code:`compute_single_atom_shellwise_reparameterization_matrices`,
the rest is provided by the class :code:`ShellwiseNaturalReparameterization`.

Note that in step 4, the matrices are actually computed only once for :math:`m=-l`, the others are the
same if the condition on the basis is met.
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import torch
import zarr

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices_torch,
)
from mldft.utils.overlap_matrices import get_atomwise_overlap_matrix
from mldft.utils.sparse import construct_block_diag_coo_tensor


class ShellwiseNaturalReparameterization(torch.nn.Module):
    # default constructor, not intended to be used
    def __init__(self, transformation_dict: Dict[int, torch.Tensor], sparse: bool = False):
        """Shellwise natural reparametrization class.

        Args:
            transformation_dict (Dict[int, torch.Tensor]): A dictionary mapping atomic numbers to their matrix.
            sparse (bool): Whether to return the sparse version of the natural reparametrization matrix.
        """
        super().__init__()
        self.transformation_dict = transformation_dict
        self.sparse = sparse

    @classmethod
    def from_basis_info(
        cls,
        basis_info: BasisInfo,
        orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
        sparse: Optional[bool] = False,
    ):
        r"""Constructor that computes the transformation_dict for a given basis from scratch.

        Args:
            basis_info (BasisInfo): The basis info of the chosen basis set.
            orthogonalization (str): The type of orthogonalization to use which determines how
                the reparametrization matrix is formed from the overlap matrix' eigendecomposition
                Either "symmetric" (default) or "canonical".
            sparse (Bool):          Whether to save the reparam matrices as sparse.

        Returns:
            ShellwiseNaturalReparameterization: The class instance.
        """
        transformation_dict = {
            atomic_number: construct_single_atom_shellwise_reparameterization_matrices(
                atomic_number, basis_info, orthogonalization=orthogonalization, sparse=sparse
            )
            for atomic_number in basis_info.atomic_numbers.tolist()
        }

        return cls(transformation_dict, sparse=sparse)

    @classmethod
    def from_file(cls, dict_path: str, sparse: bool = False):
        """Constructor that loads the transformation_dict from a zarr file.

        Args:
            dict_path (str):  The path to the zarr file containing the dictionary mapping atomic numbers
                        to precomputed reparametrization matrices.
            sparse: Whether to return the sparse version of the natural reparametrization matrix.

        Returns:
            ShellwiseNaturalReparameterization: The class instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """

        dict_path = Path(dict_path)
        if not dict_path.exists():
            raise FileNotFoundError(f"{dict_path} does not exist.")

        dict_file = zarr.open_group(dict_path, mode="r")
        transformation_dict = {
            int(atomic_number): (
                torch.as_tensor(m, dtype=torch.float64),
                torch.as_tensor(m_inv, dtype=torch.float64),
            )[:]
            for atomic_number, (m, m_inv) in dict_file.items()
        }

        return cls(transformation_dict, sparse=sparse)

    def save_dict(self, dict_path: str, exist_ok: bool = False) -> None:
        """Save the transformation_dict to a .zarr file.

        Args:
            dict_path (str): The path to the .zarr file.
            exist_ok (Bool): Whether to overwrite the file if it already exists. Defaults to False.
        Raises:
            FileExistsError: If the file already exists and exist_ok is False.
        """
        dict_path = Path(dict_path)
        if dict_path.exists() and not exist_ok:
            raise FileExistsError(f"{dict_path} already exists.")

        assert (
            dict_path.suffix == ".zarr" or dict_path.suffix == ".zip"
        ), f"File {dict_path} must be a .zarr or .zip file, but is {dict_path.suffix}."

        with zarr.ZipStore(dict_path, mode="w") as store:
            root = zarr.open_group(store=store, mode="w")
            for atomic_number, (m, m_inv) in self.transformation_dict.items():
                root[str(atomic_number)] = (m, m_inv)

    def forward(self, atomic_numbers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the natural reparametrization matrix for a given sample.

        Args:
            sample (OFData):   The molecule geometry.

        Returns:
            Tuple[Tensor]: The natural reparametrization matrix and its inverse.
        """

        m_list, m_inv_list = zip(*[self.transformation_dict[Z.item()] for Z in atomic_numbers])

        if self.sparse:
            m = construct_block_diag_coo_tensor(*m_list)
            m_inv = construct_block_diag_coo_tensor(*m_inv_list)
        else:
            m = torch.block_diag(*m_list)
            m_inv = torch.block_diag(*m_inv_list)

        return m, m_inv

    def sample_forward(self, sample: OFData) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for the forward method that takes an OFData object as input."""
        return self.forward(sample.atomic_numbers)


def construct_single_atom_shellwise_reparameterization_matrices(
    Z: int,
    basis_info: BasisInfo,
    orthogonalization: Literal["symmetric", "canonical"],
    spin: Optional[int] = None,
    sparse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a one atom molecule for the given atomic number, basis_info and spin to return the
    shellwise overlap matrix of the one atom molecule.

    Args:
        Z (int):          The atomic number of the atom, e.g. 1 for hydrogen.
        basis_info (BasisInfo): The basis info of the chosen basis set.
        orthogonalization (str): The type of orthogonalization to use. Either "symmetric" (default)
            or "canonical".
        spin (int):       The spin of the atom. Defaults to None.
        sparse (Bool):     Whether to return a sparse matrix.

    Returns:
        Tuple:          The reparametrization matrix and its inverse.
    """
    # whole atom:
    overlap_matrix = get_atomwise_overlap_matrix(Z, basis_info.basis_dict, spin=spin)

    # get the quantum numbers for the basis functions of the atom
    atom_idx = basis_info.atom_ind_to_basis_function_ind[basis_info.atomic_number_to_atom_index[Z]]
    l_atom = torch.from_numpy(basis_info.l_per_basis_func[atom_idx])  # the l's of the atom
    m_atom = torch.from_numpy(basis_info.m_per_basis_func[atom_idx])  # the m's of the atom
    unique_ls, num_ls = torch.unique(l_atom, return_counts=True)
    num_funcs_per_m = (num_ls / (2 * unique_ls + 1)).type(torch.int64)
    assert torch.all(unique_ls.eq(torch.sort(unique_ls)[0])), "l's are not sorted"
    l_idx = torch.cumsum(num_ls, dim=0)  # presumes that the l's are ordered

    # l=0 doesn’t need special treatment
    overlap_matrix_l0 = overlap_matrix[: l_idx[0], : l_idx[0]]
    m_0, m_0_inv = natural_reparametrization_matrices_torch(
        overlap_matrix_l0, orthogonalization=orthogonalization
    )
    m_list = [m_0]
    m_inv_list = [m_0_inv]

    for l in unique_ls[unique_ls != 0]:  # l=0 already done # noqa: E741 (ignore flake8 error)
        overlap_matrix_l = overlap_matrix[
            l_idx[l - 1] : l_idx[l], l_idx[l - 1] : l_idx[l]
        ]  # subblock of W for l
        p = torch.argsort(m_atom[l_atom == l], dim=0, stable=True)  # permutation,
        # stable=True to preserve order of equal vals of m
        p_inv = torch.argsort(p, dim=0)
        overlap_matrix_l_permuted = overlap_matrix_l[p][:, p]  # should have block structure

        # only compute the reparametrization matrices for m=-l, presume the others are the same
        m_lm_permuted, m_lm_inv_permuted = natural_reparametrization_matrices_torch(
            overlap_matrix_l_permuted[: num_funcs_per_m[l], : num_funcs_per_m[l]],
            orthogonalization=orthogonalization,
        )
        m_l_permuted = torch.block_diag(*[m_lm_permuted for _ in range(2 * unique_ls[l] + 1)])
        m_l_inv_permuted = torch.block_diag(
            *[m_lm_inv_permuted for _ in range(2 * unique_ls[l] + 1)]
        )

        # permute back
        m_list.append(m_l_permuted[p_inv][:, p_inv])
        m_inv_list.append(m_l_inv_permuted[p_inv][:, p_inv])

    if sparse:
        m = construct_block_diag_coo_tensor(*m_list)
        m_inv = construct_block_diag_coo_tensor(*m_inv_list)
    else:
        m = torch.block_diag(*m_list).to(dtype=torch.get_default_dtype())
        m_inv = torch.block_diag(*m_inv_list).to(dtype=torch.get_default_dtype())

    return m, m_inv
