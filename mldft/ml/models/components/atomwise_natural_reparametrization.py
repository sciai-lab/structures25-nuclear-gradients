"""Atomwise Natural reparametrization.

The implementation is inspired by the global natural reparametrization which is described in
[M-OFDFT]_, in section
`4.3 paragraph 2 <https://arxiv.org/html/2309.16578v2#S4.SS3.SSS0.Px2>`_
and appendix `B.3.2 <https://arxiv.org/html/2309.16578v2#A2.SS3.SSS2>`_. It is also described in
[STRUCTURES25]_, Supplementary Section
`S.3 <https://arxiv.org/html/2503.00443v2#Ax1.SS3>`_.
However, the atomwise natural
reparametrization does not construct the eigendecomposition of the overlap matrix for the whole
sample but does so for overlap matrices of one atom molecules. That is, every atom type is
independently transformed by an atomwise natural reparametrization matrix, hence retaining the
locality of the density coefficients.
"""

from pathlib import Path
from typing import Literal, Optional

import torch
import zarr
from loguru import logger

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices_torch,
)
from mldft.utils.overlap_matrices import get_atomwise_overlap_dict
from mldft.utils.sparse import construct_block_diag_coo_indices_and_shape


class AtomwiseNaturalReparametrization(torch.nn.Module):
    """Class to calculate the atomwise natural reparametrization matrices. The natural reparametrization is described in
    [M-OFDFT]_, section `4.3 paragraph 2 <https://arxiv.org/html/2309.16578v2#S4.SS3.SSS0.Px2>`_ and appendix
    `B.3.2 <https://arxiv.org/html/2309.16578v2#A2.SS3.SSS2>`_ and
    [STRUCTURES25]_, Supplementary Section `S.3 <https://arxiv.org/html/2503.00443v2#Ax1.SS3>`_.

    The transformation matrix and its inverse are constructed by computing overlap matrices for
    every atomic number individually, calculating the transformation by eigenvalue decomposition
    and aligning the transformation matrices to a block diagonal matrix for the whole sample.

    .. note::
        As eigenvalues may not be unique, it is advised to load the transformation dictionaries
        from a file instead of computing them anew from basis_info objects. That is, the class
        should preferably be initialized via the from_file method.
    """

    def __init__(
        self, transformation_dict: dict[int, tuple[torch.Tensor]], sparse: bool = False
    ) -> None:
        """Initializes the AtomwiseNaturalReparametrization from the transformation dictionary.

        Args:
            transformation_dict: The dictionary containing the natural reparametrization
                transformation matrix and its inverse per atomic number.
            sparse: Whether to return the sparse version of the natural reparametrization matrix.
                Defaults to False.
        """

        super().__init__()
        self.transformation_dict = transformation_dict
        self.sparse = sparse

    @classmethod
    def from_basis_info(
        cls,
        basis_info: BasisInfo,
        orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
        sparse: bool = False,
        spin: Optional[int] = None,
    ) -> "AtomwiseNaturalReparametrization":
        """Alternative constructor that constructs the transformation_dict from a BasisInfo.

        Args:
            basis_info: The basis_info used to construct the transformation matrices.
            orthogonalization: Type of orthogonalization to use for the construction of the natural
                reparametrization matrices. Either "symmetric" (default) or "canonical".
            sparse: Whether to return the sparse version of the natural reparametrization matrix.
                Defaults to False.
            spin: The spin of the sample. Defaults to None.

        Returns:
            cls: The class instance initialized from basis_info.

        .. note::
            As eigenvalues may not be unique, it is advised to load the transformation dictionaries
            from a file instead of computing them from basis_info objects.
        """

        logger.warning(
            """Computing Transformation matrices from basis info objects is not
                       advised as this may lead to different transformation matrices for the same
                       atomic number due to non-unique eigenvalues. If possible load from file."""
        )
        transformation_dict = get_atomwise_transformation_dict_from_basis_dict(
            basis_dict=basis_info.basis_dict, orthogonalization=orthogonalization, spin=spin
        )
        return cls(transformation_dict, sparse)

    @classmethod
    def from_file(cls, dict_path: str, sparse: bool = False) -> "AtomwiseNaturalReparametrization":
        """Alternative constructor that loads the transformation_dict from a zarr file. This is the
        preferred way of initializing the class.

        Args:
            dict_path: The path to the zarr file containing the dictionary mapping atomic numbers
              to precomputed reparametrization matrices.
            sparse: Whether to return the sparse version of the natural reparametrization matrix.

        Returns:
            cls: The class instance.

        Raises:
            FileNotFoundError: If the transformation dictionary file does not exist.
        """

        transformation_dict = get_atomwise_transformation_dict_from_file(dict_path)

        return cls(transformation_dict, sparse)

    def save_dict(self, dict_path: str, exist_ok: bool = False) -> None:
        """Save the transformation_dict to a .zarr file.

        Args:
            dict_path: The path at which the .zarr file shall be saved.
            exist_ok: Whether to overwrite the file if it already exists. Defaults to False.
        Raises:
            FileExistsError: If the file already exists at save_path and exist_ok is False.
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

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_coo_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Composes the atomwise natural reparametrization matrix and its inverse given atomic
        numbers from known atomwise blocks.

        Args:
            atomic_numbers: The atomic numbers of the atoms in the sample.
            atom_coo_indices: The indices of the atoms in the sample needed to construct a sparse
                tensor. Defaults to None.

        Returns:
            m: The natural reparametrization matrix for the given sample.
            m_inv: The inverse of the natural reparametrization matrix for the given sample.
        """

        # create a list of the atomwise natural reparametrization matrices
        ms, ms_inv = zip(
            *[self.transformation_dict[atomic_number.item()] for atomic_number in atomic_numbers]
        )

        if not self.sparse and atom_coo_indices is not None:
            logger.warning(
                """atom_coo_indices are provided but sparse is False. Provided atom_coo_indices
                will be ignored and dense transformation matrices will be returned."""
            )

        if self.sparse:
            if atom_coo_indices is not None:
                n_basis = sum([m.shape[0] for m in ms])  # equals sample.n_basis
                shape = (n_basis, n_basis)

            else:
                atom_coo_indices, shape = construct_block_diag_coo_indices_and_shape(
                    *[m.shape for m in ms],
                )

            sparse_m = torch.sparse_coo_tensor(
                indices=atom_coo_indices,
                values=torch.cat([m.flatten() for m in ms]),
                size=shape,
            )
            sparse_m_inv = torch.sparse_coo_tensor(
                indices=atom_coo_indices,
                values=torch.cat([m_inv.flatten() for m_inv in ms_inv]),
                size=shape,
            )

            return sparse_m, sparse_m_inv

        else:
            # Create dense block diagonal transformation matrices for the whole sample
            m = torch.block_diag(*ms).to(device=atomic_numbers.device)
            m_inv = torch.block_diag(*ms_inv).to(device=atomic_numbers.device)

            return m, m_inv

    def sample_forward(self, sample: OFData) -> tuple[torch.tensor, torch.tensor]:
        """Wrapper for the forward method that takes an OFData object as input."""

        if hasattr(sample, "atom_coo_indices"):
            return self.forward(sample.atomic_numbers, sample.atom_coo_indices)
        else:
            return self.forward(sample.atomic_numbers)


def get_atomwise_transformation_dict_from_file(
    dict_path: str | Path,
) -> dict[int, tuple[torch.tensor, torch.tensor]]:
    """Reads the transformation dictionary from a zarr file.

    Args:
        dict_path: The path to the zarr file containing the dictionary mapping atomic numbers to
            precomputed reparametrization matrices.

    Returns:
        transformation_dict: The dictionary mapping atomic numbers to their atomwise natural
            reparametrization matrices, where each value is tuple consisting of the transformation
            matrix and its inverse.
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

    return transformation_dict


def get_atomwise_transformation_dict_from_basis_dict(
    basis_dict: dict,
    orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
    spin: Optional[int] = None,
) -> dict[int, tuple[torch.tensor, torch.tensor]]:
    """Constructs a dictionary that maps atomic numbers to their atomwise natural reparametrization
    matrices and their inverses.

    Args:
        basis_dict: The basis_infos dictionary mapping atomic numbers to basis sets.
        orthogonalization: Type of orthogonalization to use for the construction of the natural
            reparametrization matrices. Either "symmetric" (default) or "canonical".
        spin: The spin of the molecule. Defaults to None.

    Returns:
        atomwise_transformation_dict: The transformation dictionary with atomic numbers as keys and
            a tuple of their respective atomwise natural reparametrization matrices (M and M^{-1})
            as values.
    """

    atomwise_overlap_dict = get_atomwise_overlap_dict(basis_dict, spin=spin)
    atomwise_transformation_dict = {
        atomic_number: natural_reparametrization_matrices_torch(
            overlap_matrix, orthogonalization=orthogonalization
        )
        for atomic_number, overlap_matrix in atomwise_overlap_dict.items()
    }

    return atomwise_transformation_dict
