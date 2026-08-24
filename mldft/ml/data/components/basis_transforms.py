"""Transforms that act on :class:`OFData` objects.

The idea is to apply a subset of transforms as part of the loader, i.e. execute them per-molecule
on the cpu, before batches of multiple molecules are assembled and moved to the GPU.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import AddLocalFrames  # noqa
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.atomwise_natural_reparametrization import (
    AtomwiseNaturalReparametrization,
)
from mldft.ml.models.components.local_frames_module import (
    LocalFramesTransformMatrixDense,
    LocalFramesTransformMatrixSparse,
)
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices_torch,
)
from mldft.ml.models.components.shellwise_natural_reparameterization import (
    ShellwiseNaturalReparameterization,
)
from mldft.utils import RankedLogger
from mldft.utils.local_frames import pyscf_to_e3nn_matrix

log = RankedLogger(__name__, rank_zero_only=True)


def transform_tensor(
    tensor: torch.Tensor,
    transformation_matrix: torch.Tensor,
    inv_transformation_matrix: torch.Tensor,
    representation: Representation,
):
    r"""Transform tensor according to given representation.

    The behavior of the transformation is determined by its `representation`. The available options
    for :class:`~mldft.ml.data.components.of_data.Representation` are:

    -   :class:`Representation.NONE` or :class:`Representation.SCALAR`: no transformation is applied.
    -   :class:`Representation.VECTOR`: e.g. the density coefficients :math:`p`,

        .. math::

            p' = T p.

        This also determines the convention for the transformation matrix.

    -   :class:`Representation.DUAL_VECTOR` and :class:`Representation.GRADIENT` e.g. gradients, vectors that are
        applied to other vectors like the nuclear attraction vector, the basis_integrals and the
        inverse transformation matrix :math:`M = \tilde T^{-1}` after another transformation :math:`T`,

        .. math::

            M' = M T^{-1}.

        The practical difference between dual vectors and gradients is that only gradients will be
        projected in :py:class:`~mldft.ml.data.components.convert_transforms.ProjectGradient`.

    -   :class:`Representation.ENDOMORPHISM`:

        .. math::
            E' = T E T^{-1}.

    -   :class:`Representation.BILINEAR_FORM`: e.g. the coulomb matrix,

        .. math::

            C' = T^{-T} C T^{-1}.

    -   :class:`Representation.AO`: the atomic orbitals, these transform

        .. math::
            A' = A T^{-1}

        which is the same as the inverse vector representation.

    Args:
        tensor: The tensor to be transformed.
        transformation_matrix: The transformation matrix.
        inv_transformation_matrix: The inverse of the transformation matrix.
        representation: The representation that defines the transformation behavior of the tensor.

    Returns:
        The transformed tensor.

    Raises:
        KeyError: If the representation is not known.
    """
    # TODO: Refactor multiplications with einsum to handle potential batch axes
    if representation == Representation.NONE:
        return tensor
    elif representation == Representation.SCALAR:
        return tensor
    elif representation == Representation.VECTOR:
        return transformation_matrix @ tensor
    elif representation == Representation.DUAL_VECTOR or representation == Representation.GRADIENT:
        return tensor @ inv_transformation_matrix
    elif representation == Representation.ENDOMORPHISM:
        return transformation_matrix @ tensor @ inv_transformation_matrix
    elif representation == Representation.BILINEAR_FORM:
        return inv_transformation_matrix.T @ tensor @ inv_transformation_matrix
    elif representation == Representation.AO:
        return tensor @ inv_transformation_matrix
    else:
        raise KeyError(f"Unknown representation {representation}.")


def transform_tensor_with_sample(
    sample: OFData, tensor: torch.Tensor, representation: Representation, invert: bool = False
) -> torch.Tensor:
    """Transform the given tensor with the transformation that the sample was transformed with.

    The transformation_matrix and inv_transformation_matrix have to be added to the sample before
    it is transformed.

    Args:
        sample: The sample that was already transformed and contains the transformation matrices.
        tensor: The tensor to be transformed.
        representation: The representation of the tensor to transform.
        invert: Whether to invert the transformation.

    Returns:
        The transformed tensor.
    """
    assert hasattr(
        sample, "transformation_matrix"
    ), "The sample does not have a transformation matrix."
    assert hasattr(
        sample, "inv_transformation_matrix"
    ), "The sample does not have an inverse transformation matrix."

    transformation_matrix = sample.transformation_matrix
    inv_transformation_matrix = sample.inv_transformation_matrix

    if invert:
        transformation_matrix, inv_transformation_matrix = (
            inv_transformation_matrix,
            transformation_matrix,
        )
    return transform_tensor(
        tensor, transformation_matrix, inv_transformation_matrix, representation
    )


class ApplyBasisTransformation(torch.nn.Module):
    """Transform all fields in the sample according to their representation.

    The transformation is determined by the transformation matrix and its inverse. The
    representation should be saved in the sample.representations.
    """

    def forward(
        self,
        sample: OFData,
        transformation_matrix: torch.Tensor,
        inv_transformation_matrix: torch.Tensor,
        invert: bool = False,
    ) -> OFData:
        """
        Args:
            sample: The sample to transform.
            transformation_matrix: The transformation matrix.
            inv_transformation_matrix: The inverse of the transformation matrix.
            invert: Whether to invert the transformation.

        Returns:
            The transformed sample.
        """
        t_matrix, inv_t_matrix = transformation_matrix, inv_transformation_matrix  # for brevity
        if invert:
            t_matrix, inv_t_matrix = inv_t_matrix, t_matrix
        for key, rep in sample.representations.items():
            if not hasattr(sample, key):
                continue
            attr_value = getattr(sample, key)
            tensor = transform_tensor(attr_value, t_matrix, inv_t_matrix, rep)
            setattr(sample, key, tensor)
        return sample


class ToLocalFrames(ApplyBasisTransformation):
    """Apply the local frames transformation to the sample
    (see [M-OFDFT]_, `B.2 <https://arxiv.org/html/2309.16578v2#A2.SS2>`_)."""

    def __init__(self, sparse: bool = True, add_lframes: bool = False):
        """
        Args:
            sparse: Whether to use a sparse matrix for the transformation.
            add_lframes: Whether to add the local frames to the sample.
        """
        super().__init__()
        self.sparse = sparse
        if self.sparse:
            self.local_frames = LocalFramesTransformMatrixSparse()
        else:
            self.local_frames = LocalFramesTransformMatrixDense()
        self.add_lframes = add_lframes

    def __call__(self, sample: OFData, invert: bool = False) -> OFData:
        """Transform the sample to local frames in-place.

        Args:
            sample: The sample.
        """
        transformation_matrix, lframes = self.local_frames.sample_forward(
            sample, return_lframes=True
        )
        if self.add_lframes:
            sample.add_item("lframes", lframes, Representation.NONE)

        return super().__call__(
            sample, transformation_matrix, transformation_matrix.T, invert=invert
        )


class ToGlobalNatRep(ApplyBasisTransformation):
    """Apply the global natural reparametrization to the sample."""

    def __init__(self, orthogonalization: Literal["symmetric", "canonical"] = "symmetric"):
        """
        Args:
            orthogonalization: The type of orthogonalization to use. Either "symmetric" (default)
                or "canonical".
        """
        super().__init__()
        self.orthogonalization = orthogonalization

    def __call__(self, sample: OFData, invert: bool = False) -> OFData:
        """Naturally reparametrize the sample in-place.

        Args:
            sample: The sample to be transformed.
        """
        assert hasattr(
            sample, "overlap_matrix"
        ), "OFdata sample does not have an overlap matrix to which natural reparametrization needs to be applied."

        (
            transformation_matrix,
            inv_transformation_matrix,
        ) = natural_reparametrization_matrices_torch(
            sample.overlap_matrix, orthogonalization=self.orthogonalization
        )
        return super().__call__(
            sample, transformation_matrix.T, inv_transformation_matrix.T, invert=invert
        )


class ToAtomwiseNatRep(ApplyBasisTransformation):
    """Apply the atomwise natural reparametrization to the sample."""

    natrep_cls = AtomwiseNaturalReparametrization

    def __init__(
        self,
        dict_path: str = None,
        basis_info: BasisInfo = None,
        orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
        sparse: bool = False,
    ):
        """
        Args:
            dict_path: The path to the dictionary containing the atomwise natural reparametrization
                matrices per atomic number.
            basis_info: The basis information in the BasisInfo format.
            orthogonalization: The type of orthogonalization to use. Either "symmetric" (default)
                or "canonical". Only needed when initialized from basis_info.
            sparse: Whether to use a sparse matrix for the transformation. Defaults to False.
        """
        super().__init__()
        if (
            dict_path is None
            and basis_info is None
            or dict_path is not None
            and basis_info is not None
        ):
            raise ValueError(
                "Either dict_path or basis_info must be given, but not both or neither."
            )

        if dict_path is not None:
            assert (
                orthogonalization in dict_path
            ), f"""{orthogonalization} is not in the transformation dictionary name. Chosen
                reparametrization type might not be applied."""
            self.natural_reparametrization = self.natrep_cls.from_file(
                dict_path=dict_path, sparse=sparse
            )
        elif basis_info is not None:
            self.natural_reparametrization = self.natrep_cls.from_basis_info(
                basis_info, orthogonalization=orthogonalization, sparse=sparse
            )

    def __call__(self, sample: OFData, invert: bool = False) -> OFData:
        """Naturally reparametrize the sample in-place.

        Args:
            sample: The sample to be transformed.
        """
        (
            transformation_matrix,
            inv_transformation_matrix,
        ) = self.natural_reparametrization.sample_forward(sample)
        return super().__call__(
            sample, transformation_matrix.T, inv_transformation_matrix.T, invert=invert
        )


class ToShellwiseNatRep(ToAtomwiseNatRep):
    natrep_cls = ShellwiseNaturalReparameterization


class ToE3NNConvention(ApplyBasisTransformation):
    """Transform the sample to the e3nn convention."""

    def __init__(self, basis_info: BasisInfo):
        super().__init__()
        irreps_per_atom = basis_info.irreps_per_atom
        self.atomic_number_to_atom_index = basis_info.atomic_number_to_atom_index
        self.wigner_matrices = []
        for i, ir in enumerate(irreps_per_atom):
            self.wigner_matrices.append(pyscf_to_e3nn_matrix(ir))

    def __call__(self, sample: OFData, invert: bool = False) -> OFData:
        """Transform the sample to the e3nn convention in-place.

        Args:
            sample: The sample to be transformed.
        """
        wigner_indices = self.atomic_number_to_atom_index[
            sample.atomic_numbers.detach().cpu().numpy()
        ]
        if not isinstance(wigner_indices, Iterable):
            # make it a list if there is only one atom
            wigner_indices = [wigner_indices]
        expanded_wigner = [self.wigner_matrices[i] for i in wigner_indices]
        transformation_matrix = torch.block_diag(*expanded_wigner)
        # move to the same device as the sample
        transformation_matrix = transformation_matrix.to(sample.coeffs.device)

        return super().__call__(
            sample, transformation_matrix, transformation_matrix.T, invert=invert
        )


class MasterTransformation(torch.nn.Module):
    def __init__(
        self,
        name: str,
        use_cached_data: bool = True,
        pre_transforms: list = None,
        cached_transforms: DictConfig = None,
        basis_transforms: list[ApplyBasisTransformation] = None,
        post_transforms: list = None,
        add_transformation_matrix: bool = False,
        **kwargs,
    ) -> None:
        """Class that handles the basis transformations.

        Before using the forward method, the configure_cached_transforms method should be called which
        configures whether there are cached basis transforms that can be used.

        Args:
            target: The target transformation.
            pre_transforms: List of transforms to apply before the basis transforms.
            cached_transforms: List of basis transforms that are cached and can be used if the cached
                option is set.
            basis_transforms: List of basis transforms to apply.
            post_transforms: List of transforms to apply after the basis transforms
            add_transformation_matrix: Whether to add the transformation matrix to the sample.
        """
        super().__init__()
        if len(kwargs) > 0:
            logger.warning(
                f"Keyword arguments {kwargs} passed, but will not be used anymore. "
                f"You're probably using an old checkpoint."
            )
        self.name = name
        self.use_cached_data = use_cached_data
        self.pre_transforms = [] if pre_transforms is None else pre_transforms
        self.post_transforms = [] if post_transforms is None else post_transforms
        self.additional_pre_transforms = (
            cached_transforms.get("additional_pre_transforms", [])
            if cached_transforms is not None
            else []
        )
        self.cached_basis_transforms = (
            cached_transforms.get("transforms", []) if cached_transforms is not None else []
        )
        if not self.use_cached_data:
            self.label_subdir = "labels"
        else:
            # Configure which label subdir to use
            cached_transform_name = (
                cached_transforms.name if cached_transforms is not None else "none"
            )
            if cached_transform_name == "none":
                self.label_subdir = "labels"
            else:
                self.label_subdir = f"labels_{cached_transform_name}"
        self.basis_transforms = [] if basis_transforms is None else basis_transforms
        self.add_transformation_matrix = add_transformation_matrix
        if add_transformation_matrix and use_cached_data:
            raise ValueError(
                "Transformation matrix will be wrong when using cached data. Either don't add the transformation matrix or don't use cached data."
            )

    def forward(self, sample: OFData) -> OFData:
        """Apply the configured transformation to a sample."""
        if self.add_transformation_matrix:
            sample = self.initialize_transformation_matrices(sample)
        if not self.use_cached_data:
            for transform in self.additional_pre_transforms:
                sample = transform(sample)
        for transform in self.pre_transforms:
            sample = transform(sample)
        if not self.use_cached_data:
            for transform in self.cached_basis_transforms:
                sample = transform(sample)
        for transform in self.basis_transforms:
            sample = transform(sample)
        for transform in self.post_transforms:
            sample = transform(sample)
        return sample

    def basis_transform(self, sample: OFData) -> OFData:
        """Apply the forward basis transforms to a sample."""
        for transform in self.basis_transforms:
            sample = transform(sample)
        return sample

    def invert_basis_transform(self, sample: OFData) -> OFData:
        """Invert the basis transforms of a sample.

        Inverse transforms are currently used for visualization during training and for obtaining
        the final density in ofdft. This will be done in the torch default float dtype.
        """
        # If the sample has a transformation matrix, the transforms were saved and can be directly applied using
        # the transformation matrices. Otherwise we recompute the transformation matrices using a dummy sample.
        if not (
            hasattr(sample, "transformation_matrix")
            and hasattr(sample, "inv_transformation_matrix")
        ):
            dummy_sample = sample.clone()
            dummy_sample = self.initialize_transformation_matrices(dummy_sample)
            for transform in self.additional_pre_transforms:
                dummy_sample = transform(dummy_sample)
            for transform in self.pre_transforms:
                dummy_sample = transform(dummy_sample)
            for transform in self.cached_basis_transforms:
                dummy_sample = transform(dummy_sample)
            for transform in self.basis_transforms:
                dummy_sample = transform(dummy_sample)
            for transform in self.post_transforms:
                dummy_sample = transform(dummy_sample)
            transformation_matrix = dummy_sample.transformation_matrix
            inv_transformation_matrix = dummy_sample.inv_transformation_matrix
        else:
            transformation_matrix = sample.transformation_matrix
            inv_transformation_matrix = sample.inv_transformation_matrix
        # The transformation matrix will be in the default dtype, as well as the sample
        ApplyBasisTransformation()(
            sample, transformation_matrix, inv_transformation_matrix, invert=True
        )
        return sample

    @staticmethod
    def initialize_transformation_matrices(sample: OFData):
        """Add the transformation matrices to the sample."""
        sample.add_item(
            "transformation_matrix",
            np.eye(sample.coeffs.shape[0], dtype=np.float64),
            Representation.VECTOR,
        )
        sample.add_item(
            "inv_transformation_matrix",
            np.eye(sample.coeffs.shape[0], dtype=np.float64),
            Representation.DUAL_VECTOR,
        )
        return sample
