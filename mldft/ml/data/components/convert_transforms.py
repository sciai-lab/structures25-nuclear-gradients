import copy
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import torch
from loguru import logger
from pyscf import dft
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import Adj

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.local_frames_module import LocalBasisModule
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ml.models.components.virtual_nodes import (
    find_max_cliques,
    pick_min_overlapping_sets,
)
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
)
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import build_molecule_ofdata
from mldft.utils.sparse import construct_block_diag_coo_indices_and_shape

DEFAULT_FLOAT64_KEYS = ("energy_label",)


def apply_to_attributes(f: Callable, sample: OFData, attributes: tuple) -> OFData:
    """Simplified version of :meth:`Data.apply`, with the crucial difference that errors are not
    quietly ignored.

    Applies the function f to the attributes of the object obj specified in the tuple attributes.

    .. warning::
        This function modifies the object in-place.

    Args:
        f: The function to apply.
        sample: The object to apply the function to.
        attributes: The names of the attributes to apply the function to.

    Returns:
        The object with the attributes modified.
    """
    for attr in attributes:
        try:
            setattr(sample, attr, f(getattr(sample, attr)))
        except Exception:
            raise RuntimeError(f"Error applying {f} to attribute {attr} of {sample}.")
    return sample


def split_by_atom(
    sample: OFData, fields=("coeffs", "ground_state_coeffs", "gradient_label")
) -> OFData:
    """Split all basis-function wise fields by atom. Currently, not planned to be used, but
    demonstrates how fields can be split when needed.

    Args:
        sample: The sample.
        fields: The fields to split.

    Returns:
        The sample with the fields split by atom.
    """
    for key in fields:
        setattr(sample, key, np.split(getattr(sample, key), sample.atom_ptr[:-1]))
    return sample


class SplitByAtom:
    """Split all basis-function wise fields by atom."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`split_by_atom` for details.
        """
        return split_by_atom(sample)


class ProjectGradient(torch.nn.Module):
    """Project gradients stored in a sample onto the tangent space of the manifold of constant
    electron number."""

    def forward(self, sample: OFData) -> OFData:
        """Project the fields of the sample as configured in  :meth:`__init__`.

        Args:
            sample (OFData): The sample.

        Returns:
            OFData: The sample with the projected gradients
        """
        for key, representation in sample.representations.items():
            if representation == Representation.GRADIENT and hasattr(sample, key):
                sample[key] = project_gradient(getattr(sample, key), sample)
        return sample


def dtype_map(dtype: np.dtype, float_dtype: np.dtype | None | str = None) -> None | torch.dtype:
    """Map a numpy dtype to a torch dtype.

    Args:
        dtype: The numpy dtype.
        float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings,
            "torch.float64" and "torch.float32" are supported.

    Returns:
        The torch dtype, or None if no mapping exists.
    """
    # This adds hydra support for torch.float64 and torch.float32
    if isinstance(float_dtype, str):
        float_dtype = str_to_torch_float_dtype(float_dtype)
    if (
        dtype == np.float64
        or dtype == np.float32
        or dtype == np.float16
        or dtype == torch.float64
        or dtype == torch.float32
        or dtype == torch.float16
        or dtype == torch.bfloat16
    ):
        return torch.get_default_dtype() if float_dtype is None else float_dtype
    elif dtype == np.uint8:
        return torch.int
    else:
        return None


def tensor_or_array_to_torch(
    x: torch.Tensor | np.ndarray,
    device=None,
    float_dtype: np.dtype | torch.dtype | str | None = None,
) -> torch.Tensor:
    dtype = dtype_map(x.dtype, float_dtype=float_dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_torch(
    sample: OFData,
    device=None,
    float_dtype: np.dtype | torch.dtype | str | None = None,
    float64_keys: Sequence[str] | None = DEFAULT_FLOAT64_KEYS,
) -> OFData:
    """Convert all numpy arrays in the sample to torch tensors.

    Args:
        sample: The sample.
        device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
        float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
            and "torch.float32" are supported to enable hydra support.
        float64_keys: Keys for which to always use float64 dtype, e.g. for the energy_label.
            If None, no keys are forced to float64.

    Returns:
        The sample with all numpy arrays converted to torch tensors.
    """
    keys = []
    for key in sample.keys():
        if isinstance(getattr(sample, key), np.ndarray):
            if not getattr(sample, key).dtype == np.object_:
                keys.append(key)
        elif isinstance(getattr(sample, key), torch.Tensor):
            keys.append(key)

    float64_keys = () if float64_keys is None else tuple(float64_keys)

    func = partial(tensor_or_array_to_torch, device=device, float_dtype=float_dtype)
    func64 = partial(tensor_or_array_to_torch, device=device, float_dtype=torch.float64)
    for key in keys:
        converter = func64 if key in float64_keys else func
        setattr(sample, key, converter(getattr(sample, key)))
    return sample


class ToTorch:
    """Convert all numpy arrays in the sample to torch tensors."""

    def __init__(
        self,
        device: torch.device = None,
        float_dtype: np.dtype | torch.dtype | None = None,
        float64_keys: Sequence[str] | None = DEFAULT_FLOAT64_KEYS,
    ):
        """Initialize the transform.

        Args:
            device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
            float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
                and "torch.float32" are supported.
            float64_keys: Keys for which to always use float64 dtype, e.g. for the energy_label.
                If None, no keys are forced to float64.
        """
        self.device = device
        self.float_dtype = float_dtype
        self.float64_keys = () if float64_keys is None else tuple(float64_keys)

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`to_torch` for details.
        """
        return to_torch(
            sample,
            device=self.device,
            float_dtype=self.float_dtype,
            float64_keys=self.float64_keys,
        )


def to_numpy(sample: OFData) -> OFData:
    """Convert all torch tensors in the sample to numpy arrays."""
    keys = tuple(key for key in sample.keys() if isinstance(getattr(sample, key), torch.Tensor))
    return apply_to_attributes(lambda x: x.detach().cpu().numpy(), sample, keys)


class ToNumpy:
    """Convert all torch tensors in the sample to numpy arrays."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`to_numpy` for details.
        """
        return to_numpy(sample)


def add_atom_coo_indices(sample: OFData) -> OFData:
    """Add a field "atom_coo_indices" to the sample, containing an index tensor of shape (2,
    n_basis), which can be used to construct a block diagonal sparse matrix with each block
    corresponding to the basis functions of one atom.

    Args:
        sample: The sample.

    Returns:
        The sample with the field "atom_coo_indices" added.
    """
    block_shapes = [(n_basis, n_basis) for n_basis in sample.n_basis_per_atom]
    sample.add_item(
        "atom_coo_indices",
        construct_block_diag_coo_indices_and_shape(*block_shapes)[0],
        representation=Representation.NONE,
    )
    return sample


class AddAtomCooIndices:
    """Add a field "atom_coo_indices" to the sample, containing an index tensor of shape (2,
    n_basis), which can be used to construct a block diagonal sparse matrix with each block
    corresponding to the basis functions of one atom."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`add_atom_coo_indices` for details.
        """
        return add_atom_coo_indices(sample)


class AddOverlapMatrix:
    """Adds the overlap matrix to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        # The overlap matrix is only ever needed to compute the natrep transformation matrix.
        sample.add_item("overlap_matrix", get_overlap_matrix(mol), Representation.BILINEAR_FORM)
        return sample


class AddFullEdgeIndex(torch.nn.Module):
    """Add a full edge index to the sample."""

    def forward(self, sample: OFData) -> OFData:
        """Add a full edge index to the sample and returns the same sample with the additional
        edge_index attribute.

        Args:
            sample (OFData): Sample data object.

        Returns:
            OFData: The same sample data object with the additional edge_index attribute.
        """

        number_nodes = sample.pos.shape[0]
        edge_index = torch.stack(
            torch.meshgrid(torch.arange(number_nodes), torch.arange(number_nodes), indexing="ij")
        ).reshape(2, -1)

        sample.add_item("edge_index", edge_index, Representation.NONE)
        return sample


class AddRadiusEdgeIndex(torch.nn.Module):
    """Add a radius edge index to the sample."""

    def __init__(
        self,
        radius: float,
        loop: bool = True,
        max_num_neighbors: int = 32,
    ):
        """
        Args:
            radius: The radius to use for the edge index.
            loop: Whether to include self-loops in the edge index.
            max_num_neighbors: Maximum number of neighbors returned by
                :func:`torch_geometric.nn.radius_graph` for each node.
        """
        super().__init__()
        self.radius = radius
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors

    def forward(self, sample: OFData) -> OFData:
        """Add a radius edge index to the sample and returns the same sample with the additional
        edge_index attribute.

        Args:
            sample (OFData): Sample data object.

        Returns:
            OFData: The same sample data object with the additional edge_index attribute.
        """

        pos = sample.pos
        edge_index = radius_graph(
            pos,
            r=self.radius,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        sample.add_item("edge_index", edge_index, Representation.NONE)
        return sample


class AddBasisInfo:
    """Adds the nuclear attraction vector to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """

        sample.add_item("basis_info", self.basis_info, Representation.NONE)
        return sample


class AddNuclearAttractionVector:
    """Adds the nuclear attraction vector to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        nuclear_attraction_vector = torch.as_tensor(
            get_nuclear_attraction_vector(mol), dtype=torch.float64
        )

        sample.add_item(
            "nuclear_attraction_vector", nuclear_attraction_vector, Representation.DUAL_VECTOR
        )
        return sample


class AddCoulombMatrix:
    """Adds the coulomb matrix to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        coulomb_matrix = torch.as_tensor(get_coulomb_matrix(mol), dtype=torch.float64)

        sample.add_item("coulomb_matrix", coulomb_matrix, Representation.BILINEAR_FORM)
        return sample


class AddMol:
    """Adds the mol to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)

        sample.add_item("mol", mol, Representation.NONE)
        return sample


class AddLocalFrames:
    """Add the local frames to the sample."""

    def __init__(self):
        super().__init__()
        self.local_basis_module = LocalBasisModule()

    def __call__(self, sample: OFData, invert: bool = False) -> OFData:
        """Add the local frames to the sample.

        Args:
            sample: The sample.
        """

        lframes = self.local_basis_module(pos=sample.pos, atomic_numbers=sample.atomic_numbers)
        sample.add_item("lframes", lframes, Representation.NONE)
        return sample


class PrepareForDensityOptimization:
    def __init__(
        self,
        basis_info: BasisInfo,
        add_grid: bool = False,
        grid_level: int = 2,
        grid_prune: str = "nwchem_prune",
    ):
        """Transform adding the necessary information for density optimization."""
        self.basis_info = basis_info
        self.add_grid = add_grid
        self.grid_level = grid_level
        self.grid_prune = grid_prune

    def __call__(self, sample: OFData) -> OFData:
        """Prepare the sample for density optimization.

        Args:
            sample: The sample.

        Returns:
            OFData: The prepared sample.
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        sample.add_item("mol", mol, representation=Representation.NONE)
        if not hasattr(sample, "basis_info"):
            sample.add_item("basis_info", self.basis_info, representation=Representation.NONE)
        if not hasattr(sample, "overlap_matrix"):
            sample = AddOverlapMatrix(basis_info=self.basis_info)(sample)
        if not hasattr(sample, "coulomb_matrix"):
            coulomb_matrix = get_coulomb_matrix(mol)
            sample.add_item(
                "coulomb_matrix", coulomb_matrix, representation=Representation.BILINEAR_FORM
            )
        if not hasattr(sample, "nuclear_attraction_vector"):
            nuclear_attraction_vector = get_nuclear_attraction_vector(mol)
            sample.add_item(
                "nuclear_attraction_vector",
                nuclear_attraction_vector,
                representation=Representation.DUAL_VECTOR,
            )
        if self.add_grid:
            logger.info("Adding grid to sample.")
            grid = grid_setup(mol, self.grid_level, self.grid_prune)
            sample.add_item("grid_level", self.grid_level, representation=Representation.NONE)
            sample.add_item("grid_prune", self.grid_prune, representation=Representation.NONE)
            ao = np.asarray(dft.numint.eval_ao(mol, grid.coords, deriv=1))
            sample.add_item("ao", ao, representation=Representation.AO)
        return sample


class AddCliqueNodes(BaseTransform):
    def __init__(self, n_min_cliques: int, n_max_cliques: int = None):
        """Adaptation of pytorch geometric's VirtualNode class to create clique nodes.

        Args:
            n_min_cliques: The number of clique virtual nodes added to the graph regardless of
                overlap between cliques. In effect, this is silently set to
                min(<n_cliques_found_in_the_graph>, n_min_cliques).
            n_max_cliques: The possible number of clique virtual nodes added in total.
                Cliques nodes beyond `n_min_cliques` are only added if their cliques do not overlap
                with cliques of already added virtual nodes. Defaults to `None` in which case
                `n_min_cliques` alone sets the number of cliques added.
        """
        super().__init__()
        if n_min_cliques <= 0:
            raise ValueError(
                f"Minimal number cliques needs to be positive, but is {n_min_cliques}"
            )
        if n_max_cliques is None:
            n_max_cliques = n_min_cliques
        if n_max_cliques < n_min_cliques:
            raise ValueError(
                f"Maximal number cliques cannot be smaller than minimal number of cliques, but is {n_max_cliques}< {n_min_cliques}"
            )
        self.n_min_cliques = n_min_cliques
        self.n_max_cliques = n_max_cliques

    def _get_virtual_graph(self, edge_index: Adj, num_real_nodes: int) -> tuple[torch.Tensor, Adj]:
        """Creates the virtual graph (V, E) of virtual node indices V and edge index E, which
        connects clique members with their clique node and clique nodes amongst each other.

        Args:
            edge_index: The real edge_index of the to be transformed sample.
            num_real_nodes: Number of real nodes needed to compute virtual node indices.
                Defaults to `None` in which case the maximal index of the `edge_index`
                will be used.

        Returns:
            cl_node_ind: Tensor indexing clique nodes, taking into account the offset due to the real node index.
            cl_edge_index: Edge index of the clique node graph which has to be concatenated to the input sample edge_index.
        """

        if num_real_nodes is not None:
            cl_node_offset = num_real_nodes
        else:
            cl_node_offset = edge_index.max().item() + 1

        cliques = find_max_cliques(edge_index)

        cliques = pick_min_overlapping_sets(
            cliques, k_min=self.n_min_cliques, k_max=self.n_max_cliques
        )

        n_cliques = len(cliques)

        cl_node_ind = torch.arange(
            cl_node_offset, cl_node_offset + n_cliques
        )  # clique node indices

        # Every node in every clique gets two edges: one as src and one as dst.
        # We add a self loop for the virtual node
        cl_edge_index_size = sum([(2 * len(cl) + 1) for cl in cliques])
        cl_edge_index = torch.zeros(
            (2, cl_edge_index_size), dtype=edge_index.dtype, device=edge_index.device
        )

        cl_ptr = 0  # pointing to start of to be filled virtual edges
        for clique, cl_node in zip(cliques, cl_node_ind):
            cl_len = len(clique)
            cl_edge_index[0, cl_ptr : cl_ptr + cl_len] = cl_node
            cl_edge_index[1, cl_ptr : cl_ptr + cl_len] = torch.tensor(
                clique, dtype=edge_index.dtype, device=edge_index.device
            )
            cl_ptr += cl_len

        # Add reverse edges for undirected graph
        cl_edge_index[0, cl_ptr : 2 * cl_ptr] = cl_edge_index[1, :cl_ptr]
        cl_edge_index[1, cl_ptr : 2 * cl_ptr] = cl_edge_index[0, :cl_ptr]

        # Add self-loops (No reverse edges here)
        cl_edge_index[0, -n_cliques:] = cl_node_ind
        cl_edge_index[1, -n_cliques:] = cl_node_ind

        # Add inter-clique edges
        if n_cliques > 1:
            inter_cl_edges = torch.combinations(cl_node_ind).T
            inter_cl_edges = inter_cl_edges.to(
                dtype=cl_edge_index.dtype, device=cl_edge_index.device
            )
            inter_cl_edges = torch.cat([inter_cl_edges, inter_cl_edges[[1, 0]]], dim=1)
            cl_edge_index = torch.cat([cl_edge_index, inter_cl_edges], dim=1)

        return cl_node_ind, cl_edge_index

    def forward(self, data: OFData):
        """Adds clique node data to `OFData`.

        For ease of computations in the model, we let old fields like edge_index, atom_ind, atomic
        numbers, etc. remain. We simply add additional virtual and `mixed` fields such that virtual
        nodes can be incorporated in calculations when needed but not by default.
        """
        assert data.edge_index is not None
        edge_index = data.edge_index.clone()
        if hasattr(data, "edge_type"):
            edge_type = data.edge_type
        else:  # keep 'real' edge_type that might differentiate different real edges
            edge_type = torch.zeros_like(edge_index[0])
            data.add_item("edge_type", edge_type, representation=Representation.NONE)
        num_real_nodes = copy.deepcopy(data.num_nodes)
        assert num_real_nodes is not None

        v_node_ind, v_edge_index = self._get_virtual_graph(edge_index, num_real_nodes)
        num_v_nodes = v_node_ind.size(0)
        num_v_edges = v_edge_index.size(1)

        last_real_edge_type = edge_type.max() if edge_type.numel() > 0 else 0
        # We have RV, VR, VV as edge types additionally to (possibly multiple) real edge types
        num_vv = num_v_nodes**2  # edges to itself and all other virtual nodes
        num_rv = (num_v_edges - num_vv) / 2
        assert float(
            num_rv
        ).is_integer(), (
            f"Number of edges between real and virtual nodes, {num_rv=}, must be an integer"
        )
        num_rv = int(num_rv)
        rv_type = edge_type.new_full((num_rv,), last_real_edge_type + 1)
        vv_type = edge_type.new_full((num_vv,), last_real_edge_type + 3)

        # vr_type is rv_type + 1
        edge_type_mixed = torch.cat([edge_type, rv_type, rv_type + 1, vv_type], dim=0)

        old_data = copy.copy(data)
        for key, value in old_data.items():
            if key in ("edge_index", "edge_type"):
                continue

            if key == "lframes":
                spatial_dim = 3
                identity = torch.eye(spatial_dim, device=value.device)
                identities = identity.unsqueeze(0).expand(num_v_nodes, spatial_dim, spatial_dim)
                data.add_item(
                    "lframes_mixed",
                    torch.cat([value, identities], dim=0),
                    Representation.NONE,
                )
                continue

            if isinstance(value, torch.Tensor):
                dim = old_data.__cat_dim__(key, value)
                size = list(value.size())

                fill_value = None
                if key == "edge_weight":
                    size[dim] = num_v_edges
                    fill_value = 1
                    virtual_edge_weight = value.new_full(
                        size, fill_value, device=edge_index.device
                    )
                    data.add_item(
                        "edge_weight_mixed",
                        torch.cat([value, virtual_edge_weight], dim=dim),
                        Representation.NONE,
                    )
                elif key == "batch":
                    fill_value = int(value[0])
                    virtual_batch = torch.full(
                        (num_v_nodes,),
                        fill_value,
                        device=edge_index.device,
                        dtype=value.dtype,
                    )
                    data.add_item(
                        "batch_mixed",
                        torch.cat([value, virtual_batch]),
                        Representation.NONE,
                    )

        data.add_item("num_nodes_mixed", num_real_nodes + num_v_nodes, Representation.NONE)

        is_virtual_node = torch.cat(
            [
                torch.zeros(num_real_nodes, dtype=torch.bool, device=edge_index.device),
                torch.ones(num_v_nodes, dtype=torch.bool, device=edge_index.device),
            ],
            dim=0,
        )
        data.add_item("is_virtual_node", is_virtual_node, Representation.NONE)

        data.edge_index = edge_index
        data.add_item("edge_type_mixed", edge_type_mixed, Representation.NONE)
        data.add_item(
            "edge_index_mixed", torch.cat([edge_index, v_edge_index], dim=1), Representation.NONE
        )
        num_real_edges = edge_index.size(1)
        is_virtual_edge = torch.cat(
            [
                torch.zeros(num_real_edges, dtype=torch.bool, device=edge_index.device),
                torch.ones(num_v_edges, dtype=torch.bool, device=edge_index.device),
            ],
            dim=0,
        )
        data.add_item("is_virtual_edge", is_virtual_edge, Representation.NONE)

        return data
