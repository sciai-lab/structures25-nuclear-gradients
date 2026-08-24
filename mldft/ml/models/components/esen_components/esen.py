"""ESEN public API backed by the fast execution layout.

Fresh models constructed through this module use the faster execution path.
The legacy runtime layout is still recognized in ``forward`` so Lightning checkpoints that
pickle instantiated old eSEN modules remain executable after deserialization.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch_geometric

from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.dimension_wise_rescaling import DimensionWiseRescaling
from mldft.ml.models.components.equiformerv2_components.density_embedding import (
    DensityEmbedding,
)
from mldft.ml.models.components.equiformerv2_components.edge_rot_mat import (
    init_edge_rot_mat,
)
from mldft.ml.models.components.equiformerv2_components.initial_guess_delta_module import (
    InitialGuessDelta,
)
from mldft.ml.models.components.esen_components.base import GraphModelMixin
from mldft.ml.models.components.esen_components.embedding import EdgeDegreeEmbedding
from mldft.ml.models.components.esen_components.esen_block import eSEN_Block
from mldft.ml.models.components.esen_components.fast_layers import (
    FastEdgeDegreeEmbedding,
    FastESENBlock,
    FastExecutionBackend,
)
from mldft.ml.models.components.esen_components.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from mldft.ml.models.components.esen_components.radial import (
    EnvelopedBesselBasis,
    GaussianSmearing,
    PolynomialEnvelope,
)
from mldft.ml.models.components.esen_components.rotation import rotation_to_wigner
from mldft.ml.models.components.esen_components.so3 import CoefficientMapping, SO3_Grid
from mldft.ml.models.components.esen_components.so3_layers import SO3_Linear
from mldft.ml.models.components.model_output import ModelOutput


class eSEN_Fast_Backbone(nn.Module, GraphModelMixin):
    """Fast eSEN backbone used for all newly constructed models."""

    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int | list[int] = 2,
        mmax: int | list[int] = 2,
        grid_resolution: int | None = None,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        activation_checkpointing: bool = False,
        deterministic: bool = False,
    ) -> None:
        """Initialize the fast eSEN backbone."""
        super().__init__()
        self.max_num_elements = max_num_elements
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.enforce_max_neighbors_strictly = False
        self.activation_checkpointing = activation_checkpointing
        self.deterministic = deterministic
        self.mlp_type = mlp_type
        self.use_envelope = use_envelope
        self.num_layers = num_layers

        if isinstance(lmax, int):
            self.lmax_list = [lmax] * num_layers
        elif len(lmax) == num_layers:
            self.lmax_list = lmax
        else:
            raise ValueError(f"lmax must be an int or a list of length num_layers ({num_layers})")

        if isinstance(mmax, int):
            self.mmax_list = [mmax] * num_layers
        elif len(mmax) == num_layers:
            self.mmax_list = mmax
        else:
            raise ValueError(f"mmax must be an int or a list of length num_layers ({num_layers})")

        if any(
            lmax_prev < lmax_next
            for lmax_prev, lmax_next in zip(self.lmax_list, self.lmax_list[1:])
        ):
            raise ValueError("eSEN requires non-increasing lmax across layers.")

        Jd_list_pt = torch.load(Path(__file__).parent.parent / "equiformerv2_components" / "Jd.pt")
        self.max_lmax_for_jd = max(self.lmax_list)
        for l_val in range(self.max_lmax_for_jd + 1):
            self.register_buffer(f"Jd_{l_val}", Jd_list_pt[l_val])

        self.sph_feature_size = int((self.lmax_list[0] + 1) ** 2)
        self.mappingReduceds = nn.ModuleList()
        self.SO3_grids = nn.ModuleList()
        for i in range(num_layers):
            self.mappingReduceds.append(CoefficientMapping(self.lmax_list[i], self.mmax_list[i]))
            grid_layer = nn.ModuleDict()
            grid_layer["lmax_lmax"] = SO3_Grid(
                self.lmax_list[i], self.lmax_list[i], resolution=grid_resolution, rescale=True
            )
            grid_layer["lmax_mmax"] = SO3_Grid(
                self.lmax_list[i], self.mmax_list[i], resolution=grid_resolution, rescale=True
            )
            self.SO3_grids.append(grid_layer)

        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels)
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                2.0,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")

        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.backend = FastExecutionBackend()
        self.edge_degree_embedding = FastEdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax_list[0],
            mmax=self.mmax_list[0],
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            mappingReduced=self.mappingReduceds[0],
            backend=self.backend,
        )
        if self.use_envelope:
            self.envelope = PolynomialEnvelope(exponent=5)

        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(
                FastESENBlock(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax_list[i],
                    self.mmax_list[i],
                    self.mappingReduceds[i],
                    self.SO3_grids[i],
                    self.edge_channels_list,
                    self.norm_type,
                    self.act_type,
                    self.mlp_type,
                    self.backend,
                )
            )

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax_list[-1],
            num_channels=self.sphere_channels,
        )

    def get_rotmat_and_wigner(
        self,
        edge_distance_vecs: torch.Tensor,
        lmax_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build edge-frame rotation matrices and Wigner matrices for one block resolution."""
        edge_rot_mat = init_edge_rot_mat(
            edge_distance_vecs,
            rot_clip=(not self.direct_forces),
            deterministic=self.deterministic,
        )
        Jd_buffers = [
            getattr(self, f"Jd_{l_val}").type(edge_rot_mat.dtype)
            for l_val in range(lmax_block + 1)
        ]
        wigner = rotation_to_wigner(
            edge_rot_mat,
            0,
            lmax_block,
            Jd_buffers,
            rot_clip=(not self.direct_forces),
        )
        if wigner.dtype != edge_distance_vecs.dtype:
            wigner = wigner.to(edge_distance_vecs.dtype)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        return edge_rot_mat, wigner, wigner_inv

    def generate_graph(self, *args, **kwargs):
        """Normalize the graph builder output into the dictionary layout expected below."""
        graph = super().generate_graph(*args, **kwargs)
        return {
            "edge_index": graph.edge_index,
            "edge_distance": graph.edge_distance,
            "edge_distance_vec": graph.edge_distance_vec,
            "cell_offsets": graph.cell_offsets,
            "offset_distances": None,
            "neighbors": None,
            "node_offset": 0,
            "batch_full": graph.batch_full,
            "atomic_numbers_full": graph.atomic_numbers_full,
        }

    def _compute_geometry(self, data_dict) -> dict:
        """Build all coefficient-independent tensors: graph, Wigner matrices, edge features."""
        if self.otf_graph:
            graph_dict = self.generate_graph(data_dict)
        else:
            if self.use_pbc:
                raise NotImplementedError(
                    "eSEN does not currently support `otf_graph=False` with `use_pbc=True`."
                )
            edge_source = data_dict["edge_index"][0]
            edge_target = data_dict["edge_index"][1]
            edge_distance_vec = data_dict["pos"].index_select(0, edge_source) - data_dict[
                "pos"
            ].index_select(0, edge_target)
            edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)
            graph_dict = {
                "atomic_numbers_full": data_dict["atomic_numbers_full"],
                "batch_full": data_dict["batch_full"],
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
                "node_offset": 0,
            }

        wigner_cache = {}
        for lmax_val in sorted(set(self.lmax_list)):
            _, wigner, wigner_inv = self.get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"], lmax_val
            )
            wigner_cache[lmax_val] = (wigner, wigner_inv)

        wigners_m, wigner_invs_m = [], []
        for block in self.blocks:
            wigner, wigner_inv = wigner_cache[block.lmax]
            block_out_mask = block.edge_wise.out_mask.to(device=wigner.device)
            wigner_m, wigner_inv_m = self.backend.prepare_wigner(
                wigner,
                wigner_inv,
                block.edge_wise.mappingReduced,
                block_out_mask,
            )
            wigners_m.append(wigner_m)
            wigner_invs_m.append(wigner_inv_m)

        edge_degree_wigner_inv = wigner_invs_m[0]
        if self.use_envelope:
            edge_envelope = self.envelope(graph_dict["edge_distance"] / self.cutoff).view(-1, 1, 1)
            wigner_invs_m = [wigner_inv_m * edge_envelope for wigner_inv_m in wigner_invs_m]
            edge_degree_wigner_inv = edge_degree_wigner_inv * edge_envelope

        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])
        edge_source = graph_dict["edge_index"][0]
        edge_target = graph_dict["edge_index"][1]
        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"].index_select(0, edge_source)
        )
        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"].index_select(0, edge_target)
        )
        x_edge = torch.cat((edge_distance_embedding, source_embedding, target_embedding), dim=1)

        return {
            "graph_dict": graph_dict,
            "wigners_m": wigners_m,
            "wigner_invs_m": wigner_invs_m,
            "edge_degree_wigner_inv": edge_degree_wigner_inv,
            "x_edge": x_edge,
        }

    def forward(self, data_dict) -> dict[str, torch.Tensor]:
        """Run the fast backbone on a prepared graph dictionary and return node embeddings."""
        # The geometry pipeline (graph, Wigner matrices, edge features) is independent of the
        # density coefficients, so during density optimization (fixed positions, many
        # functional evaluations) it can be reused. The cache is only valid while the very
        # same, unmodified ``pos`` tensor is passed in, and must not be used when gradients
        # w.r.t. positions are required (nuclear gradients) or during training.
        pos = data_dict["pos"]
        cache = getattr(self, "_geometry_cache", None)
        pos_version = getattr(pos, "_version", None)
        cacheable = not self.training and not pos.requires_grad and pos_version is not None
        if (
            cacheable
            and cache is not None
            and cache["pos"] is pos
            and cache["pos_version"] == pos_version
        ):
            geometry = cache["geometry"]
        else:
            geometry = self._compute_geometry(data_dict)
            self._geometry_cache = (
                {"pos": pos, "pos_version": pos_version, "geometry": geometry}
                if cacheable
                else None
            )
        graph_dict = geometry["graph_dict"]
        wigners_m = geometry["wigners_m"]
        wigner_invs_m = geometry["wigner_invs_m"]
        edge_degree_wigner_inv = geometry["edge_degree_wigner_inv"]
        x_edge = geometry["x_edge"]

        scalar_embedding = self.sphere_embedding(data_dict["atomic_numbers"]).unsqueeze(1)
        if self.sph_feature_size == 1:
            x_message = scalar_embedding
        else:
            higher_order_zeros = scalar_embedding.new_zeros(
                data_dict["pos"].shape[0],
                self.sph_feature_size - 1,
                self.sphere_channels,
            )
            x_message = torch.cat((scalar_embedding, higher_order_zeros), dim=1)
        x_message = x_message + data_dict["density_embedding"]

        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            graph_dict["edge_index"],
            edge_degree_wigner_inv,
            graph_dict["node_offset"],
        )

        if graph_dict["edge_index"].shape[1] != 0:
            for i in range(self.num_layers):
                current_block_lmax = self.blocks[i].lmax
                num_spherical_harmonics = (current_block_lmax + 1) ** 2
                x_message_input = x_message[:, :num_spherical_harmonics, :]
                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message_input,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigners_m[i],
                        wigner_invs_m[i],
                        graph_dict["node_offset"],
                        use_reentrant=False,
                    )
                else:
                    x_message = self.blocks[i](
                        x_message_input,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigners_m[i],
                        wigner_invs_m[i],
                        node_offset=graph_dict["node_offset"],
                    )

        x_message = self.norm(x_message)
        out = {"node_embedding": x_message}
        out.update(graph_dict)
        return out

    @property
    def num_params(self):
        """Return the total number of parameters in the backbone."""
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        """Return the set of parameter names to exclude from weight decay."""
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


class eSEN_Fast(nn.Module):
    """Top-level fast eSEN model that adds density embedding, atom refs, and the energy head."""

    def __init__(
        self,
        backbone: eSEN_Fast_Backbone,
        density_embedding: DensityEmbedding,
        atom_ref_module: AtomRef,
        initial_guess_delta_module: InitialGuessDelta | None = None,
        gradient_module: InitialGuessDelta | None = None,
        feature_wise_rescaling_module: DimensionWiseRescaling | None = None,
    ) -> None:
        """Initialize the top-level fast eSEN model."""
        super().__init__()
        self.backbone = backbone
        self.density_embedding = density_embedding
        self.atom_ref_module = atom_ref_module
        self.initial_guess_delta_module = initial_guess_delta_module
        self.gradient_module = gradient_module
        self.feature_wise_rescaling_module = feature_wise_rescaling_module
        self.backbone.energy_block = None
        self.backbone.force_block = None
        self.backbone.direct_forces = False
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    @property
    def deterministic(self) -> bool:
        """Return whether the backbone runs in deterministic mode."""
        return self.backbone.deterministic

    @deterministic.setter
    def deterministic(self, deterministic: bool) -> None:
        """Set whether the backbone runs in deterministic mode."""
        self.backbone.deterministic = deterministic

    @staticmethod
    def build_data_dict(batch: OFBatch) -> torch_geometric.data.Data:
        """Convert an ``OFBatch`` into the compact graph object consumed by the backbone."""
        data_dict = torch_geometric.data.Data(
            edge_index=batch.edge_index,
            pos=batch.pos,
            batch=(
                batch.batch
                if batch.batch is not None
                else torch.zeros(len(batch.pos), dtype=torch.long, device=batch.pos.device)
            ),
            natoms=(
                batch.batch.unique(return_counts=True)[1]
                if batch.batch is not None
                else torch.tensor(len(batch.pos), device=batch.pos.device).unsqueeze(0)
            ),
            atomic_numbers=batch.atomic_numbers,
        ).to(batch.pos.device)
        return data_dict

    def forward(self, batch: OFBatch) -> ModelOutput:
        """Predict graph energies and optional coefficient deltas for one minibatch."""
        energy = self.atom_ref_module.sample_forward(batch)
        data = self.build_data_dict(batch)
        if self.feature_wise_rescaling_module is not None:
            rescaled_coeffs = self.feature_wise_rescaling_module(batch)
        else:
            rescaled_coeffs = batch.coeffs
        data["density_embedding"] = self.density_embedding(batch, rescaled_coeffs)
        emb = self.backbone(data)
        if self.initial_guess_delta_module is None:
            pred_diff = torch.zeros_like(batch.coeffs)
        else:
            pred_diff = self.initial_guess_delta_module(
                emb["node_embedding"],
                basis_function_ind=batch.basis_function_ind,
                coeff_ind_to_node_ind=batch.coeff_ind_to_node_ind,
            )

        node_energy = (
            self.energy_block(emb["node_embedding"].narrow(1, 0, 1).squeeze(1))
            .view(-1, 1, 1)
            .to(torch.float64)
        )
        energy.index_add_(0, data["batch"], node_energy.view(-1))
        return ModelOutput(pred_energy=energy, pred_diff=pred_diff)


class eSEN_Backbone(eSEN_Fast_Backbone):
    """Public backbone that uses the fast path for new models and a legacy path for old
    checkpoints."""

    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int | list[int] = 2,
        mmax: int | list[int] = 2,
        grid_resolution: int | None = None,
        otf_graph: bool = False,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = True,
        regress_forces: bool = True,
        regress_stress: bool = False,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "s2",
        mlp_type: str = "grid",
        use_envelope: bool = False,
        use_quaternion_wigner: bool = False,
        activation_checkpointing: bool = False,
        deterministic: bool = False,
    ):
        """Initialize the public eSEN backbone, dispatching kwargs to the fast backbone."""
        super().__init__(
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=lmax,
            mmax=mmax,
            grid_resolution=grid_resolution,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            cutoff=cutoff,
            edge_channels=edge_channels,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            direct_forces=direct_forces,
            regress_forces=regress_forces,
            regress_stress=regress_stress,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            act_type=act_type,
            mlp_type=mlp_type,
            use_envelope=use_envelope,
            activation_checkpointing=activation_checkpointing,
            deterministic=deterministic,
        )
        # Keep the deprecated flag so configs and checkpoints that still contain it instantiate.
        self.use_quaternion_wigner = use_quaternion_wigner

    def forward(self, data_dict) -> dict[str, torch.Tensor]:
        """Dispatch to the fast path for new modules and to the legacy layout for unpickled
        checkpoints."""
        edge_degree_embedding = getattr(self, "edge_degree_embedding", None)
        if not isinstance(edge_degree_embedding, EdgeDegreeEmbedding) and not any(
            isinstance(block, eSEN_Block) for block in getattr(self, "blocks", ())
        ):
            return super().forward(data_dict)

        if self.otf_graph:
            graph_dict = self.generate_graph(data_dict)
        else:
            if self.use_pbc:
                raise NotImplementedError(
                    "eSEN does not currently support `otf_graph=False` with `use_pbc=True`."
                )
            edge_source = data_dict["edge_index"][0]
            edge_target = data_dict["edge_index"][1]
            edge_distance_vec = data_dict["pos"].index_select(0, edge_source) - data_dict[
                "pos"
            ].index_select(0, edge_target)
            edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)
            graph_dict = {
                "atomic_numbers_full": data_dict["atomic_numbers_full"],
                "batch_full": data_dict["batch_full"],
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
                "node_offset": 0,
            }

        wigner_cache = {}
        for lmax_val in sorted(set(self.lmax_list)):
            _, wigner, wigner_inv = super().get_rotmat_and_wigner(
                graph_dict["edge_distance_vec"], lmax_val
            )
            wigner_cache[lmax_val] = (wigner, wigner_inv)

        edge_degree_out_mask = self.edge_degree_embedding.out_mask.to(
            device=wigner_cache[self.lmax_list[0]][1].device
        )
        edge_degree_wigner_inv = wigner_cache[self.lmax_list[0]][1].index_select(
            2, edge_degree_out_mask
        )
        edge_degree_wigner_inv = edge_degree_wigner_inv.index_select(
            2, self.edge_degree_embedding.mappingReduced.to_m_index
        )
        wigners_m, wigner_invs = [], []
        for block in self.blocks:
            wigner, wigner_inv = wigner_cache[block.lmax]
            block_out_mask = block.edge_wise.out_mask.to(device=wigner.device)
            block_wigner = wigner.index_select(1, block_out_mask)
            wigners_m.append(
                block_wigner.index_select(1, block.edge_wise.mappingReduced.to_m_index)
            )
            block_wigner_inv = wigner_inv.index_select(2, block_out_mask)
            wigner_invs.append(
                block_wigner_inv.index_select(2, block.edge_wise.mappingReduced.to_m_index)
            )

        scalar_embedding = self.sphere_embedding(data_dict["atomic_numbers"]).unsqueeze(1)
        if self.sph_feature_size == 1:
            x_message = scalar_embedding
        else:
            higher_order_zeros = scalar_embedding.new_zeros(
                data_dict["pos"].shape[0],
                self.sph_feature_size - 1,
                self.sphere_channels,
            )
            x_message = torch.cat((scalar_embedding, higher_order_zeros), dim=1)
        x_message = x_message + data_dict["density_embedding"]

        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])
        edge_source = graph_dict["edge_index"][0]
        edge_target = graph_dict["edge_index"][1]
        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"].index_select(0, edge_source)
        )
        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"].index_select(0, edge_target)
        )
        x_edge = torch.cat((edge_distance_embedding, source_embedding, target_embedding), dim=1)
        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            edge_degree_wigner_inv,
            graph_dict["node_offset"],
        )

        if graph_dict["edge_index"].shape[1] != 0:
            for i in range(self.num_layers):
                current_block_lmax = self.blocks[i].lmax
                num_spherical_harmonics = (current_block_lmax + 1) ** 2
                x_message_input = x_message[:, :num_spherical_harmonics, :]

                if self.activation_checkpointing:
                    x_message = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x_message_input,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigners_m[i],
                        wigner_invs[i],
                        graph_dict["node_offset"],
                        use_reentrant=False,
                    )
                else:
                    x_message = self.blocks[i](
                        x_message_input,
                        x_edge,
                        graph_dict["edge_distance"],
                        graph_dict["edge_index"],
                        wigners_m[i],
                        wigner_invs[i],
                        node_offset=graph_dict["node_offset"],
                    )

        x_message = self.norm(x_message)
        out = {"node_embedding": x_message}
        out.update(graph_dict)
        return out


class eSEN(eSEN_Fast):
    """Legacy public eSEN class name backed by the fast execution layout."""
