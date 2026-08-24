from typing import Union

import torch
from torch.nn import Module
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    LayerNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    radius_graph,
)

from tensorframes.lframes import ThreeNNLFrames
from tensorframes.lframes.learning_lframes import (
    LearnedGramSchmidtLFrames,
    WrappedLearnedLFrames,
)
from tensorframes.nn.local_global import FromLocalToGlobalFrame
from tensorframes.nn.mlp import MLP
from tensorframes.nn.tensorformer import TensorFormer
from tensorframes.nn.tensormace import TensorMACE
from tensorframes.reps import Irreps, TensorReps


class MoleculeNet(Module):
    """Neural network module for molecular data processing in the TensorFrames framework.

    This class is used to build tensor frames based networks with different buildings blocks on the
    config level. It handles the computation of lframes, radial and angular features, and the
    application of the network layers and the transformation of global features into local ones and
    back.
    """

    def __init__(
        self,
        l_frames_module: Module,
        radial_module: Module,
        angular_module: Module,
        layers: torch.nn.ModuleList,
        global_out: bool,
        pooling_method: str,
        output_reps: Union[TensorReps, Irreps],
        edge_embedding_module: Module | None = None,
        radial_cutoff: float | None = None,
    ):
        """Initialize the MoleculeNet module.

        Args:
            l_frames_module (Module): Module for computing local frames.
            radial_module (Module): Module for computing radial features.
            angular_module (Module): Module for computing angular features.
            layers (torch.nn.ModuleList): List of layers in the network.
            global_out (bool): Flag indicating whether to include global output.
            pooling_method (str): Method for pooling local features.
            output_tensor_reps (Union[TensorReps, Irreps]): Tensor representations of the output.
            edge_embedding_module (Module | None, optional): Module for computing edge embeddings. Defaults to None.
            radial_cutoff (float | None, optional): Radial cutoff for locally connected edge index. Defaults to None.
        """
        super().__init__()

        self.l_frames_module = l_frames_module
        self.radial_module = radial_module
        self.angular_module = angular_module
        self.layers = layers
        self.global_out = global_out
        self.pooling_method = pooling_method
        self.output_reps = output_reps
        self.edge_embedding_module = edge_embedding_module
        self.radial_cutoff = radial_cutoff
        self.from_local_to_global = FromLocalToGlobalFrame(output_reps)

    def forward(self, sample: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass of the MoleculeNet module.

        Args:
            sample (Union[Data, Batch]): Input sample containing molecular data.

        Returns:
            x (torch.Tensor): Output tensor of the module.
        """
        if self.radial_cutoff is not None:
            # use locally connected edge index instead of full graph:
            sample.edge_index = radius_graph(
                x=sample.pos, r=self.radial_cutoff, batch=sample.batch
            )

        radial_out = self.radial_module(pos=sample.pos, edge_index=sample.edge_index)
        x = sample.x
        if isinstance(self.l_frames_module, WrappedLearnedLFrames):
            x, l_frames = self.l_frames_module(x=x, pos=sample.pos, batch=sample.batch)
        elif isinstance(self.l_frames_module, LearnedGramSchmidtLFrames):
            l_frames = self.l_frames_module(
                pos=sample.pos,
                x=sample.x,
                radial=radial_out,
                edge_index=sample.edge_index,
                edge_attr=sample.edge_attr,
            )
        elif isinstance(self.l_frames_module, ThreeNNLFrames):
            l_frames = self.l_frames_module(pos=sample.pos)
        else:
            raise NotImplementedError(f"{self.l_frames_module} is not supported.")

        angular_out = self.angular_module(
            pos=sample.pos, edge_index=sample.edge_index, lframes=l_frames
        )

        if self.edge_embedding_module is not None:
            edge_embedding = self.edge_embedding_module(
                radial_embedding=radial_out,
                angular_embedding=angular_out,
                edge_attr=sample.edge_attr,
            )

        for i, layer in enumerate(self.layers):
            if isinstance(layer, TensorFormer):
                x = layer(
                    x=x,
                    lframes=l_frames,
                    edge_index=sample.edge_index,
                    pos=sample.pos,
                    batch=sample.batch,
                    edge_embedding=edge_embedding,
                )
            elif isinstance(layer, TensorMACE):
                x = layer(
                    x=x,
                    lframes=l_frames,
                    edge_index=sample.edge_index,
                    edge_embedding=edge_embedding,
                    batch=sample.batch,
                )
            elif isinstance(layer, MLP):
                x = layer(x=x)
            elif isinstance(layer, LayerNorm):
                x = layer(x=x, batch=sample.batch)
            else:
                x = layer(x=x)

        x = self.from_local_to_global(x, l_frames)

        if self.global_out:
            if self.pooling_method == "add":
                x = global_add_pool(x, sample.batch)
            elif self.pooling_method == "mean":
                x = global_mean_pool(x, sample.batch)
            elif self.pooling_method == "max":
                x = global_max_pool(x, sample.batch)

        return x
