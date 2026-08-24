from typing import Union

import torch
from torch.nn import Module
from torch_geometric.nn import global_add_pool

from tensorframes.lframes import ThreeNNLFrames
from tensorframes.lframes.classical_lframes import IdentityLFrames
from tensorframes.lframes.learning_lframes import (
    LearnedGramSchmidtLFrames,
    WrappedLearnedLFrames,
)
from tensorframes.nn.local_global import FromGlobalToLocalFrame
from tensorframes.nn.tensormace import TensorMACE
from tensorframes.reps import Irreps, TensorReps
from tensorframes.reps.utils import extract_even_scalar_mask_from_reps


class TensorMACEnet(torch.nn.Module):
    """TensorFrames-style MACE architecture for molecular property prediction."""

    def __init__(
        self,
        radial_module: Module,
        angular_module: Module,
        edge_embedding_module: Module,
        layers: int,
        reps: Union[TensorReps, Irreps],
        output_reps: Union[TensorReps, Irreps],
        num_types: int,
        edge_emb_dim: int,
        hidden_dim: int,
        l_frames_module: Module = None,
        input_reps: Irreps = None,
        max_order: int = 3,
        dropout: float = 0.05,
        bias: bool = False,
        atom_wise: bool = False,
        global_out: bool = True,
    ) -> None:
        """Initializes the TensorMACEnet.

        Args:
            l_frames_module (Module): The module for generating local reference frames.
            radial_module (Module): The module for computing radial features.
            angular_module (Module): The module for computing angular features.
            edge_embedding_module (Module): The module for computing edge embeddings.
            layers (int): The number of layers in the model.
            reps (Union[TensorReps, Irreps]): The tensor representations or irreducible representations used in the model.
            output_reps (Union[TensorReps, Irreps]): The tensor representations or irreducible representations used for the output.
            num_types (int): The number of atom types.
            edge_emb_dim (int): The dimension of the edge embeddings.
            hidden_dim (int): The dimension of the hidden layers.
            max_order (int, optional): The maximum order of the tensor product. Defaults to 3.
            dropout (float, optional): The dropout rate. Defaults to 0.05.
            bias (bool, optional): Whether to include bias terms. Defaults to False.
            atom_wise (bool, optional): Whether to compute atom-wise outputs. Defaults to False.
            global_out (bool, optional): Whether to compute a global output. Defaults to True.
        """
        super().__init__()
        self.radial_module = radial_module
        self.angular_module = angular_module
        self.edge_embedding_module = edge_embedding_module
        self.layers = layers
        self.global_out = global_out

        self.module_list = torch.nn.ModuleList()
        self.output_linear = torch.nn.ModuleList()

        self.input_reps = input_reps

        self.from_global_to_local = FromGlobalToLocalFrame(reps=input_reps)

        self.scalar_x_mask = extract_even_scalar_mask_from_reps(self.input_reps)

        if l_frames_module is not None:
            self.l_frames_module = l_frames_module
        else:
            assert self.input_reps is not None, "input_reps must be provided."
            self.l_frames_module = WrappedLearnedLFrames(
                in_reps=self.input_reps,
                hidden_channels=[256, 256],
                radial_module=self.radial_module,
                max_radius=10.0,
                exceptional_choice="random",
                activation_layer=torch.nn.SiLU,
            )

        for i in range(self.layers):
            self.module_list.append(
                TensorMACE(
                    in_tensor_reps=reps,
                    out_tensor_reps=reps,
                    edge_emb_dim=edge_emb_dim,
                    num_types=num_types,
                    hidden_dim=hidden_dim,
                    max_order=max_order,
                    dropout=dropout,
                    bias=bias,
                    atom_wise=atom_wise,
                )
            )
            if i != layers - 1:
                self.output_linear.append(torch.nn.Linear(reps.dim, output_reps.dim))

    def forward(self, sample):
        """Forward pass of the TensorMaceNet model.

        Args:
            sample (Data): The input sample containing the following attributes:
                - pos (Tensor): Node positions.
                - edge_index (LongTensor): Graph edge indices.
                - x (Tensor): Node features.
                - z (Tensor): Atomic indices.
                - batch (LongTensor): Batch indices.
                - edge_attr (Tensor): Edge attributes.

        Returns:
            Tensor: The output tensor of the forward pass.
        """

        radial_out = self.radial_module(sample.pos, sample.edge_index)
        x = sample.x
        if isinstance(self.l_frames_module, WrappedLearnedLFrames):
            x, l_frames = self.l_frames_module(x=x, pos=sample.pos, batch=sample.batch)
        elif isinstance(self.l_frames_module, LearnedGramSchmidtLFrames):
            tmp = torch.nn.functional.one_hot(sample.z.to(torch.int64), num_classes=5)

            l_frames = self.l_frames_module(
                pos=sample.pos,
                x=tmp,
                radial=radial_out,
                edge_index=sample.edge_index,
                edge_attr=sample.edge_attr,
                batch=sample.batch,
            )
            x = self.from_global_to_local(x=x, lframes=l_frames)
        elif isinstance(self.l_frames_module, ThreeNNLFrames):
            l_frames = self.l_frames_module(pos=sample.pos, batch=sample.batch)
        elif isinstance(self.l_frames_module, IdentityLFrames):
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

        output_cache = []

        for i, layer in enumerate(self.module_list):
            x = layer(
                x=x,
                types=sample.z,
                lframes=l_frames,
                edge_index=sample.edge_index,
                edge_embedding=edge_embedding,
                batch=sample.batch,
            )
            if i != self.layers - 1:
                output_cache.append(self.output_linear[i](x))

        # sum over the output cache and add it to out
        out = x
        for output in output_cache:
            out += output

        if self.global_out:
            out = global_add_pool(out, sample.batch)

        return out
