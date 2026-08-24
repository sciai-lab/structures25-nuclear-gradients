import torch
from torch.nn import Module
from torchvision.ops import MLP


class EdgeEmbedding(Module):
    """Edge embedding module used in the TensorFrames-style MoleculeNet."""

    def __init__(
        self,
        out_dim: int,
        radial_dim: int,
        angular_dim: int,
        hidden_layers: list[int],
        edge_attr_dim: int = 0,
        activation: Module = torch.nn.SiLU,
        dropout: float = 0.0,
        **mlp_kwargs,
    ):
        """Initializes an instance of EdgeEmbedding.

        Args:
            out_dim (int): The output dimension of the edge embedding.
            radial_dim (int): The dimension of the radial embedding.
            angular_dim (int): The dimension of the angular embedding.
            hidden_layers (list[int]): A list of integers representing the sizes of the hidden layers in the MLP.
            edge_attr_dim (int, optional): The dimension of the edge attributes. Defaults to 0.
            activation (Module, optional): The activation function to be used in the MLP. Defaults to torch.nn.SiLU.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            **mlp_kwargs: Additional keyword arguments to be passed to the MLP constructor.
        """
        super().__init__()
        self.radial_dim = radial_dim
        self.angular_dim = angular_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_layers = hidden_layers.copy()
        self.hidden_layers.append(out_dim)
        self.mlp = MLP(
            radial_dim + angular_dim + edge_attr_dim,
            self.hidden_layers,
            activation_layer=activation,
            **mlp_kwargs,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        radial_embedding: torch.Tensor,
        angular_embedding: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass of the EdgeEmbedding module.

        Args:
            radial_embedding (torch.Tensor): The radial embedding tensor.
            angular_embedding (torch.Tensor): The angular embedding tensor.
            edge_attr (torch.Tensor | None): The edge attribute tensor, or None if not available.

        Returns:
            torch.Tensor: The output tensor that combines the edge information.
        """

        edge_embedding = torch.cat([radial_embedding, angular_embedding], dim=1)

        if edge_attr is not None:
            edge_embedding = torch.cat([edge_embedding, edge_attr], dim=1)

        return self.dropout(self.mlp(edge_embedding))
