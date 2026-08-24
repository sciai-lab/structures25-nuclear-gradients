from typing import Union

import torch

from tensorframes.nn.mlp import MLPWrapped
from tensorframes.reps import Irreps, TensorReps
from tensorframes.reps.utils import extract_even_scalar_mask_from_reps, parse_reps


class EnergyMLPWrapped(torch.nn.Module):
    """A wrapped version of MLPWrapped to accept irreps and only apply the mlp on the even scalars.

    Attributes:
        in_reps (TensorReps, Irreps, str): The input representations for the MLP.
        even_scalar_mask (torch.Tensor): A mask indicating which input features are even scalars.
        mlp (MLPWrapped): The MLP model.
    """

    def __init__(self, in_reps: Union[TensorReps, Irreps, str], **mlp_kwargs):
        """Initializes the EnergyMLPWrapped class.

        Args:
            in_reps (Union[TensorReps, Irreps, str]): The input representations for the MLP.
            **mlp_kwargs: Additional keyword arguments to be passed to the MLPWrapped constructor.
        """
        super().__init__()
        self.in_reps = parse_reps(in_reps)
        self.even_scalar_mask = extract_even_scalar_mask_from_reps(self.in_reps)
        self.mlp = MLPWrapped(in_channels=int(self.even_scalar_mask.sum()), **mlp_kwargs)

    def forward(self, x):
        """Performs a forward pass through the MLP model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.mlp(x[..., self.even_scalar_mask])
