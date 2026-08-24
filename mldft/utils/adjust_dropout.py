import torch
from loguru import logger


def adjust_dropout(model: torch.nn.Module, p: float) -> None:
    """Adjust the dropout probability of all dropout layers (instances of
    :class:`torch.nn.Dropout`) in the model.

    Args:
        model (torch.nn.Module): The model to adjust the dropout probability of.
        p (float): The new dropout probability.
    """
    n_dropout_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            n_dropout_layers += 1
            module.p = p
    logger.info(f"Adjusted dropout probability of {n_dropout_layers} dropout layers to {p}.")
