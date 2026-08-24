from io import StringIO

import rich
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from rich.tree import Tree

from mldft.utils.hydra_utils import get_defaults_tree
from mldft.utils.rich_utils import convert_default_tree_to_rich


def dict_to_tree(
    data: dict | DictConfig, tree: Tree = None, name: str = "Config", **tree_kwargs
) -> Tree:
    """Convert a nested dictionary to a Rich Tree."""
    if tree is None:
        tree = Tree(name, **tree_kwargs)

    for key, value in data.items():
        if isinstance(value, (dict, DictConfig)):
            dict_to_tree(value, tree.add(f"[cyan]{key}[/cyan]"))
        else:
            tree.add(f"[yellow]{key}[/yellow]: {value}")

    return tree


def tree_to_string(tree: Tree) -> str:
    """Convert a Rich Tree to a string."""
    s = StringIO()
    rich.print(tree, file=s)
    s.seek(0)
    tree_str = s.read()
    s.close()
    return tree_str


def log_config_text_to_tensorboard(
    cfg: DictConfig, trainer: Trainer, add_defaults_tree=True
) -> None:
    """Log the config tree to TensorBoard.

    Args:
        cfg: A DictConfig composed by Hydra.
        trainer: The Lightning trainer.
        add_defaults_tree: Whether to add the Hydra defaults tree to TensorBoard.
    """
    # send hparams to all loggers
    for logger in trainer.loggers:
        # for tensorboard logger, additionally log hparams as text
        if isinstance(logger, TensorBoardLogger):
            hparams_text = tree_to_string(dict_to_tree(cfg))
            hparams_text = "```\n" + hparams_text + "\n```"
            logger.experiment.add_text("hparams", hparams_text)
            if add_defaults_tree:
                try:
                    defaults_tree = get_defaults_tree()
                    defaults_tree = convert_default_tree_to_rich(defaults_tree, guide_style="dim")
                    defaults_text = tree_to_string(defaults_tree)
                    defaults_text = "```\n" + defaults_text + "\n```"
                    logger.experiment.add_text("hydra_defaults_tree", defaults_text)
                except RuntimeError:
                    # If hydra is not initialized, we cannot get the defaults tree
                    # This happens insides some tests.
                    logger.experiment.add_text(
                        "hydra_defaults_tree",
                        "Failed printing config defaults tree with Rich. "
                        "Are you using a @hydra.main decorated function or are you "
                        "inside a `with initialize(..)` block?",
                    )
