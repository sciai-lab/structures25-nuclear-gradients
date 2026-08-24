from typing import Union

from hydra.core.default_element import (
    ConfigDefault,
    DefaultsTreeNode,
    GroupDefault,
    VirtualRoot,
)
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from loguru import logger


def get_defaults_tree(
    remove_hydra_internal_configs: bool = True,
) -> Union[DefaultsTreeNode, ConfigDefault, GroupDefault, VirtualRoot]:
    """Get the defaults tree from the current Hydra config. Only works when Hydra is initialized.

    Args:
        remove_hydra_internal_configs: If True, will remove the internal hydra configs from the tree.
            Defaults to True.

    Returns:
        The defaults tree root node.
    """

    global_hydra = GlobalHydra.instance()
    if not global_hydra.is_initialized():
        raise RuntimeError(
            "GlobalHydra is not initialized. "
            "Use this function inside a @hydra.main decorated function or inside a `with initialize(..)` block."
        )

    hydra_config = HydraConfig.get()

    config_name = hydra_config.job.config_name

    run_mode_string = (
        "RUN"  # default to RUN, other options are MULTIRUN and SWEEP which we never use currently
    )
    for override in hydra_config.overrides.hydra:
        if override.startswith("hydra.mode="):
            run_mode_string = override.split("=")[1]

    # get command line overrides
    overrides = hydra_config.overrides.task

    config_loader = global_hydra.hydra.config_loader
    defaults = config_loader.compute_defaults_list(
        config_name=config_name,
        overrides=overrides,
        run_mode=getattr(RunMode, run_mode_string),
    )
    defaults_tree = (
        defaults.defaults_tree
    )  # this contains all the info we need (see https://hydra.cc/docs/advanced/defaults_list/#debugging-the-defaults-list), below we format it nicely.

    if remove_hydra_internal_configs:
        # we expect hydra/config to be the first of two nodes in the defaults tree
        # if not, we return the full tree
        if (
            len(defaults_tree.children) == 2
            and defaults_tree.children[0].node.path == "hydra/config"
        ):
            return defaults_tree.children[1]
        logger.warning(
            "Could not find hydra/config as first of two children in defaults tree. Returning full tree."
        )
    return defaults_tree
