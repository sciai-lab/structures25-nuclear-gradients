"""Entry point for computing natural reparametrization transformations
(see [M-OFDFT]_, Section `4.3 <https://arxiv.org/html/2309.16578v2#S4.SS3>`_ - Natural Reparametrization
and [STRUCTURES25]_, Supplementary Section `S.3 <https://arxiv.org/html/2503.00443v2#Ax1.SS3>`_- Enhancement Modules)

The transformation matrices are computed from the training set and saved to disk, at the
``output_dir`` of the run, as a ``.zarr`` file.
Additionally, if ``extra_save_path`` is specified, the transformation dictionaries are also saved
to that path. By default, this is set to the location of where the data is loaded from in the given
training configuration.

By default, no transforms are applied to the dataset.

To run e.g. with local frames, specify the following in ``configs/ml/data/default.yaml``
(or override from any other configuration file):

.. code-block:: yaml

 data:
  datamodule:
    transforms:
     - _target_: mldft.ml.data.components.convert_transforms.AddOverlapMatrix
     - _target_: mldft.ml.data.components.convert_transforms.ToTorch
     - _target_: mldft.ml.data.components.convert_transforms.AddAtomCooIndices
     - _target_: mldft.ml.data.components.basis_transforms.ToLocalFrames
       matrices: ["overlap_matrix"]
     - _target_: mldft.ml.data.components.convert_transforms.ToNumpy

Warning:
    Make sure that the transforms are consistent with what you plan to do during training!

To use e.g. during training, uncomment the corr. lines in data/default.yaml.

Example:
    >>> python mldft/ml/compute_nat_rep_transformations.py data=md17
"""

# TODO: read transforms from data config as well and add ToNumpy programmatically as final transform

from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from omegaconf import DictConfig

from mldft.ml.data.datamodule import OFDataModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/config.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/config.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

# this import registers custom omegaconf resolvers
import mldft.utils.omegaconf_resolvers  # noqa
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.models.components.atomwise_natural_reparametrization import (
    AtomwiseNaturalReparametrization,
)
from mldft.ml.models.components.shellwise_natural_reparameterization import (
    ShellwiseNaturalReparameterization,
)
from mldft.utils import RankedLogger, extras, task_wrapper
from mldft.utils.utils import set_default_torch_dtype

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
@set_default_torch_dtype(torch.float64)
def compute_nat_rep_transformations(cfg: DictConfig):
    """Compute natural reparametrization matrices and save them to disk."""

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # This might be needed in the future, if error "too many open files" occurs
    torch.multiprocessing.set_sharing_strategy("file_system")

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: OFDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    # sets up train, val, test datasets
    datamodule.setup(stage="fit")

    basis_info: BasisInfo = datamodule.basis_info

    # this could be done with hydra instantiate but I don't know how. Since it doesnt’s seem likely
    # that we will add more scopes for the nat rep, I think this is fine.
    if cfg.scope == "atomwise":
        natural_reparametrization = AtomwiseNaturalReparametrization.from_basis_info(
            basis_info, orthogonalization=cfg.orthogonalization
        )
    elif cfg.scope == "shellwise":
        natural_reparametrization = ShellwiseNaturalReparameterization.from_basis_info(
            basis_info, orthogonalization=cfg.orthogonalization
        )
    else:
        raise ValueError(f"Unknown scope {cfg.scope}")

    log.info(f"{natural_reparametrization=}")
    filename = (
        str(cfg.scope) + str(cfg.orthogonalization) + "_natural_reparametrization_dict.zarr.zip"
    )
    out_path = Path(cfg.paths.output_dir) / filename
    log.info(f"Saving {cfg.scope} natural reparametrization dictionary to {out_path}")

    natural_reparametrization.save_dict(out_path, exist_ok=False)

    extra_save_path = cfg.get("extra_save_path", False)
    if extra_save_path:
        extra_save_path = Path(extra_save_path)
        extra_save_path.parent.mkdir(exist_ok=True)
        try:
            extra_save_path.parent.chmod(0o770)
        except PermissionError:
            log.warning(f"Could not set permissions for {extra_save_path.parent}. ")
        if extra_save_path.exists() and not cfg.get("overwrite", False):
            raise FileExistsError(
                f"Natural Reparametrization transformations already exist at {extra_save_path}."
            )
        log.info(
            f"""Saving natural reparametrization transformations next to the dataset
              at {extra_save_path}"""
        )
        natural_reparametrization.save_dict(extra_save_path, exist_ok=cfg.overwrite)
        extra_save_path.chmod(0o770)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
    }

    metric_dict = {"natural_reparametrization": natural_reparametrization}

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3",
    config_path="../../configs/ml",
    config_name="natural_reparametrization.yaml",
)
def main(cfg: DictConfig):
    """Entry point for computing natural reparametrization.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric_dict, object_dict = compute_nat_rep_transformations(cfg)
    nat_rep_dict = metric_dict["natural_reparametrization"]

    return nat_rep_dict

    # Probably do not need a return for this task


if __name__ == "__main__":
    main()
