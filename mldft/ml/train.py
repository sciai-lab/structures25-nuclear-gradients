"""Main entry point for training."""

import os
import tempfile

# Limit BLAS/OpenMP thread pools unless the user already configured them.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import warnings
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Optional, Tuple

import e3nn
import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

from mldft.utils.data_staging import (
    get_local_rank,
    resolve_stage_path,
    stage_training_data,
    wait_for_stage_marker,
)
from mldft.utils.instantiators import instantiate_callbacks, instantiate_loggers
from mldft.utils.log_utils.config_in_tensorboard import log_config_text_to_tensorboard
from mldft.utils.manage_training_modes import set_model_weights, set_training_mode
from mldft.utils.reproducibility import create_code_backup

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
import torch

# this import registers custom omegaconf resolvers
import mldft.utils.omegaconf_resolvers  # noqa
from mldft.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _collect_stage_extra_paths(cfg: DictConfig) -> list[str]:
    """Collect statistics paths that must be available under staged DFT_DATA.

    We include both data and model paths because model statistics are often rewritten to no-basis
    variants used by SAD-related components.
    """
    extra_paths: list[str] = []
    for cfg_key in ("data", "model"):
        cfg_node = cfg.get(cfg_key)
        if cfg_node is None:
            continue
        dataset_statistics = cfg_node.get("dataset_statistics")
        if dataset_statistics is None:
            continue
        stats_path = dataset_statistics.get("path", None)
        if stats_path is not None:
            extra_paths.append(str(stats_path))
    return extra_paths


def maybe_stage_training_data(cfg: DictConfig) -> None:
    stage_cfg = cfg.get("stage_data")
    if stage_cfg is None or not bool(stage_cfg.get("enabled", False)):
        return

    src_cfg = stage_cfg.get("src_dft_data", None)
    if src_cfg is None:
        raise ValueError(
            "stage_data.enabled=true but stage_data.src_dft_data is not set. "
            "Set it explicitly or ensure DFT_DATA is configured."
        )
    src_root = resolve_stage_path(str(src_cfg), "stage_data.src_dft_data")
    dst_root = resolve_stage_path(
        str(stage_cfg.get("dst_dft_data", "$TMPDIR/mldft_data")), "stage_data.dst_dft_data"
    )

    split_file = stage_cfg.get("split_file", None)
    wait_timeout_s = int(stage_cfg.get("wait_timeout_s", 7200))

    sentinel = dst_root / ".mldft_data_staged"
    local_rank = get_local_rank()

    # Stage exactly once per node, then synchronize local ranks via a sentinel file.
    if local_rank == 0:
        log.info(f"Staging training data from {src_root} to {dst_root}")
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        if sentinel.exists():
            sentinel.unlink()
        stats = stage_training_data(
            src_dft_data=src_root,
            dst_dft_data=dst_root,
            dataset=str(stage_cfg.dataset),
            transform=str(stage_cfg.transform),
            target_key=str(stage_cfg.target_key),
            label_subdir=stage_cfg.label_subdir,
            split_file=None if split_file is None else str(split_file),
            include_denop_h5=bool(stage_cfg.include_denop_h5),
            extra_paths=_collect_stage_extra_paths(cfg),
        )
        sentinel.write_text("ok\n", encoding="utf-8")
        log.info(
            "Finished staging data: "
            f"copied={stats.copied}, skipped={stats.skipped}, missing_optional={stats.missing_optional}"
        )
    else:
        log.info(
            f"Waiting for local-rank-0 staging on this node (marker: {sentinel}, timeout={wait_timeout_s}s)."
        )
        wait_for_stage_marker(sentinel, wait_timeout_s)

    os.environ["DFT_DATA"] = str(dst_root)
    os.environ["MLDFT_STAGED_DFT_DATA"] = str(dst_root)
    with open_dict(cfg):
        cfg.stage_data.resolved_src_dft_data = str(src_root)
        cfg.stage_data.resolved_dst_dft_data = str(dst_root)


def configure_python_tempdir(output_dir: Path) -> None:
    """Keep Python temp files on the run filesystem.

    Lightning's HPC checkpoint save uses fsspec's local transaction support, which writes a temporary
    file via ``tempfile.mkstemp()`` before moving it to the final checkpoint path. On HoreKa the
    process environment often points ``$TMPDIR`` at node-local scratch, while run outputs live on
    ``/home``. That cross-device move can fail during Slurm requeue handling. Setting
    ``tempfile.tempdir`` keeps Python-managed temp files on the same filesystem as the run output
    without changing Slurm's ``$TMPDIR`` that we use for dataset staging.
    """
    python_tmpdir = output_dir / ".tmp"
    python_tmpdir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(python_tmpdir)
    os.environ["MLDFT_PYTHON_TMPDIR"] = str(python_tmpdir)


def _latest_epoch_checkpoint_path(output_dir: Path) -> Path | None:
    """Return the newest epoch-end checkpoint if one exists in the run directory."""
    checkpoint_dir = output_dir / "checkpoints"
    candidates: list[tuple[int, float, Path]] = []
    for checkpoint_path in checkpoint_dir.glob("last_epoch_*.ckpt"):
        epoch_str = checkpoint_path.stem.rsplit("_", 1)[-1]
        if not epoch_str.isdigit():
            continue
        candidates.append((int(epoch_str), checkpoint_path.stat().st_mtime, checkpoint_path))
    if not candidates:
        return None
    return max(candidates)[-1]


def resolve_requeue_fit_checkpoint(
    ckpt_path: str | Path | None, output_dir: Path
) -> str | Path | None:
    """Resume from the latest regular epoch checkpoint when restarting in an existing run dir.

    For submitit-based requeues we intentionally rely on the regular ``last_epoch_*.ckpt`` files
    instead of Lightning's HPC checkpoints. Fresh runs have no epoch checkpoint yet, so they still
    start from scratch.
    """
    if ckpt_path is not None:
        return ckpt_path

    latest_epoch_ckpt = _latest_epoch_checkpoint_path(output_dir)
    if latest_epoch_ckpt is not None:
        restart_count = int(os.getenv("SLURM_RESTART_COUNT", "0"))
        submitit_executor = os.getenv("SUBMITIT_EXECUTOR")
        restart_reason = (
            f"SLURM_RESTART_COUNT={restart_count}"
            if restart_count > 0
            else (
                f"submitit executor={submitit_executor}"
                if submitit_executor is not None
                else "existing epoch checkpoint"
            )
        )
        log.info(
            f"Detected {restart_reason}. Resuming from latest epoch checkpoint {latest_epoch_ckpt}."
        )
        return str(latest_epoch_ckpt)

    return None


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    training_mode = cfg.get("training_mode")
    cfg, ckpt_path, weight_ckpt_path = set_training_mode(training_mode, cfg)
    maybe_stage_training_data(cfg)
    # Apply optional utilities (warnings/tags/pretty-print) on the final effective config.
    extras(cfg)

    # Save config as unresolved yaml for checkpoint loading later
    output_dir = Path(cfg.paths.output_dir)
    configure_python_tempdir(output_dir)
    fit_ckpt_path = resolve_requeue_fit_checkpoint(ckpt_path, output_dir)
    yaml_path = output_dir / "hparams.yaml"
    yaml_path_resolved = output_dir / "hparams_resolved.yaml"
    # Copy source code for reproducibility
    if cfg.extras.get("save_code_backup", True):
        create_code_backup(output_dir)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    else:
        seed = randint(0, 2**32 - 1)
        L.seed_everything(seed, workers=True)
        with open_dict(cfg):
            cfg.seed = seed
    log.info(f"Logging config to {yaml_path}")
    OmegaConf.save(cfg, yaml_path, resolve=False)
    OmegaConf.save(cfg, yaml_path_resolved, resolve=True)
    # This might be needed in the future, if error "too many open files" occurs
    torch.multiprocessing.set_sharing_strategy("file_system")
    # This is set to highest, since for some new gpus high leads to discrepancies between training and evaluation in density optimization
    torch.set_float32_matmul_precision("highest")
    # If precision is 64, set default dtype to float64 which changes the dtype in the last to_torch transform
    if cfg.trainer.precision == 64:
        torch.set_default_dtype(torch.float64)
    # Lightning's PyTorchProfiler registers forward hooks on submodules, which is incompatible
    # with e3nn-generated ScriptModules.
    if cfg.trainer.get("profiler") is not None:
        e3nn.set_optimization_defaults(jit_script_fx=False)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # log config to tensorboard as text
    log_config_text_to_tensorboard(cfg, trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    # If there is no hparams.yaml already this saves a hparams.yaml from lightning logger, but we have one already.
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    # Should safely load model weights
    set_model_weights(training_mode, model, ckpt_path, weight_ckpt_path)

    # suppress warning about low num_workers in the 'val_dataloader' — 1 worker is intentional
    warnings.filterwarnings(
        "ignore",
        category=PossibleUserWarning,
        message=".*'val_dataloader' does not have many workers.*",
    )
    if cfg.get("validate_before_train"):
        log.info("Starting validation before training!")
        trainer.num_sanity_val_steps = 0
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=fit_ckpt_path)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=fit_ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("validate"):
        log.info("Starting validation!")
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    val_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs/ml", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
