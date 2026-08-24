import os
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.trainer import Trainer
from loguru import logger


def get_trainer_loops_state_dict(trainer: Trainer) -> dict[str, Any]:
    """Returns the trainer's fit, validate, test and predict loop state dicts.

    This is required for manually saving a training checkpoint.
    """

    return {
        "fit_loop": trainer.fit_loop.state_dict(),
        "validate_loop": trainer.validate_loop.state_dict(),
        "test_loop": trainer.test_loop.state_dict(),
        "predict_loop": trainer.predict_loop.state_dict(),
    }


def save_checkpoint_manually(
    ckpt_path: str | Path,
    model: LightningModule,
    trainer: Trainer,
    pl_version: str = None,
    exist_ok: bool = True,
):
    """Manually saves a training checkpoint to the given checkpoint path, enabling saving the
    model's state prior to training.

    Args:
        ckpt_path (str | Path): The path where the checkpoint will be saved.
        model (LightningModule): The model whose state is to be saved.
        trainer (Trainer): The trainer object containing training information.
        pl_version (str, optional): The version of PyTorch Lightning being used. If None, it will be automatically determined. Defaults to None.
        exist_ok (bool, optional): If True, will not raise an error if the directory already exists. Defaults to True.
    """

    ckpt_path = Path(str(ckpt_path))
    if pl_version is None:
        try:
            import lightning.__version__ as pl_version
        except Exception as e:
            logger.error(e)

    checkpoint = {
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "pytorch-lightning_version": pl_version,
        "state_dict": model.state_dict(),
        "loops": get_trainer_loops_state_dict(trainer),
    }
    # save hyper-parameters
    if model.hparams and hasattr(model, "_hparams_name"):
        checkpoint[model.CHECKPOINT_HYPER_PARAMS_NAME] = model._hparams_name
        checkpoint[model.CHECKPOINT_HYPER_PARAMS_KEY] = dict(model.hparams)

    os.makedirs(ckpt_path.parent.absolute(), exist_ok=exist_ok)
    torch.save(checkpoint, ckpt_path)
