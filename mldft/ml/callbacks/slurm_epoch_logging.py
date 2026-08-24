import logging

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class SlurmEpochLogging(Callback):
    """Log coarse-grained epoch progress for non-interactive Slurm jobs."""

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log.info(f"Starting training epoch {trainer.current_epoch + 1}/{trainer.max_epochs}")

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        log.info(
            f"Starting validation after epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = []
        for metric_name in ("train/loss", "train/energy_loss", "train/gradient_loss"):
            metric_value = trainer.callback_metrics.get(metric_name)
            if metric_value is None:
                continue
            if hasattr(metric_value, "item"):
                metric_value = metric_value.item()
            metrics.append(f"{metric_name}={metric_value:.6g}")

        suffix = f" with {' '.join(metrics)}" if metrics else ""
        log.info(
            f"Finished training epoch {trainer.current_epoch + 1}/{trainer.max_epochs}{suffix}"
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        metrics = []
        for metric_name in ("val/loss", "val/energy_loss", "val/gradient_loss"):
            metric_value = trainer.callback_metrics.get(metric_name)
            if metric_value is None:
                continue
            if hasattr(metric_value, "item"):
                metric_value = metric_value.item()
            metrics.append(f"{metric_name}={metric_value:.6g}")

        suffix = f" with {' '.join(metrics)}" if metrics else ""
        log.info(
            f"Finished validation after epoch {trainer.current_epoch + 1}/{trainer.max_epochs}{suffix}"
        )
