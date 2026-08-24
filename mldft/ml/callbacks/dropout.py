import lightning as pl
from lightning import Callback

from mldft.utils.adjust_dropout import adjust_dropout


class DisableDropout(Callback):
    """Disable dropout by setting the dropout rate to 0.0 before training, using
    :func:`~mldft.utils.adjust_dropout.adjust_dropout`.

    Useful when continuing training from a checkpoint where dropout was used, but is not desired
    anymore.
    """

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Sets the dropout rate to 0.0."""
        adjust_dropout(pl_module, 0.0)


class AdjustDropoutProbabilityLinear(Callback):
    """Adjusts the dropout probability of all :class:`torch.nn.Dropout` layers during training,
    starting at a given value and epoch, and ending at a given value and epoch."""

    def __init__(self, start_p: float, end_p: float, start_epoch: int, end_epoch: int):
        """
        Args:
            start_p: The initial dropout probability.
            end_p: The final dropout probability.
            start_epoch: The epoch at which to start adjusting the dropout probability.
            end_epoch: The epoch at which to end adjusting the dropout probability.
        """
        super().__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Sets the dropout rate to the initial value."""
        adjust_dropout(pl_module, self.start_p)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Adjusts the dropout probability of all :class:`torch.nn.Dropout` layers."""
        epoch = trainer.current_epoch
        if epoch == self.end_epoch:
            adjust_dropout(pl_module, self.end_p)
            pl_module.log("z_dropout_probability", self.end_p, on_step=False, on_epoch=True)
        elif self.start_epoch <= epoch < self.end_epoch:
            p = self.start_p + (self.end_p - self.start_p) * (epoch - self.start_epoch) / (
                self.end_epoch - self.start_epoch
            )
            adjust_dropout(pl_module, p)
            pl_module.log("z_dropout_probability", p, on_step=False, on_epoch=True)
