"""Lightning module for the neural network."""

from typing import Any, Dict

import torch
import torch_geometric
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torchmetrics import MeanMetric, MinMetric

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.local_frames_module import LocalBasisModule
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.components.model_output import ModelOutput
from mldft.ml.models.components.sample_weighers import GroundStateOnlySampleWeigher
from mldft.ml.models.components.training_metrics import (
    MAEEnergy,
    MAEGradient,
    MAEInitialGuess,
)
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.denop.denoprunner_engine import DenopRunnerEngine
from mldft.ofdft.density_optimization import compute_density_optimization_metrics
from mldft.ofdft.optimizer import Optimizer
from mldft.utils import RankedLogger
from mldft.utils.log_utils.logging_mixin import locate_logging_mixins
from mldft.utils.plotting.plot_density_optimization import plot_density_optimization
from mldft.utils.sample import add_density_optimization_trajectories_to_sample


def get_layers(model: torch.nn.Module):
    children = list(model.children())
    return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]


class MLDFTLitModule(LightningModule):
    """The MLDFTLitModule class is a LightningModule that is used to wrap the neural network in the
    pytorch lightning framework."""

    def __init__(
        self,
        net: torch.nn.Module,
        basis_info: BasisInfo,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        target_key: str,
        loss_function: torch.nn.Module,
        compile: bool,
        variational: bool = True,
        metric_interval: int = 10,
        loss_log_interval: int = 10,
        enable_initial_guess_metrics: bool = False,
        scheduler_interval: str = "epoch",
        logging_mixin_interval: int | None = None,
        show_logging_mixins_in_progress_bar: bool = False,
        do_denop_val: bool = False,
        dataset_statistics: DatasetStatistics | None = None,
        density_optimizer: Optimizer | None = None,
        denop_settings: DictConfig[str, Any] | None = None,
        print_denop: bool = False,
    ) -> None:
        """Initializes the MLDFTLitModule object.

        Args:
            net (torch.nn.Module): the neural network
            basis_info (BasisInfo): the basis info object used for the dataset. Added here for logging purposes.
            optimizer (torch.optim.Optimizer): the optimizer to use for training
            scheduler (torch.optim.lr_scheduler): the learning rate scheduler to use for training
            loss_function (torch.nn.Module): the loss function to use for training
            target_key (str): the name of the target key. Added here to easily determine which
                targets were used for training afterward.
            compile (bool): whether to compile the model with :func:`torch.compile`
            variational (bool): whether the model is variational or not. If True, the network
                output must contain ``pred_energy`` and ``pred_diff``, and ``pred_gradient`` is
                computed via autograd. If False, the network must directly return
                ``pred_energy``, ``pred_gradient`` and ``pred_diff``.
            metric_interval (int): the interval (in steps) at which the metrics are calculated and logged.
            enable_initial_guess_metrics (bool): whether to include MAE initial-guess metrics.
                Defaults to False.
            scheduler_interval (str): "epoch"/"step" determines whether the iterations of the lr_scheduler work on an epoch or step basis
            logging_mixin_interval (int | None): the interval (in steps) at which the logging mixins are called.
                Defaults to None, which means that the logging mixins are not called.
            show_logging_mixins_in_progress_bar (bool): whether to show the values logged using the logging mixins in
                the progress bar. Defaults to False.
            do_denop_val (bool): whether to perform density optimization validation. Defaults to False.
            dataset_statistics (DatasetStatistics | None): the dataset statistics object, will only be used for the SAD guesser
                during density optimization validation. Defaults to None.
            density_optimizer (Optimizer | None): the optimizer to use for density optimization validation. Defaults to None.
            denop_settings (DictConfig[str, Any] | None): the settings for the density optimization validation. Defaults to None.
            print_denop (bool): whether to print the density optimization outputs. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # logger=false means that the hyperparameters will not be logged
        self.save_hyperparameters(logger=False)

        # save the neural network
        self.net = net
        self.target_key = target_key
        self.variational = variational

        self.scheduler_interval = scheduler_interval

        # define loss function
        self.loss_function = loss_function
        self.console_logger = RankedLogger(__name__, rank_zero_only=True)
        self.compute_nuclear_gradient = False
        if "nuclear_gradient_loss" in self.loss_function.loss_module_dict:
            self.compute_nuclear_gradient = True

        # metric objects for calculating and averaging accuracy across batches

        # The metric objects have to be initialized in this way, otherwise Lighting doesn't recognize them
        # Dicts of the metrics to iterate through when logging
        def create_metric_module_dict(split: str):
            """Creates a dictionary of metrics for a given split."""
            metrics = {
                f"{split}_metrics/mae_energy": MAEEnergy(mode="per molecule"),
                f"{split}_metrics_ground_state/mae_energy_ground_state": MAEEnergy(
                    mode="per molecule",
                    sample_weigher=GroundStateOnlySampleWeigher(),
                ),
                f"{split}_metrics_per_electron/mae_energy_per_electron": MAEEnergy(
                    mode="per electron",
                ),
                f"{split}_metrics/mae_gradient": MAEGradient(mode="per molecule"),
                f"{split}_metrics_ground_state/mae_gradient_ground_state": MAEGradient(
                    mode="per molecule",
                    sample_weigher=GroundStateOnlySampleWeigher(),
                ),
                f"{split}_metrics_per_electron/mae_gradient_per_electron": MAEGradient(
                    mode="per electron"
                ),
            }
            if enable_initial_guess_metrics:
                metrics.update(
                    {
                        f"{split}_metrics/mae_initial_guess": MAEInitialGuess(mode="per molecule"),
                        f"{split}_metrics_per_electron/mae_initial_guess_per_electron": MAEInitialGuess(
                            mode="per electron"
                        ),
                    }
                )
            return torch.nn.ModuleDict(metrics)

        self.train_metrics = create_metric_module_dict("train")
        self.val_metrics = create_metric_module_dict("val")
        self.test_metrics = create_metric_module_dict("test")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for each validation metric, also track the best value so far
        self.val_metrics_best = torch.nn.ModuleDict(
            {
                f"val_metrics_best/{key.split('/')[-1]}": MinMetric()
                for key in self.val_metrics.keys()
            }
        )

        # logging mixins
        self.logging_mixin_interval = logging_mixin_interval
        self.use_logging_mixins = self.logging_mixin_interval is not None
        self.val_logging_mixin_interval = (
            self.logging_mixin_interval // 5 if self.use_logging_mixins else None
        )
        if self.use_logging_mixins:
            self.logging_mixin_dict = locate_logging_mixins(self.net)
        else:
            self.logging_mixin_dict = {}
        self.show_logging_mixins_in_progress_bar = show_logging_mixins_in_progress_bar

        self.basis_info = basis_info
        if do_denop_val:
            self.dataset_statistics = dataset_statistics
            self.denop_settings = denop_settings

            if density_optimizer is None:
                raise ValueError("Density optimizer must be provided for denop validation.")

            self.denoprunner = DenopRunnerEngine(
                cfg=self.denop_settings,
                model=self,
                optimizer=density_optimizer,
                dataset_statistics=self.dataset_statistics,
                basis_info=self.basis_info,
                device="cpu",
                batch_size=1,  # TODO: change this to the batch size given in the trainer!
            )
            self.print_denop = print_denop

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()
        for metric_best in self.val_metrics_best.values():
            metric_best.reset()

    def activate_logging_mixins(self) -> None:
        """Activates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.activate_logging()

    def deactivate_logging_mixins(self) -> None:
        """Deactivates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.deactivate_logging()

    def forward(self, batch: Batch, always_retain_graph=False) -> ModelOutput:
        """Applies the forward pass of the model to the batch, and computes the energy gradients
        via backprop in the variational case (otherwise they are calculated directly).

        Args:
            batch (Batch): Batch object containing the data
            always_retain_graph (bool): Whether to always retain the graph of the energy
                backpropagation, independent of the training state. This is needed for the Hessian
                calculation. Defaults to False.

        Returns:
            ModelOutput: Typed model outputs containing ``pred_energy``,
                ``pred_gradient`` and ``pred_diff``.
        """
        if not self.variational:
            # model predicts energy, gradient and initial guess directly
            if not batch.coeffs.requires_grad:  # coeffs are not leafs in Hessian computation
                batch.coeffs.requires_grad = True  # for atomref image logging
            model_output = self.net(batch)
        else:
            # enable gradient calculation, even when we use torch.no_grad at inference time
            with torch.enable_grad():
                # set the flag for the gradients to be calculated
                if not batch.coeffs.requires_grad:  # coeffs are not leafs in Hessian computation
                    batch.coeffs.requires_grad = True
                if self.compute_nuclear_gradient:
                    batch.pos.requires_grad = True
                    if hasattr(batch, "lframes"):
                        lbm = LocalBasisModule()
                        if isinstance(batch, OFBatch):
                            batch.lframes = lbm(batch.pos, batch.atomic_numbers, batch=batch.batch)
                        else:
                            batch.lframes = lbm(batch.pos, batch.atomic_numbers)
                # calculate the forward pass
                model_output = self.net(batch)
                # calculate the gradients from the predicted energy with autograd
                # only at train time we create the graph to calculate the gradients of the gradient
                # retrain the graph in training, for the nuclear gradient loss, or if
                # always_retain_graph is True
                retain_graph = self.net.training or always_retain_graph
                if self.compute_nuclear_gradient:
                    # compute gradients w.r.t. both coeffs and positions in a single backward pass
                    pred_gradient, pred_nuc_grad_net = torch.autograd.grad(
                        model_output.pred_energy.sum(),
                        [batch.coeffs, batch.pos],
                        create_graph=self.net.training,
                        retain_graph=retain_graph,
                    )
                    model_output.pred_nuc_grad_net = pred_nuc_grad_net

                else:
                    pred_gradient = torch.autograd.grad(
                        model_output.pred_energy.sum(),
                        batch.coeffs,
                        create_graph=self.net.training,
                        retain_graph=retain_graph,
                    )[0]

            model_output.pred_gradient = pred_gradient

        return model_output

    def sample_forward(self, sample: OFData, always_retain_graph=False) -> OFData:
        """Applies the forward pass of the model to the batch.

        Args:
            sample: OFData object.
            always_retain_graph (bool): Whether to always retain the graph of the energy
                backpropagation, independent of the training state. This is needed for the Hessian
                calculation. Defaults to False.

        Returns:
            OFData: The batch with the model predictions added
        """
        model_output = self.forward(sample, always_retain_graph)
        sample.add_item("pred_energy", model_output.pred_energy, Representation.SCALAR)
        sample.add_item("pred_gradient", model_output.pred_gradient, Representation.GRADIENT)
        sample.add_item("pred_diff", model_output.pred_diff, Representation.VECTOR)
        if model_output.pred_energies_per_atom is not None:
            sample.add_item(
                "pred_energies_per_atom",
                model_output.pred_energies_per_atom,
                Representation.SCALAR,
            )
        if model_output.pred_energies_atom_ref is not None:
            sample.add_item(
                "pred_energies_atom_ref",
                model_output.pred_energies_atom_ref,
                Representation.SCALAR,
            )
        return sample

    @property
    def tensorboard_logger(self):
        """Get the tensorboard logger from the trainer."""
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
        return tb_logger

    def training_step(self, batch: Batch) -> dict:
        """Performs a single training step on a batch of data from the training set.

        Args:
            batch (Batch): A batch of data

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        log_loss_this_step = self.trainer.global_step % self.hparams.loss_log_interval == 0
        compute_metrics_this_step = self.trainer.global_step % self.hparams.metric_interval == 0
        use_logging_mixins_this_step = (
            self.use_logging_mixins and self.trainer.global_step % self.logging_mixin_interval == 0
        )

        # activate logging mixins if steps matches logging_mixin_interval
        if use_logging_mixins_this_step:
            self.activate_logging_mixins()

        model_output = self.forward(batch)

        # calculate and log the losses
        model_output.projected_gradient_difference = project_gradient_difference(
            model_output.pred_gradient, batch
        )
        weight_dict, loss_dict = self.loss_function(
            batch,
            model_output=model_output,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])

        if log_loss_this_step:
            self.log(
                "train_loss/total",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch.batch_size,
            )
            self.train_loss(loss)
            self.log_dict(
                loss_dict,
                logger=False,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=batch.batch_size,
            )
            log_dict = {f"train_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch.batch_size,
            )

        # update and log every metric
        if compute_metrics_this_step:
            for metric in self.train_metrics.values():
                metric(batch, model_output)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=False)

        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_train_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
            # For some reason, the below would significantly slow down training, even for iterations where it is not
            # called.
            # I think it's because lightning does not support logging at times other than on_step and on_epoch, i.e.
            # if you call log(..., on_step=True) once, the values will be logged at every step
            # Hence, we log directly to tensorboard (see above)
            # self.log_dict(
            #     mixin_log_dict,
            #     on_step=True,
            #     prog_bar=self.show_logging_mixins_in_progress_bar,
            #     batch_size=batch.batch_size
            # )
            self.deactivate_logging_mixins()
        return dict(
            loss=loss,
            model_output=model_output,
        )

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the train step
        # self.console_logger.info(f"Train loss: {self.train_loss.compute()}")

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        super().on_validation_epoch_start()
        self.activate_logging_mixins()  # log values from logging mixins at every step during validation

    def normal_validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Performs a single validation step on a batch of data from the validation set, identical
        to training steps.

        Args:
            batch (Batch): A batch of data
            batch_idx (int): The index of the batch
            dataloader_idx (int): (optional) id of dataloader in combined_loader

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        compute_metrics_this_step = batch_idx % self.hparams.metric_interval == 0
        use_logging_mixins_this_step = (
            self.use_logging_mixins
            and self.trainer.global_step % self.val_logging_mixin_interval == 0
        )

        model_output = self.forward(batch)
        # calculate the loss
        model_output.projected_gradient_difference = project_gradient_difference(
            model_output.pred_gradient, batch
        )
        weight_dict, loss_dict = self.loss_function(
            batch,
            model_output=model_output,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"val_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            add_dataloader_idx=False,
        )
        # log losses
        self.val_loss(loss)
        self.log(
            "val_loss/total",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_val_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
        # update and log metrics
        if compute_metrics_this_step:
            for metric in self.val_metrics.values():
                metric(batch, model_output)
            self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
            )

        return dict(
            loss=loss,
            model_output=model_output,
        )

    @staticmethod
    def _aggregate_denop_metrics(
        metric_dicts: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Aggregate per-sample denop metrics via a batch mean."""
        if len(metric_dicts) == 0:
            raise RuntimeError("Cannot aggregate denop metrics from an empty list.")

        aggregated_metrics: dict[str, float] = {}
        for key in metric_dicts[0]:
            values = []
            for metric_dict in metric_dicts:
                if key not in metric_dict:
                    raise KeyError(f"Missing denop metric key '{key}' in one of the results.")
                value = metric_dict[key]
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        raise ValueError(
                            f"Denop metric '{key}' must be scalar, got tensor shape {tuple(value.shape)}."
                        )
                    values.append(float(value.detach().cpu().item()))
                else:
                    values.append(float(value))
            aggregated_metrics[key] = sum(values) / len(
                values
            )  # average the metric across the batch
        return aggregated_metrics

    def denop_validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Performs a single validation step on a batch of data from the denop validation set <=>
        sample with added info.

        Args:
            batch (Batch): A batch of data.
            batch_idx (int): The index of the batch
            dataloader_idx (int): (optional) id of dataloader in combined_loader

        Returns:
            dict: A dict containing only the model predictions (optimized samples).
        """

        # DenopRunnerEngine is constructed before Lightning places the module on its final
        # device. Keep the runner device in sync to avoid CPU/CUDA index mismatches.
        has_deterministic = hasattr(self.net, "deterministic")
        prev_deterministic = self.net.deterministic if has_deterministic else None

        # Some models use stochastic edge-frame construction during training. For density
        # optimization validation we switch the top-level model flag to deterministic mode and
        if has_deterministic:
            self.net.deterministic = True

        self.denoprunner.device = self.device

        original_compute_nuclear_gradient = self.compute_nuclear_gradient

        original_max_cycle = self.denoprunner.optimizer.max_cycle

        input_samples = Batch.to_data_list(batch)

        if len(input_samples) == 0:
            raise RuntimeError("Denop validation received an empty batch.")

        try:
            self.to(OFDFT_DTYPE)
            self.compute_nuclear_gradient = False
            if self.trainer.sanity_checking:
                # Sanity check to not notice bugs only after a whole training epoch, but only do 2 cycles
                self.denoprunner.optimizer.max_cycle = 2

            results = self.denoprunner.run_samples(input_samples, progressbar=False)
            if len(results) == 0:
                raise RuntimeError("DenopRunner returned no results for denop validation batch.")

            metric_dicts = []
            energies_labels = []
            for state in results:
                energies_label = self.denoprunner.func_factory.get_energies_label(
                    state.sample, basis_info=self.basis_info
                )
                energies_labels.append(energies_label)
                metric_dicts.append(
                    compute_density_optimization_metrics(
                        state.callback,
                        energies_label,
                        self.denoprunner.convergence_criterion,
                    )
                )

            metric_dict = self._aggregate_denop_metrics(metric_dicts)
            # Log to progress bar
            self.log_dict(
                metric_dict,
                logger=False,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(results),
                add_dataloader_idx=False,
            )
            # Log to tensorboard
            log_dict = {f"val_denop_metrics/{key}": metric_dict[key] for key in metric_dict.keys()}
            self.log_dict(
                log_dict,
                on_step=False,
                on_epoch=True,
                batch_size=len(results),
                add_dataloader_idx=False,
            )

            first_state = results[0]
            primary_sample = first_state.sample
            if not self.trainer.sanity_checking:
                # Keep denop validation visualization cheap: plot only the first sample.
                callback = first_state.callback
                energies_label = energies_labels[0]
                sample = add_density_optimization_trajectories_to_sample(
                    first_state.sample, callback, energies_label, save_coeff_interval=1
                )
                gs_coeffs = transform_tensor_with_sample(
                    sample,
                    sample.ground_state_coeffs,
                    Representation.VECTOR,
                    invert=True,
                )
                # After transform go to cpu
                gs_coeffs = gs_coeffs.detach().cpu()
                sample = sample.cpu()
                figure = plot_density_optimization(
                    sample, gs_coeffs, stopping_index=sample.stopping_index
                )
                figure.set_dpi(70)
                tb_logger = self.tensorboard_logger
                if tb_logger is not None:
                    tb_logger.add_figure(
                        f"val_Denop_{batch_idx}",
                        figure,
                        global_step=self.trainer.global_step,
                    )

                return dict(model_outputs=sample, optimized_sample=sample)
        finally:
            self.denoprunner.optimizer.max_cycle = original_max_cycle
            self.compute_nuclear_gradient = original_compute_nuclear_gradient
            self.to(torch.float32)
            if has_deterministic:
                self.net.deterministic = prev_deterministic

        return dict(optimized_sample=primary_sample, model_outputs=primary_sample)

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Performs a single validation step on a batch of data matching the call signature of a CombinedLoader:
        The first loader will be used for normal validation, the second for denop.

        Args:
            batch (Batch): A batch of data
            batch_idx (int): The index of the batch
            dataloader_idx (int): The number of the dataloader

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        assert dataloader_idx in [0, 1]

        if dataloader_idx == 0:
            return self.normal_validation_step(
                batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )
        else:
            return self.denop_validation_step(
                batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
            )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        for metric, (metric_best_key, metric_best) in zip(
            self.val_metrics.values(), self.val_metrics_best.items()
        ):
            scalar = metric.compute()  # get current val acc
            metric_best.update(scalar)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(metric_best_key, metric_best.compute(), sync_dist=True, prog_bar=False)

        # deactivate logging mixins after validation
        self.deactivate_logging_mixins()

    def test_step(self, batch: Batch) -> None:
        """Performs a single test step on a batch of data from the test set.

        Args:
            batch (Batch): A batch of data
        """
        model_output = self.forward(batch)
        # calculate the loss
        model_output.projected_gradient_difference = project_gradient_difference(
            model_output.pred_gradient, batch
        )
        weight_dict, loss_dict = self.loss_function(
            batch,
            model_output=model_output,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"test_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        self.test_loss(loss)

        self.log(
            "test_loss/total",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        # update and log every metric
        for metric in self.test_metrics.values():
            metric(batch, model_output)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the test step
        # self.console_logger.info(f"Test loss: {self.test_loss.compute()}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"val"`, `"test"`, or `"predict"`.
        """
        # Doesn't work for double backward and doesn't provide a speed up currently.
        if self.hparams.compile:
            self.net = torch_geometric.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            Dict[str, Any]: A dict containing the configured optimizers and learning-rate schedulers
                to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if self.scheduler_interval == "epoch":
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            elif self.hparams.scheduler_interval == "step":
                steps_per_epoch = self.trainer.estimated_stepping_batches / self.trainer.max_epochs
                # This currently only works for the scheduler provided by sequential_scheduler in lr_scheduling.py
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer, steps_per_epoch=steps_per_epoch
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss/total",
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
