from __future__ import annotations

import shutil
import traceback
from collections.abc import Collection
from contextlib import nullcontext

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rootutils import rootutils
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.denop import (
    prepare_config,
    prepare_data,
    prepare_paths,
    prepare_runtime,
)
from mldft.ofdft.denop.denoprunner_engine import DenopRunnerEngine, MoleculeState
from mldft.ofdft.denop.evaluation import DenopEvaluator, DenopPlotter
from mldft.ofdft.denop.functional_bridge import (
    PreparedFunctionalContext,
)
from mldft.ofdft.denop.optimizer_session import (
    is_legacy_optimizer,
)
from mldft.ofdft.density_optimization import (
    compute_density_optimization_metrics,
    density_optimization_with_label,
)
from mldft.ofdft.optimizer import Optimizer
from mldft.utils import extras
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.instantiators import instantiate_model
from mldft.utils.rich_utils import print_config_tree
from mldft.utils.seeding import seed_all_rngs

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class DenopRunner(DenopRunnerEngine):
    """Single-device denop execution runner."""

    def __init__(self, cfg: DictConfig):
        """Initialize denop runtime state, IO paths, and execution components.

        Args:
            cfg (DictConfig): Hydra/OmegaConf runtime configuration.

        Raises:
            ValueError: If negative-density penalty is enabled while grid operations are disabled.
        """
        # Side effects:
        # - Resolves and mutates runtime config fields (for example checkpoint and split paths).
        # - Creates output directories and writes run metadata/artifacts.
        # - Loads models, dataset objects, and evaluator/plotter components into runner state.

        # Resolve the user-facing config first so all later setup steps operate
        # on a normalized runtime configuration.
        self.config, self.model_config, self.ckpt_path, self.guess_ckpt_path = (
            prepare_config.prepare_config(cfg)
        )
        denop_cfg = self.config.get("denop_settings")

        if denop_cfg.get(
            "negative_integrated_density_penalty_weight", 0.0
        ) != 0.0 and not self.config.get("enable_grid_operations", False):
            raise ValueError(
                "Cannot use negative integrated density penalty without grid operations. "
                "Set enable_grid_operations to True."
            )

        device = self.config.get("device")
        self.device = device
        self.model_dtype = str_to_torch_float_dtype(self.config.get("model_dtype", torch.float64))
        self.deterministic = self.config.get("deterministic")

        log_file = prepare_runtime.prepare_logger(self.config)

        OmegaConf.save(self.config, log_file.parent / "hparams_resolved.yaml", resolve=True)
        self.save_dir = log_file.parent

        # Prepare the output directory structure before loading any data so all
        # downstream components can write their artifacts immediately.
        self.save_dir.mkdir(exist_ok=True)
        (self.save_dir / "sample_trajectories").mkdir(exist_ok=True)

        _, self.paths, self.num_scf_iterations_per_path = prepare_paths.prepare_paths(
            self.config.get("denop_dir"),
            self.config.get("n_molecules"),
            self.config.get("molecule_ids"),
            self.config.get("molecule_choice"),
            self.config.get("continue_denop"),
            self.config.get("split_file_path"),
            self.config.get("split"),
            self.config.get("seed"),
        )

        self.basis_info: BasisInfo = instantiate(self.model_config.data.basis_info)
        torch.save(self.basis_info, self.save_dir / "sample_trajectories" / "basis_info.pt")

        # these parts are specific to `DenopRunenr` and extend the parent class, which is agnostic to model/data loading and just implements the optimization loop and related logic
        transforms = prepare_data.prepare_transforms(
            self.model_config,
            instantiate(self.model_config.data.transforms),
            self.basis_info,
            denop_cfg.get("negative_integrated_density_penalty_weight", 0.0),
            denop_cfg.get("grid_level"),
            denop_cfg.get("grid_prune"),
        )

        self.dataset_statistics_path, self.dataset, self.dataloader = (
            prepare_data.prepare_dataset_and_loader(
                self.config,
                self.model_config,
                denop_cfg,
                self.paths,
                self.num_scf_iterations_per_path,
                self.basis_info,
                transforms,
            )
        )

        self.batch_size = int(self.config.get("batch_size"))
        if self.batch_size > len(self.dataset):
            logger.warning(
                f"Batch size {self.batch_size} is greater than dataset size {len(self.dataset)}. "
                f"Setting batch size to {len(self.dataset)}."
            )
            self.batch_size = len(self.dataset)

        if self.batch_size <= 0:
            logger.warning("Batch size must be positive. Setting batch size to 1.")
            self.batch_size = 1

        optimizer: Optimizer = instantiate(self.config.get("density_optimizer"))
        fail_fast = bool(self.config.get("fail_fast", False))

        lightning_module = instantiate_model(
            self.ckpt_path,
            device,
            self.model_dtype,
            self.deterministic,
            batch_size=(
                self.batch_size if is_legacy_optimizer(optimizer) else 0
            ),  # use the BatchedInferenceModule to have batched model forward on the legacy path at least
        )

        super().__init__(
            cfg=denop_cfg,
            model=lightning_module,
            optimizer=optimizer,
            dataset_statistics=DatasetStatistics(self.dataset_statistics_path),
            basis_info=self.basis_info,
            device=device,
            batch_size=self.batch_size,
            fail_fast=fail_fast,
        )

        # this gets originally instantiated in the parent constructor but needs to be re-instantiated here after the super().__init__() call
        # Load the projection model first, then build the functional bridge used
        # by both the legacy and lockstep execution paths.
        if self.guess_ckpt_path is not None:
            self.proj_minao_module = instantiate_model(
                self.guess_ckpt_path, device, self.model_dtype, self.deterministic
            )
        else:
            self.proj_minao_module = instantiate_model(
                self.ckpt_path, device, self.model_dtype, self.deterministic
            )
        self.finalize_at_least_n = denop_cfg.get("finalize_at_least_n", self.batch_size)

        # The below is specific to this class and extends the parent
        # add plotter if necessary, then add evaluator
        self.plot_every_n = self.config.get("plot_every_n")
        if self.plot_every_n is not None and self.plot_every_n > 0:
            # build plotter
            plotter = DenopPlotter(
                save_dir=self.save_dir,
                plotting_functions=(
                    [instantiate(func) for func in cfg.get("subplot_list")]
                    if cfg.get("subplot_list") is not None
                    else []
                ),
                enable_grid_plots=self.config.get("enable_grid_plots", False),
                basis_info=self.basis_info,
                save_individual_plots=self.config.get("save_individual_plots", False),
                fail_fast=self.config.get("fail_fast", False),
                enable_grid_operations=self.config.get("enable_grid_operations", False),
                compute_final_hessian=self.config.get("compute_final_hessian", False),
            )
        else:
            plotter = None

        self.evaluator = DenopEvaluator(
            save_dir=self.save_dir,
            n_molecules=len(self.paths),
            compute_grid_metrics=cfg.get("compute_grid_metrics"),
            compare_to_ks=cfg.get("compare_to_ks"),
            metrics_grid_level=cfg.get("metrics_grid_level"),
            metrics_grid_prune=cfg.get("metrics_grid_prune"),
            swarm_plot_subsample=self.config.get("swarm_plot_subsample", 1.0),
            plot_summary=self.config.get("plot_summary", True),
            save_coeff_interval=self.save_coeff_interval,
            compute_final_hessian=self.config.get("compute_final_hessian", False),
            plotter=plotter,
            plot_every_n=self.plot_every_n if self.plot_every_n is not None else 0,
        )
        self._profiler = None

    def _finalize_run(self) -> None:
        """Run the evaluator after optimization if finalization is enabled."""
        logger.success("Density optimization completed successfully. Evaluating metrics.")
        try:
            self.evaluator.evaluate()
        except Exception as exc:
            # Avoid passing non-pickleable exception objects (e.g. Boost.Python errors)
            # through multiprocessing logger queues.
            logger.error("Denop evaluation failed: {}\n{}", exc, traceback.format_exc())
            if self.fail_fast:
                raise

    def run(self, progressbar: bool = False) -> list[MoleculeState]:
        """Execute density optimization over the configured dataset.

        Args:
            progressbar: Whether to display a progress bar during optimization.

        Returns:
            list[MoleculeState]: Finished states returned by the selected
            execution path.
        """
        # Side effects:
        # - Resets per-run mutable state such as ``self.step_count``.
        # - Advances optional profiler state via ``self._profiler.step()``.
        # - Triggers sample finalization/evaluation and associated artifact writes.

        self.step_count = 0

        # The legacy optimizer consumes dataloader batches, while the modern
        # path works on the full dataset through the lockstep loop.
        if is_legacy_optimizer(self.optimizer):
            results = []
            for batch in self.dataloader:
                samples = batch.to_data_list()
                for sample in samples:
                    sample.to(self.device)
                batch_results = self._run_legacy(samples)
                results.extend(batch_results)

                if self._profiler is not None:
                    self._profiler.step()

        else:
            results = self._run_batched(self.dataset, progressbar=progressbar)

        self._finalize_run()
        return results

    def run_samples(
        self, samples: Collection[OFData | BaseData], progressbar: bool = False
    ) -> list[MoleculeState]:
        """Execute denop for an explicit set of samples.

        Args:
            samples: Samples to optimize.
            progressbar: Whether to display a progress bar.

        Returns:
            list[MoleculeState]: Finished states from the optimization run.
        """
        # Side effects:
        # - Mutates per-run runner state (for example step counters and active-session state).
        # - Triggers finalization/evaluation that writes output artifacts.

        results = super().run_samples(samples, progressbar=progressbar)
        self._finalize_run()
        return results

    def reset(self) -> None:
        """Reset mutable runner state between invocations."""
        # Recreate the optimizer so repeated runs do not share optimizer state.
        self.step_count = 0
        self.optimizer = instantiate(self.config.get("density_optimizer"))

    def _run_legacy(self, samples: Collection[OFData | BaseData]) -> list[MoleculeState]:
        """Run legacy optimizer path that owns its own inner optimization loop.

        Args:
            samples: Batch of samples to process.

        Returns:
            list[MoleculeState]: Finished states when finalization is disabled.

        Raises:
            Exception: Re-raises optimizer failures when ``fail_fast`` is enabled.
        """
        # Side effects:
        # - Updates sample/callback state through the legacy optimization routine.
        # - Finalizes successful samples via the evaluator, which persists output files.
        # - Emits log messages for failures and progress.

        finished_states: list[MoleculeState] = []
        # The legacy path optimizes one sample at a time and optionally forwards
        # the finished sample straight into the evaluator.
        for sample_index, sample in enumerate(samples):
            self._clear_prediction_fields(sample)
            callback = ConvergenceCallback(mol=sample.mol)
            try:
                metric_dict, callback, energies_label, energy_functional = (
                    density_optimization_with_label(
                        sample=sample,
                        mol=sample.mol,
                        optimizer=self.optimizer,
                        func_factory=self.func_factory,
                        callback=callback,
                        max_xc_memory=self.max_xc_memory,
                        initialization=self.initialization,
                        normalize_initial_guess=self.normalize_initial_guess,
                        ks_basis=self.ks_basis,
                        proj_minao_module=self.proj_minao_module,
                        sad_guess_kwargs=self.sad_guess_kwargs,
                        convergence_criterion=self.convergence_criterion,
                        disable_printing=True,
                        position=0,
                    )
                )
                self.evaluator.finalize_sample(
                    sample=sample,
                    callback=callback,
                    energies_label=energies_label,
                    metric_dict=metric_dict,
                    energy_functional=energy_functional,
                    sample_index=sample_index,
                )

            except Exception as exc:
                logger.exception(f"Legacy run failed for molecule {sample.mol_id}: {exc}")
                if self.fail_fast:
                    raise

        return finished_states

    def _run_batched(
        self,
        samples: Collection[OFData | BaseData],
        progressbar: bool = False,
    ) -> list[MoleculeState] | None:
        """Run lockstep active-set optimization with batched model evaluation.

        Args:
            samples: Dataset collection yielding optimization samples. Must
                support ``len`` for progress tracking.
            progressbar: Whether to display a progress bar during optimization.

        Returns:
            list[MoleculeState]: Any remaining finished states after the loop.

        Raises:
            Exception: Re-raises finalize failures when ``fail_fast`` is enabled.
        """
        # Side effects:
        # - Mutates active/finished state queues and per-sample optimization state in place.
        # - Advances progress-bar state.
        # - Periodically finalizes completed states, which writes output artifacts.

        logger.info(f"Running lockstep optimization on a dataset of size {len(samples)}")
        # Fill the active pool once, then keep cycling between batched model
        # evaluation, per-sample optimizer steps, and finalize/refill checks.
        samples_iter = iter(samples)
        self._next_sample_index = 0
        active_states = []
        active_samples: list[OFData] = []
        active_contexts: list[PreparedFunctionalContext] = []

        for _ in range(self.batch_size):
            sample = next(samples_iter, None)
            if sample is None:
                break
            else:
                sample.to(self.device)
                sample_index = self._next_sample_index
                self._next_sample_index += 1
                new_state = self._prepare_sample(sample, sample_index=sample_index)
                active_states.append(new_state)
                active_samples.append(new_state.sample)
                active_contexts.append(new_state.context)

        finished_states: list[MoleculeState] = []
        finished_counts = 0
        dataset_exhausted = False

        progress_bar_context = (
            tqdm(total=len(samples), desc="Denop", unit="mol") if progressbar else nullcontext()
        )

        # run until all molecules have finished optimization, checking for refill at intervals
        with progress_bar_context as progress_bar:
            # run over the dataset until it's exhausted and all active molecules are finalized
            while finished_counts < len(samples):
                exhausted = self._run_batched_step(
                    dataset_iter=samples_iter,
                    active_states=active_states,
                    active_samples=active_samples,
                    active_contexts=active_contexts,
                    finished_states=finished_states,
                )

                if self._profiler is not None:
                    self._profiler.step()

                dataset_exhausted = dataset_exhausted or exhausted

                # evaluate a batch of finished molecules to not accumulate too much memory overhead in this function
                should_finalize_batch = len(finished_states) >= self.finalize_at_least_n

                # check if the last batch is being finalized to make sure we finalize all molecules even if the last batch is smaller than the specified batch size
                should_finalize_tail = (
                    dataset_exhausted
                    and len(finished_states) > 0
                    and (len(active_states) == 0 or all(state.finished for state in active_states))
                )

                if should_finalize_batch or should_finalize_tail:
                    prev_finished_counts = finished_counts
                    finished_states, finished_counts = self._run_finalize(
                        finished_states, finished_counts
                    )

                    if progress_bar:
                        progress_bar.update(finished_counts - prev_finished_counts)
        return finished_states

    def _run_finalize(
        self, finished_states: list[MoleculeState], finished_counts: int
    ) -> tuple[list[MoleculeState], int]:
        """Finalize a queue of completed lockstep molecule states.

        Args:
            finished_states: Queue of completed states awaiting evaluation.
            finished_counts: Number of states finalized so far.

        Returns:
            tuple[list[MoleculeState], int]: The cleared finished-state queue
            and the updated finalized count.
        """
        # Side effects:
        # - Finalizes samples through the evaluator, persisting sample artifacts/metrics.
        # - Emits failure logs and may re-raise exceptions when ``fail_fast`` is enabled.

        # Convert completed optimization traces into metrics and persisted
        # outputs before clearing the queue for the next finalize batch.
        for state in finished_states:
            try:
                energies_label = self.func_factory.get_energies_label(
                    state.sample, basis_info=state.sample.basis_info
                )
                metric_dict = compute_density_optimization_metrics(
                    state.callback, energies_label, self.convergence_criterion
                )
                self.evaluator.finalize_sample(
                    sample=state.sample,
                    callback=state.callback,
                    energies_label=energies_label,
                    metric_dict=metric_dict,
                    energy_functional=state.context.energy_functional,
                    sample_index=state.sample_index,
                )
            except Exception as exc:
                logger.exception(
                    f"Lockstep finalize failed for molecule {state.sample.mol_id}: {exc}"
                )
                if self.fail_fast:
                    raise
        finished_counts += len(finished_states)

        return (
            [],
            finished_counts,
        )  # clear the finished states after finalization to free memory for the next batch

    def cleanup(self) -> None:
        """Remove known output artifacts from the current denop save directory."""
        # Restrict cleanup to the artifacts this runner knows how to recreate.
        to_cleanup = [
            "sample_trajectories",
            "hparams_resolved.yaml",
            "reduced_denop_data.csv",
            "denop_config.yaml",
            "denop_log",
            "density_optimization_metrics.yaml",
            "density_optimization_summary.pdf",
        ]
        for item in to_cleanup:
            path_to_object = self.save_dir / item
            if path_to_object.is_file():
                path_to_object.unlink()
            elif path_to_object.is_dir():
                shutil.rmtree(path_to_object)


def run_denop_single_device(
    cfg: DictConfig, cleanup: bool = False, progress_bar: bool = False
) -> None:
    """Run denop on a single device and optionally clean outputs afterwards.

    Args:
        cfg (DictConfig): Runtime configuration passed to :class:`DenopRunner`.
        cleanup (bool): Whether to remove run artifacts after completion.
        progress_bar (bool): Whether to display a progress bar.
    """
    # Side effects:
    # - Seeds global RNG state.
    # - Executes denop and writes run outputs under the configured directory.
    # - Optionally deletes known output artifacts when ``cleanup`` is ``True``.

    # Seed first so all downstream initialization and optimization steps are
    # reproducible under the requested runtime configuration.
    seed_all_rngs(cfg.get("seed"))
    runner = DenopRunner(cfg)
    runner.run(progressbar=progress_bar)
    if cleanup:
        runner.cleanup()


@hydra.main(
    version_base="1.3",
    config_path="../../../configs/ofdft",
    config_name="density_optimization.yaml",
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that prints config, applies extras, and runs denop.

    Args:
        cfg (DictConfig): Hydra runtime configuration.
    """
    # Side effects:
    # - Prints configuration to stdout/rich output.
    # - Applies runtime extras that may mutate logging/runtime behavior.
    # - Executes a full denop run with filesystem side effects.

    # Print and augment the config before dispatching so CLI runs remain easy
    # to inspect and debug.
    print_config_tree(cfg)
    extras(cfg)
    run_denop_single_device(cfg)


if __name__ == "__main__":
    main()
