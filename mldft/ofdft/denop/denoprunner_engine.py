from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from lightning import LightningModule
from loguru import logger
from omegaconf import DictConfig
from rootutils import rootutils
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.denop.functional_bridge import (
    BatchedFunctionalBridge,
    PreparedFunctionalContext,
)
from mldft.ofdft.denop.optimizer_session import (
    OptimizerSessionBase,
    build_optimizer_session,
    is_legacy_optimizer,
)
from mldft.ofdft.density_optimization import (
    density_optimization_with_label,
)
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.initial_guess import initial_guess
from mldft.ofdft.optimizer import Optimizer
from mldft.utils.sad_guesser import SADNormalizationMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@dataclass
class MoleculeState:
    """Track mutable per-molecule state during batched optimization.

    Attributes:
        sample: Molecule sample currently being optimized.
        callback: Callback collecting optimization diagnostics.
        context: Prepared functional context for batched evaluation.
        optimizer: Optimizer session bound to the sample.
        finished: Whether the optimizer has declared the sample done.
        finalize_queued: Whether the finished sample has been queued for
            downstream finalization.
        converged: Whether the optimizer reported convergence.
        last_energy: Most recent total energy returned by the optimizer.
        last_gradient_norm: Most recent gradient norm reported by the optimizer.
        steps: Number of optimizer steps taken for the sample.
    """

    sample: OFData
    callback: ConvergenceCallback
    context: PreparedFunctionalContext
    optimizer: OptimizerSessionBase
    finished: bool = False
    finalize_queued: bool = False
    converged: bool = False
    last_energy: Any | None = None
    last_gradient_norm: float | None = None
    steps: int = 0
    sample_index: int | None = None


class DenopRunnerEngine:
    """Shared runner functionality used by both public denop runners."""

    def __init__(
        self,
        cfg: DictConfig,
        model: LightningModule,
        optimizer: Optimizer,
        dataset_statistics: DatasetStatistics,
        basis_info: BasisInfo,
        device: str,
        batch_size: int,
        fail_fast: bool = True,
    ):
        """Initialize the lightweight runner from explicit dependencies.

        Args:
            cfg: Denop configuration containing runtime options.
            model: Model used both for projections and XC evaluation.
            optimizer: Optimizer instance for density optimization.
            dataset_statistics: Dataset statistics used by SAD initialization.
            basis_info: Basis metadata shared across samples.
            device: Device identifier used for execution.
            batch_size: Maximum number of active molecules in lockstep mode.

        Kwargs:
            fail_fast: Whether runtime errors should be re-raised immediately.
        """
        # Side effects:
        # - Builds functional factory/bridge objects and shared runner state.
        # - Derives and stores initialization kwargs (including SAD statistics when requested).

        self.denop_config = cfg
        self.optimizer = optimizer
        self.device = device
        self.model_dtype = getattr(model, "dtype", OFDFT_DTYPE)
        self.step_count = 0  # counts steps in lockstep loop, used for checking refill
        self.initialization = cfg.get("initialization")
        self.normalize_initial_guess = cfg.get("normalize_initial_guess")
        self.convergence_criterion = cfg.get("convergence_criterion")
        self.save_coeff_interval = int(cfg.get("save_coeff_interval", 100))
        self.ks_basis = cfg.get("ks_basis", "6-31G(2df,p)")
        self.basis_info = basis_info
        self.max_xc_memory = cfg.get("max_xc_memory")
        self.proj_minao_module = model
        self.fail_fast = fail_fast
        self.batch_size = batch_size
        self.xc_functional = self.denop_config.get("xc_functional")
        self.negative_integrated_density_penalty_weight = self.denop_config.get(
            "negative_integrated_density_penalty_weight", 0.0
        )

        # Only SAD-style initializations need dataset statistics; all other
        # initialization modes can pass an empty kwarg dictionary.
        if self.initialization in ["sad", "sad_transformed"]:
            self.sad_guess_kwargs = dict(
                dataset_statistics=dataset_statistics,
                normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
                basis_info=basis_info,
                weigher_key="ground_state_only",
                spherical_average=True,
            )
        else:
            self.sad_guess_kwargs = {}

        self.func_factory = FunctionalFactory.from_module(
            model,
            self.xc_functional,
            self.negative_integrated_density_penalty_weight,
        )
        self.bridge = BatchedFunctionalBridge(
            self.func_factory,
            max_xc_memory=self.max_xc_memory,
        )

    def _clear_prediction_fields(self, sample: OFData | BaseData) -> None:
        """Remove transient model prediction fields from a sample, if present."""
        for key in ("pred_energy", "pred_gradient", "pred_diff"):
            if hasattr(sample, key):
                sample.delete_item(key)

    def _run_legacy(self, samples: Collection[OFData | BaseData]) -> list[MoleculeState]:
        """Run the non-lockstep optimizer path for a set of samples.

        The legacy optimizer owns the entire optimization loop for one sample,
        so this method simply iterates over the inputs, runs the helper
        function, and converts the result into ``MoleculeState`` objects.

        Args:
            samples: Samples to optimize sequentially.

        Returns:
            list[MoleculeState]: Finished sample states for successful runs.

        Raises:
            Exception: Re-raises sample failures when ``fail_fast`` is enabled.
        """
        # Side effects:
        # - Mutates sample state through the legacy optimization routine.
        # - Builds and updates callback trajectories for each processed sample.
        # - Emits logs and may re-raise failures depending on ``fail_fast``.

        finished_states: list[MoleculeState] = []
        # Run each sample independently because the legacy optimizer does not
        # support the lockstep batched execution used by the modern path.

        for sample in samples:
            self._clear_prediction_fields(sample)
            callback = ConvergenceCallback(mol=sample.mol)
            try:
                metric_dict, callback, energies_label, energy_functional = (
                    density_optimization_with_label(
                        sample=sample,
                        mol=sample.mol,
                        optimizer=self.optimizer,
                        func_factory=self.func_factory,
                        callback=ConvergenceCallback(),
                        initialization=self.initialization,
                        max_xc_memory=self.max_xc_memory,
                        normalize_initial_guess=self.normalize_initial_guess,
                        ks_basis=self.ks_basis,
                        proj_minao_module=self.proj_minao_module,
                        sad_guess_kwargs=self.sad_guess_kwargs,
                        convergence_criterion=self.convergence_criterion,
                        disable_printing=True,
                    )
                )

                finished_states.append(
                    MoleculeState(
                        sample=sample,
                        callback=callback,
                        context=None,  # not used in legacy path
                        optimizer=None,  # not used in legacy path
                        finished=True,
                        converged=metric_dict.get("converged", False),
                        last_energy=metric_dict.get("final_energy_total"),
                        last_gradient_norm=metric_dict.get("final_gradient_norm"),
                    )
                )

            except Exception as exc:
                logger.exception(f"Legacy run failed for molecule {sample.mol_id}: {exc}")
                if self.fail_fast:
                    raise

        return finished_states

    def _prepare_initial_guess(self, sample: OFData | BaseData) -> OFData | BaseData:
        """Initialize the coefficient vector for one sample.

        Args:
            sample: Molecule sample to initialize in place.

        Returns:
            OFData | BaseData: The initialized sample.
        """
        # Side effects:
        # - Assigns ``sample.coeffs`` in place based on the configured initialization strategy.

        # Accept either a precomputed tensor or a named initialization scheme.
        if isinstance(self.initialization, torch.Tensor):
            initial_coeffs = self.initialization.to(device=sample.coeffs.device, dtype=OFDFT_DTYPE)
        else:
            initial_coeffs = initial_guess(
                sample,
                sample.mol,
                self.initialization,
                normalize_initial_guess=self.normalize_initial_guess,
                ks_basis=self.ks_basis,
                proj_minao_module=self.proj_minao_module,
                sad_guess_kwargs=self.sad_guess_kwargs,
            )

            # README: these dtype casts are still sprinkled throughout the codebase, which
            # should ideally be avoided, but currently is hard to do w/o larger refactoring of the
            # data part? Type correctness should be an early guarantee, not something that needs to
            # be checked and enforced at multiple points during execution.
            if initial_coeffs.dtype != OFDFT_DTYPE:
                initial_coeffs = initial_coeffs.to(dtype=OFDFT_DTYPE)
        sample.coeffs = initial_coeffs
        return sample

    def _prepare_sample(
        self, sample: OFData | BaseData, sample_index: int | None = None
    ) -> MoleculeState:
        """Prepare one sample for lockstep optimization.

        Args:
            sample: Molecule sample loaded from the dataset.

        Returns:
            MoleculeState: Initialized state with context, callback, and optimizer copy.
        """
        # Side effects:
        # - Mutates ``sample`` by ensuring initialized coefficients.
        # - Allocates and binds per-sample context/callback/session objects.

        # The setup sequence is: normalize sample tensor dtypes, initialize
        # coefficients, prepare the batched functional context, then bind a
        # per-sample optimizer session.
        sample = to_torch(sample, device=sample.coeffs.device, float_dtype=OFDFT_DTYPE)
        prepared_sample = self._prepare_initial_guess(sample)

        self._clear_prediction_fields(prepared_sample)
        context = self.bridge.prepare(prepared_sample)
        callback = ConvergenceCallback(mol=prepared_sample.mol)

        # Optimizer/session state stays in OFDFT_DTYPE for the full trajectory.
        # Per-step model-boundary casts are handled inside the functional bridge.
        session = build_optimizer_session(self.optimizer, prepared_sample)

        return MoleculeState(
            sample=prepared_sample,
            callback=callback,
            context=context,
            optimizer=session,
            sample_index=sample_index,
        )

    def _exchange_samples(
        self,
        dataset_iter: Iterator,
        active_states: list[MoleculeState],
        active_samples: list[OFData],
        active_contexts: list[PreparedFunctionalContext],
        finished_states: list[MoleculeState],
    ) -> bool:
        """Replace a finished active slot with the next dataset sample if available.

        Args:
            dataset_iter: Iterator over remaining dataset samples.
            active_states: Mutable list of active molecule states.
            active_samples: Mutable list of active samples in lockstep order.
            active_contexts: Mutable list of active contexts in lockstep order.
            finished_states: Queue collecting finished states.

        Returns:
            bool: Whether the dataset was exhausted while trying to refill
            finished slots.
        """
        # Side effects:
        # - Mutates ``active_states``, ``active_samples``, and ``active_contexts`` in place.
        # - Appends completed states to ``finished_states`` and clears per-step scratch buffers.
        # - Pulls additional samples from ``dataset_iter`` to refill finished slots.

        exhausted = False
        # Walk from the end so exhausted slots can be removed in-place without
        # shifting unvisited indices.
        for i in range(len(active_states) - 1, -1, -1):
            state = active_states[i]
            if (
                state.finished and not state.finalize_queued
            ):  # only exchange if finished and not already queued for finalization
                # Release per-active-step scratch tensors before queueing the
                # finalized state to keep finalize-queue memory bounded.
                state.context.non_model_grad = None
                finished_states.append(state)
                state.finalize_queued = True
                sample = next(dataset_iter, None)

                if sample is not None:
                    sample.to(self.device)
                    if not hasattr(self, "_next_sample_index"):
                        self._next_sample_index = 0
                    sample_index = self._next_sample_index
                    self._next_sample_index += 1
                    new_state = self._prepare_sample(sample, sample_index=sample_index)
                    active_states[i] = new_state
                    active_samples[i] = new_state.sample
                    active_contexts[i] = new_state.context
                else:
                    exhausted = True
                    # Remove exhausted finished states from the active set so
                    # finalized samples do not get re-collated/evaluated.
                    active_states.pop(i)
                    active_samples.pop(i)
                    active_contexts.pop(i)
        return exhausted

    def _run_batched_step(
        self,
        dataset_iter: Iterator,
        active_states: list[MoleculeState],
        active_samples: list[OFData],
        active_contexts: list[PreparedFunctionalContext],
        finished_states: list[MoleculeState],
    ) -> bool:
        """Run a single step of the density optimization process on a batch of active samples.

        Args:
            dataset_iter (Iterator): Iterator over the dataset
            active_states (list[MoleculeState]): Current active molecule states.
            active_samples (list[OFData]): Current active samples in lockstep order.
            active_contexts (list[PreparedFunctionalContext]): Current active contexts in lockstep order.
            finished_states (list[MoleculeState]): Queue collecting finished states.

        Returns:
            bool: Whether the dataset was exhausted during refill.
        """
        # Side effects:
        # - Updates active_states, active_samples, and active_contexts in-place by stepping optimizers and exchanging finished samples for new ones from the dataset.
        # - adds finished states to finished_states when optimizers report completion.

        exhausted = False
        # avoid checking for refill on first step
        if self.step_count > 0:
            exhausted = self._exchange_samples(
                dataset_iter,
                active_states,
                active_samples,
                active_contexts,
                finished_states,
            )

        if len(active_states) == 0:
            self.step_count += 1
            return exhausted

        # run forward pass batched
        eval_results = self.bridge.evaluate_active(active_samples, active_contexts)

        # then run optimizer steps for each molecule sequentially to update individual states
        # we might be able to use cuda graphs here

        # README: this is the main part that blocks further speed gains, but without reimplementing optimizers,
        # it's pretty much impossible to batch this part because each sample has individual state, learning rate, momentum and
        # takes a different amount of time to optimize.
        for state, eval_result in zip(active_states, eval_results, strict=True):
            if not state.finished:  # do not modify states that have been finished already

                step_result = state.optimizer.step(
                    sample=state.sample,
                    energy=eval_result.energies,
                    gradient_vector=eval_result.gradient,
                    callback=state.callback,
                )

                state.last_energy = step_result.energy
                state.last_gradient_norm = step_result.gradient_norm
                state.finished = step_result.finished
                state.converged = step_result.converged

        self.step_count += 1
        return exhausted

    def _run_batched(
        self, samples: Collection[OFData | BaseData], progressbar: bool = False
    ) -> list[MoleculeState] | None:
        """Run the basic lockstep batched optimization loop.

        This shared implementation keeps only the core active-set behavior:
        fill the active pool, evaluate all active molecules together, step each
        optimizer session, and periodically refill completed slots.

        Args:
            samples: Collection of samples to process. Must support ``len``.

        Returns:
            list[MoleculeState]: Finished states collected during refills.
        """
        # Side effects:
        # - Initializes and mutates active/finished state queues during lockstep execution.
        # - Advances ``self.step_count`` through repeated ``_run_batched_step`` calls.
        # - Updates optional progress bar output.

        # Seed the active set with as many samples as the configured batch size
        # allows so that the batched model path starts full.
        dataset_iter = iter(samples)
        active_states = []
        active_samples: list[OFData] = []
        active_contexts: list[PreparedFunctionalContext] = []
        current_batch_size = min(self.batch_size, len(samples))

        for _ in range(current_batch_size):
            sample = next(dataset_iter, None)
            if sample is None:
                break
            else:
                sample.to(self.device)
                new_state = self._prepare_sample(sample)
                active_states.append(new_state)
                active_samples.append(new_state.sample)
                active_contexts.append(new_state.context)

        finished_states: list[MoleculeState] = []
        finished_counts = 0

        progress_bar_context = (
            tqdm(total=len(samples), desc="Denop", unit="mol") if progressbar else nullcontext()
        )

        # run until all molecules have finished optimization, checking for refill at intervals
        with progress_bar_context as progress_bar:
            # run over the dataset until it's exhausted and all active molecules are finalized
            while finished_counts < len(samples):
                _ = self._run_batched_step(
                    dataset_iter,
                    active_states,
                    active_samples,
                    active_contexts,
                    finished_states,
                )
                finished_counts = len(finished_states)

                if progress_bar:
                    progress_bar.update(finished_counts)

        return finished_states

    def run_samples(
        self, samples: Collection[OFData | BaseData], progressbar: bool = False
    ) -> list[MoleculeState]:
        """Run denop for an explicit collection of already-loaded samples.

        Args:
            samples: Samples to optimize.

        Returns:
            list[MoleculeState]: Final sample states returned by the chosen
            execution path.
        """
        # Side effects:
        # - Resets ``self.step_count`` for a fresh run.
        # - Moves samples to the configured device in legacy mode.
        # - Mutates per-sample optimization state through legacy or lockstep execution.

        self.step_count = 0
        if is_legacy_optimizer(self.optimizer):
            results = self._run_legacy([sample.to(self.device) for sample in samples])
        else:
            results = self._run_batched(samples, progressbar=progressbar)
        return results

    def set_model(self, model: LightningModule) -> None:
        """Rebuild the functional components from a new model instance.

        Args:
            model: Model providing the XC functional implementation.
        """
        # Side effects:
        # - Replaces ``self.func_factory`` and ``self.bridge`` with instances bound to ``model``.
        # - Discards prior functional state for subsequent runs.

        # Recreate both the functional factory and the bridge so subsequent
        # runs use the new model consistently.
        self.func_factory = FunctionalFactory.from_module(
            model,
            self.xc_functional,
            self.negative_integrated_density_penalty_weight,
        )
        self.bridge = BatchedFunctionalBridge(
            self.func_factory,
            max_xc_memory=self.max_xc_memory,
        )
