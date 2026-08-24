"""Per-molecule optimizer sessions for lockstep denop execution.

Design principle:
- Keep *state ownership* local to each molecule, not in global runner structures.
- Keep the runner agnostic to optimizer internals by exposing a minimal step contract.
- Preserve mathematical behavior of existing optimizers while enabling coordinated batched
  functional evaluation outside the optimizer itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch_geometric.nn import global_add_pool

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft.energies import Energies
from mldft.ofdft.optimizer import (
    SLSQP,
    GradientDescent,
    Optimizer,
    TorchOptimizer,
    TrustRegionConstrained,
    VectorAdam,
    build_reg_mat,
    regularize_gradient,
)
from mldft.ofdft.scheduler import (
    AdaptiveSigmoidScheduler,
    AdaptiveSigmoidSchedulerWithDecay,
)


@dataclass
class StepResult:
    """Result of one optimizer session step.

    Principle:
    - The runner should only depend on convergence/lifecycle facts, not optimizer internals.
    """

    converged: bool
    finished: bool
    gradient_norm: float
    learning_rate: float
    energy: Energies


class CallbackLike(Protocol):
    """Callback interface consumed by optimization sessions.

    The callback receives a dictionary containing relevant variables from the active step method,
    """

    def __call__(self, minimization_locals: dict) -> None: ...


class OptimizerSessionBase:
    """Common interface for stateful, per-molecule optimization sessions."""

    def __init__(self, optimizer: Optimizer, sample: OFData | None = None):
        """Store shared optimizer state for one molecule.

        Args:
            optimizer (Optimizer): Optimizer configuration/instance driving this session.
            sample (OFData | None): Optional sample used to initialize cached per-molecule state.
        """
        # Side effects:
        # - Initializes mutable per-session counters and cached regularization state.

        self.optimizer = optimizer
        self.cycle = 0
        self.apply_reg_mat_to_step = bool(getattr(optimizer, "apply_reg_mat_to_step", True))
        self.reg_mat = self._build_reg_mat(sample) if sample is not None else None

    @property
    def max_cycle(self) -> int:
        """Maximum number of optimization cycles allowed for this session."""
        return int(getattr(self.optimizer, "max_cycle"))

    @property
    def convergence_tolerance(self) -> float:
        """Gradient-norm threshold used to declare convergence."""
        return float(getattr(self.optimizer, "convergence_tolerance"))

    def _build_reg_mat(self, sample: OFData) -> torch.Tensor | None:
        """Initialize the cached regularization operator for one molecule."""
        return build_reg_mat(
            sample,
            device=sample.coeffs.device,
            reg_epsilon=getattr(self.optimizer, "reg_epsilon", None),
            reg_gamma=getattr(self.optimizer, "reg_gamma", None),
            hard_projection=bool(getattr(self.optimizer, "hard_projection", False)),
            eigenvalue_threshold=getattr(self.optimizer, "eigenvalue_threshold", None),
            quantile_threshold=getattr(self.optimizer, "quantile_threshold", None),
            orthogonal_projection=bool(getattr(self.optimizer, "orthogonal_projection", True)),
            ks_basis=str(getattr(self.optimizer, "ks_basis", "6-31G(2df,p)")),
            overlap_tensor_block_target_bytes=int(
                getattr(self.optimizer, "overlap_tensor_block_target_bytes", 1_000_000_000)
            ),
        )

    def _regularize_projected_gradient(
        self, projected_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, float | None, float | None]:
        """Apply cached gradient regularization if configured for this session."""
        if self.reg_mat is None:
            return projected_gradient, None, None
        return regularize_gradient(
            projected_gradient,
            self.reg_mat,
            apply_reg_mat_to_step=self.apply_reg_mat_to_step,
        )

    def step(
        self,
        sample: OFData,
        energy: Energies,
        gradient_vector: torch.Tensor,
        callback: CallbackLike | None = None,
    ) -> StepResult:
        """Execute one optimization step for a single molecule.

        Args:
            sample (OFData): Mutable molecule sample containing coefficients to update.
            energy (Energies): Current energy decomposition from the external functional pass.
            gradient_vector (torch.Tensor): Current (unprojected) coefficient gradient.
            callback (CallbackLike | None): Optional callback invoked with local step state.

        Returns:
            StepResult: Step outcome used by lockstep orchestration.
        """
        # Side effects:
        # - Intended to mutate optimizer/session and sample state in concrete subclasses.
        # - May invoke callbacks with step diagnostics in subclass implementations.

        raise NotImplementedError


class GradientDescentSession(OptimizerSessionBase):
    """Sessionized version of :class:`GradientDescent`.

    Principle:
    - Keep update mathematics identical to existing implementation.
    - Externalize only loop control so lockstep orchestration remains possible.
    """

    def __init__(self, optimizer: GradientDescent, sample: OFData):
        """Initialize gradient-descent session state.

        Args:
            optimizer (GradientDescent): Gradient descent optimizer configuration.
            sample (OFData): Sample used to initialize cached per-molecule state.
        """
        # Side effects:
        # - Initializes mutable learning-rate/session state for this molecule.

        super().__init__(optimizer, sample)
        self.learning_rate = float(optimizer.learning_rate)

    def step(
        self,
        sample: OFData,
        energy: Energies,
        gradient_vector: torch.Tensor,
        callback: CallbackLike | None = None,
    ) -> StepResult:
        """Apply one projected gradient-descent update.

        Args:
            sample (OFData): Molecule sample to update in place.
            energy (Energies): Energy values for the current iterate.
            gradient_vector (torch.Tensor): Raw gradient in coefficient space.
            callback (CallbackLike | None): Optional callback receiving step locals.

        Returns:
            StepResult: Convergence/lifecycle status and diagnostics for this cycle.
        """
        # Side effects:
        # - Mutates ``sample.coeffs`` in place when the step has not converged.
        # - Increments session cycle state and may invoke ``callback``.

        projected_gradient = project_gradient(gradient_vector, sample)
        projected_gradient, grad_norm_ratio, grad_cos_sim = self._regularize_projected_gradient(
            projected_gradient
        )
        gradient_norm = torch.norm(projected_gradient).item()

        if callable(callback):
            callback(
                {
                    "energy": energy,
                    "gradient_norm": gradient_norm,
                    "learning_rate": self.learning_rate,
                    "grad_norm_ratio": grad_norm_ratio,
                    "grad_cos_sim": grad_cos_sim,
                    "coeffs": sample.coeffs,
                    "sample": sample,
                }
            )

        converged = gradient_norm < self.convergence_tolerance
        finished = converged or (self.cycle + 1 >= self.max_cycle)
        if not converged:
            sample.coeffs -= self.learning_rate * projected_gradient

        self.cycle += 1
        return StepResult(
            converged=converged,
            finished=finished,
            gradient_norm=gradient_norm,
            learning_rate=self.learning_rate,
            energy=energy,
        )


class TorchOptimizerSession(OptimizerSessionBase):
    """Sessionized version of :class:`TorchOptimizer`.

    Principle:
    - Preserve the original separation between parameter tensor and sample tensor to avoid hidden
      aliasing of optimizer internals with model state.
    - Preserve projected-update behavior for manifold constraints.
    """

    def __init__(self, optimizer: TorchOptimizer, sample: OFData):
        """Initialize torch-optimizer-backed session state.

        Args:
            optimizer (TorchOptimizer): Wrapper specifying torch optimizer and scheduler config.
            sample (OFData): Sample used to initialize per-session parameter tensor shape/device.
        """
        # Side effects:
        # - Allocates per-session parameter tensor, optimizer, and optional scheduler objects.

        super().__init__(optimizer, sample)
        self.parameters = sample.coeffs.clone()
        self.torch_optimizer = optimizer.torch_optimizer(
            [self.parameters], **optimizer.optimizer_kwargs
        )
        if optimizer.scheduler_cfg is not None:
            self.scheduler = optimizer.scheduler_cfg(optimizer=self.torch_optimizer)
        else:
            self.scheduler = None

    def step(
        self,
        sample: OFData,
        energy: Energies,
        gradient_vector: torch.Tensor,
        callback: CallbackLike | None = None,
    ) -> StepResult:
        """Apply one projected update using a ``torch.optim`` optimizer.

        Args:
            sample (OFData): Molecule sample to update in place.
            energy (Energies): Energy values for the current iterate.
            gradient_vector (torch.Tensor): Raw gradient in coefficient space.
            callback (CallbackLike | None): Optional callback receiving step locals.

        Returns:
            StepResult: Convergence/lifecycle status and diagnostics for this cycle.
        """
        # Side effects:
        # - Updates internal torch optimizer/scheduler state.
        # - Mutates ``sample.coeffs`` and mirrored parameter tensor in place.
        # - Increments session cycle state and may invoke ``callback``.

        projected_gradient = project_gradient(gradient_vector, sample)
        projected_gradient, grad_norm_ratio, grad_cos_sim = self._regularize_projected_gradient(
            projected_gradient
        )
        gradient_norm = torch.norm(projected_gradient).item()
        learning_rate = float(self.torch_optimizer.param_groups[0]["lr"])

        if callable(callback):
            callback(
                {
                    "energy": energy,
                    "gradient_norm": gradient_norm,
                    "learning_rate": learning_rate,
                    "grad_norm_ratio": grad_norm_ratio,
                    "grad_cos_sim": grad_cos_sim,
                    "coeffs": sample.coeffs,
                    "sample": sample,
                }
            )

        converged = gradient_norm < self.convergence_tolerance
        finished = converged or (self.cycle + 1 >= self.max_cycle)

        if not converged:
            self.torch_optimizer.zero_grad()
            self.parameters.grad = projected_gradient
            self.torch_optimizer.step()

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    (AdaptiveSigmoidScheduler, AdaptiveSigmoidSchedulerWithDecay),
                ):
                    self.scheduler.step(gradient_norm)
                else:
                    self.scheduler.step()

            update = self.parameters - sample.coeffs
            sample.coeffs += project_gradient(update, sample)
            with torch.no_grad():
                self.parameters.copy_(sample.coeffs)
        self.cycle += 1
        return StepResult(
            converged=converged,
            finished=finished,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            energy=energy,
        )


class VectorAdamSession(OptimizerSessionBase):
    """Sessionized version of :class:`VectorAdam`.

    Principle:
    - Cache shell-index structure once per molecule to avoid repeated structural work in hot path.
    - Keep per-molecule moments fully isolated.
    """

    def __init__(self, optimizer: VectorAdam, sample: OFData):
        """Initialize VectorAdam session state and shell-index cache.

        Args:
            optimizer (VectorAdam): VectorAdam optimizer configuration.
            sample (OFData): Sample used to initialize moments and shell structure.
        """
        # Side effects:
        # - Allocates and initializes per-molecule Adam moments and shell-index cache.

        super().__init__(optimizer, sample)
        self.learning_rate = float(optimizer.learning_rate)
        self.beta1, self.beta2 = optimizer.betas
        self.epsilon = float(optimizer.epsilon)
        self.m = torch.zeros_like(sample.coeffs)
        self.v = torch.zeros_like(sample.coeffs)

        shell_beginning_mask = sample.basis_info.shell_beginning_mask
        shell_beginning_mask = torch.as_tensor(shell_beginning_mask, device=sample.coeffs.device)
        self.shell_indices = (
            torch.cumsum(shell_beginning_mask[sample.basis_function_ind], dim=0) - 1
        )

    def step(
        self,
        sample: OFData,
        energy: Energies,
        gradient_vector: torch.Tensor,
        callback: CallbackLike | None = None,
    ) -> StepResult:
        """Apply one VectorAdam update with projected constraints.

        Args:
            sample (OFData): Molecule sample to update in place.
            energy (Energies): Energy values for the current iterate.
            gradient_vector (torch.Tensor): Raw gradient in coefficient space.
            callback (CallbackLike | None): Optional callback receiving step locals.

        Returns:
            StepResult: Convergence/lifecycle status and diagnostics for this cycle.
        """
        # Side effects:
        # - Updates internal moment buffers and cycle counter.
        # - Mutates ``sample.coeffs`` in place when the step has not converged.
        # - May invoke ``callback`` with step diagnostics.

        projected_gradient = project_gradient(gradient_vector, sample)
        projected_gradient, grad_norm_ratio, grad_cos_sim = self._regularize_projected_gradient(
            projected_gradient
        )
        gradient_norm = torch.norm(projected_gradient).item()

        self.m = self.beta1 * self.m + (1 - self.beta1) * projected_gradient
        v_per_irrep = global_add_pool(projected_gradient**2, self.shell_indices)
        self.v = self.beta2 * self.v + (1 - self.beta2) * v_per_irrep[self.shell_indices]
        m_hat = self.m / (1 - self.beta1 ** (self.cycle + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.cycle + 1))
        update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        update = project_gradient(update, sample)

        if callable(callback):
            # do not use locals() here, because that pulls in the entire local context
            # effectively turning this callback into a closure with access to all local variables, which is very bad for readability and debugging
            callback(
                {
                    "energy": energy,
                    "gradient_norm": gradient_norm,
                    "learning_rate": self.learning_rate,
                    "grad_norm_ratio": grad_norm_ratio,
                    "grad_cos_sim": grad_cos_sim,
                    "coeffs": sample.coeffs,
                    "sample": sample,
                }
            )

        converged = gradient_norm < self.convergence_tolerance
        finished = converged or (self.cycle + 1 >= self.max_cycle)
        if not converged:
            sample.coeffs -= update

        self.cycle += 1
        return StepResult(
            converged=converged,
            finished=finished,
            gradient_norm=gradient_norm,
            learning_rate=self.learning_rate,
            energy=energy,
        )


class LegacyOptimizeSession(OptimizerSessionBase):
    """Compatibility session for optimizers that are not lockstep-step native.

    Principle:
    - Keep broad optimizer compatibility in this phase by delegating to existing optimize loop.
    - This path is intentionally non-lockstep and should be treated as compatibility fallback.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        energy_functional,
        callback: CallbackLike | None = None,
    ):
        """Create compatibility session delegating to ``optimizer.optimize``.

        Args:
            optimizer (Optimizer): Legacy optimizer that owns its optimization loop.
            energy_functional: Callable energy functional consumed by ``optimizer.optimize``.
            callback (CallbackLike | None): Optional callback forwarded to optimize.
        """
        # Side effects:
        # - Stores callback/functional references and initializes one-shot execution guard state.

        super().__init__(optimizer)
        self.energy_functional = energy_functional
        self.callback = callback
        self._done = False

    def run_to_completion(self, sample: OFData) -> tuple[Energies, bool]:
        """Run legacy optimization to completion exactly once.

        Args:
            sample (OFData): Molecule sample to optimize.

        Returns:
            tuple[Energies, bool]: Final energies and convergence flag from legacy optimize.

        Raises:
            RuntimeError: If called more than once for the same session.
        """
        # Side effects:
        # - Marks the session as consumed via ``self._done``.
        # - Runs the legacy optimizer loop, which mutates ``sample`` and callback history.

        if self._done:
            raise RuntimeError("LegacyOptimizeSession can only be executed once.")
        self._done = True
        return self.optimizer.optimize(
            sample,
            self.energy_functional,
            callback=self.callback,
            disable_pbar=True,
            position=0,
        )

    def step(
        self,
        sample: OFData,
        energy: Energies,
        gradient_vector: torch.Tensor,
        callback: CallbackLike | None = None,
    ) -> StepResult:
        """Not supported for legacy sessions.

        Raises:
            RuntimeError: Always, because this compatibility path is non-lockstep.
        """
        # Side effects:
        # - Always raises and does not return.

        raise RuntimeError("LegacyOptimizeSession does not support lockstep stepping.")


def is_legacy_optimizer(optimizer: Optimizer) -> bool:
    """Return whether optimizer requires legacy non-lockstep execution."""
    return isinstance(optimizer, (SLSQP, TrustRegionConstrained))


def build_optimizer_session(optimizer: Optimizer, sample: OFData) -> OptimizerSessionBase:
    """Build a lockstep-capable per-molecule session for supported optimizers.

    Args:
        optimizer (Optimizer): Optimizer instance/config to sessionize.
        sample (OFData): Sample providing shape/device context for session state.

    Returns:
        OptimizerSessionBase: Concrete session implementation for ``optimizer``.

    Raises:
        TypeError: If optimizer is unsupported by lockstep sessions.
    """
    # Side effects:
    # - Allocates and returns a new per-molecule session object with its own mutable state.

    if isinstance(optimizer, TorchOptimizer):
        return TorchOptimizerSession(optimizer, sample)
    if isinstance(optimizer, GradientDescent):
        return GradientDescentSession(optimizer, sample)
    if isinstance(optimizer, VectorAdam):
        return VectorAdamSession(optimizer, sample)
    raise TypeError(
        "Optimizer class is not supported by lockstep sessions. "
        "Use compatibility fallback path for this optimizer."
    )
