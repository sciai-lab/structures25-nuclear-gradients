import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Type

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize
from torch_geometric.nn import global_add_pool
from tqdm import tqdm

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import (
    get_overlap_matrix,
    get_overlap_tensor_contraction_blocked,
)
from mldft.ofdft.energies import Energies
from mldft.ofdft.scheduler import (
    AdaptiveSigmoidScheduler,
    AdaptiveSigmoidSchedulerWithDecay,
)
from mldft.utils.molecules import build_molecule_ofdata

# NOTE: In general as of oct 2025 the use of projected gradient in the optimizers where both are
# possible (SLSQP, TrustRegionConstrained, TorchOptimizer) does not make a difference


class Optimizer(ABC):
    """Base class for optimization algorithms for density optimization."""

    @abstractmethod
    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable[[OFData], tuple[Energies, torch.Tensor]],
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Perform density optimization.

        Args:
            sample: The OFData containing the initial coefficients.
            energy_functional: Callable which returns the energy and gradient vector.
            callback: Optional callback function.
            disable_pbar: Whether to disable the progress bar.
            position: The position in the progress bar.

        Returns:
            Final energy.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a string representation of the optimizer."""
        name = self.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{name}({settings})"


def build_reg_mat(
    sample: OFData,
    *,
    device: torch.device,
    reg_epsilon: float | None,
    reg_gamma: float | None,
    hard_projection: bool,
    eigenvalue_threshold: float | None,
    quantile_threshold: float | None,
    orthogonal_projection: bool,
    ks_basis: str,
    overlap_tensor_block_target_bytes: int = 1_000_000_000,
) -> torch.Tensor | None:
    """Build a linear operator (matrix) to regularize/project gradients.

    Returns None if no regularization is configured.
    """

    if not (reg_epsilon is not None or hard_projection):
        return None

    if eigenvalue_threshold is not None and quantile_threshold is not None:
        raise ValueError(
            "Cannot use both eigenvalue_threshold and quantile_threshold for hard projection."
        )

    dtype = sample.coeffs.dtype
    mol = build_molecule_ofdata(sample, sample.basis_info.basis_dict)
    mol_orbital_basis = build_molecule_ofdata(sample, ks_basis)
    A = get_overlap_tensor_contraction_blocked(
        mol,
        mol_orbital_basis,
        device=device,
        dtype=dtype,
        target_bytes=overlap_tensor_block_target_bytes,
    )

    T_inv = sample.inv_transformation_matrix.to(device=device, dtype=dtype)
    if orthogonal_projection:
        # Orthogonal projection: transform A to coefficient space first, then
        # eigendecompose. This yields a symmetric projection that cannot amplify norms.
        A = T_inv.T @ A @ T_inv

    if hard_projection:
        eigenvalues, eigenvectors = torch.linalg.eigh(A)

        if eigenvalue_threshold is not None:
            keep_mask = eigenvalues > eigenvalue_threshold
        elif quantile_threshold is not None:
            threshold_value = torch.quantile(eigenvalues, quantile_threshold)
            keep_mask = eigenvalues > threshold_value
        else:
            keep_mask = torch.ones_like(eigenvalues, dtype=torch.bool)

        D = torch.zeros_like(eigenvalues)
        D[keep_mask] = 1.0
        reg_mat = eigenvectors @ torch.diag(D) @ eigenvectors.T

        if not orthogonal_projection:
            if reg_gamma is not None:
                B = torch.as_tensor(get_overlap_matrix(mol), device=device, dtype=dtype)
                B_reg = (
                    torch.linalg.inv(
                        B + reg_gamma * torch.eye(B.shape[0], device=device, dtype=dtype)
                    )
                    @ B
                )
                reg_mat = reg_mat @ B_reg
            reg_mat = T_inv.T @ reg_mat @ T_inv
        return reg_mat

    if reg_epsilon is None:
        raise ValueError("reg_epsilon must be set for soft regularization.")

    # Soft regularization: (A + ε*I)^{-1} A
    eps = float(reg_epsilon)
    reg_mat = torch.linalg.inv(A + eps * torch.eye(A.shape[0], device=device, dtype=dtype)) @ A

    if not orthogonal_projection:
        if reg_gamma is not None:
            B = torch.as_tensor(get_overlap_matrix(mol), device=device, dtype=dtype)
            B_reg = (
                torch.linalg.inv(B + reg_gamma * torch.eye(B.shape[0], device=device, dtype=dtype))
                @ B
            )
            reg_mat = reg_mat @ B_reg
        reg_mat = T_inv.T @ reg_mat @ T_inv

    return reg_mat


def regularize_gradient(
    projected_gradient: torch.Tensor,
    reg_mat: torch.Tensor,
    *,
    apply_reg_mat_to_step: bool,
    compute_diagnostics: bool = True,
) -> tuple[torch.Tensor, float | None, float | None]:
    """Optionally apply reg_mat to the projected gradient and compute diagnostics.

    Args:
        compute_diagnostics: If False, skip the diagnostic norm/cosine computations. They
            force GPU->CPU synchronizations in every optimization cycle, so they are only
            worth computing when a progress bar or callback consumes them.
    """

    grad_before = projected_gradient
    grad_after = reg_mat @ grad_before
    grad_for_step = grad_after if apply_reg_mat_to_step else grad_before

    if not compute_diagnostics:
        return grad_for_step, None, None

    grad_norm_before = torch.norm(grad_before).item()
    grad_norm_after = torch.norm(grad_after).item()
    grad_norm_ratio = grad_norm_after / grad_norm_before if grad_norm_before > 0 else 0.0
    cos_sim = torch.dot(grad_before, grad_after) / (grad_norm_before * grad_norm_after + 1e-12)
    grad_cos_sim = cos_sim.item()

    return grad_for_step, grad_norm_ratio, grad_cos_sim


def get_pbar_str(
    sample: OFData,
    energy: Energies,
    gradient_norm: float,
    lr: float | None = None,
    grad_norm_ratio: float | None = None,
    grad_cos_sim: float | None = None,
) -> str:
    """Return a string for the tqdm progress bar.

    Args:
        sample: The OFData containing the current coefficients. If the ground state energy is
            available, the energy difference to the ground state is calculated.
        energy: The current energy.
        gradient_norm: The norm of the gradient vector.
        lr: The learning rate. If not provided, it is not displayed.
        grad_norm_ratio: Ratio of gradient norm after/before reg_mat (amplification factor).
        grad_cos_sim: Cosine similarity between gradient before/after reg_mat.

    Returns:
        A string for the tqdm progress bar.
    """
    pbar_str = ""
    if "of_labels/energies/ground_state_e_electron" in sample:
        ground_state_electronic_energy = sample["of_labels/energies/ground_state_e_electron"]
        if isinstance(ground_state_electronic_energy, torch.Tensor):
            ground_state_electronic_energy = ground_state_electronic_energy.item()
        delta_e = energy.electronic_energy - ground_state_electronic_energy
        delta_e *= 1e3  # convert to mHa
        pbar_str += f"ΔE={delta_e: .3e} mHa"
    else:
        pbar_str += f"E_elec={energy.electronic_energy:.6f} Ha"

    pbar_str += f", grad_norm={gradient_norm:.3e}"
    if lr is not None:
        pbar_str += f", lr={lr:.2e}"
    if grad_norm_ratio is not None:
        pbar_str += f", |g'|/|g|={grad_norm_ratio:.2f}"
    if grad_cos_sim is not None:
        pbar_str += f", cos={grad_cos_sim:.2f}"
    return pbar_str


class GradientDescent(Optimizer):
    """Simple gradient descent optimizer."""

    def __init__(
        self,
        learning_rate: float,
        convergence_tolerance: float,
        max_cycle: int,
        reg_epsilon: float | None = None,
        reg_gamma: float | None = None,
        hard_projection: bool = False,
        apply_reg_mat_to_step: bool = True,
        eigenvalue_threshold: float | None = None,
        quantile_threshold: float | None = None,
        orthogonal_projection: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        overlap_tensor_block_target_bytes: int = 1_000_000_000,
    ):
        """Initialize the gradient descent optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            learning_rate: The learning rate.
        """
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle
        self.reg_epsilon = reg_epsilon
        self.reg_gamma = reg_gamma
        self.hard_projection = hard_projection
        self.apply_reg_mat_to_step = apply_reg_mat_to_step
        self.eigenvalue_threshold = eigenvalue_threshold
        self.quantile_threshold = quantile_threshold
        self.orthogonal_projection = orthogonal_projection
        self.ks_basis = ks_basis
        self.overlap_tensor_block_target_bytes = overlap_tensor_block_target_bytes

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Perform gradient descent optimization."""
        reg_mat = build_reg_mat(
            sample,
            device=sample.coeffs.device,
            reg_epsilon=self.reg_epsilon,
            reg_gamma=self.reg_gamma,
            hard_projection=self.hard_projection,
            eigenvalue_threshold=self.eigenvalue_threshold,
            quantile_threshold=self.quantile_threshold,
            orthogonal_projection=self.orthogonal_projection,
            ks_basis=self.ks_basis,
            overlap_tensor_block_target_bytes=self.overlap_tensor_block_target_bytes,
        )
        converged = False
        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=position,
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            if reg_mat is None:
                grad_norm_ratio = None
                grad_cos_sim = None
            else:
                projected_gradient, grad_norm_ratio, grad_cos_sim = regularize_gradient(
                    projected_gradient,
                    reg_mat,
                    apply_reg_mat_to_step=self.apply_reg_mat_to_step,
                )
            gradient_norm = torch.norm(projected_gradient).item()

            pbar.set_description(
                get_pbar_str(
                    sample,
                    energy,
                    gradient_norm,
                    self.learning_rate,
                    grad_norm_ratio,
                    grad_cos_sim,
                )
            )
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                learning_rate = self.learning_rate
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            sample.coeffs -= self.learning_rate * projected_gradient
        pbar.close()
        return energy, converged


class TorchOptimizer(Optimizer):
    """Wrapper for torch optimizers to be used in the optimization loop."""

    def __init__(
        self,
        torch_optimizer: Type[torch.optim.Optimizer],
        convergence_tolerance: float,
        max_cycle: int,
        scheduler: functools.partial | None = None,
        reg_epsilon: float | None = None,
        reg_gamma: float | None = None,
        hard_projection: bool = False,
        apply_reg_mat_to_step: bool = True,
        eigenvalue_threshold: float | None = None,
        quantile_threshold: float | None = None,
        orthogonal_projection: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        overlap_tensor_block_target_bytes: int = 1_000_000_000,
        **optimizer_kwargs,
    ):
        """Initialize the torch optimizer.

        Args:
            torch_optimizer: The torch optimizer to use. To be able to apply the optimizer with
                hydra, the class is partially applied without any arguments.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            max_cycle: Maximum number of optimization cycles.
            optimizer_kwargs: Additional keyword arguments for the optimizer.
            scheduler: Optional configuration or callable for an adaptive scheduler controlling
                optimization hyperparameters (e.g. effective step size) during the optimization
                cycles. If None, no scheduling is applied.
            reg_epsilon: Regularization epsilon value for soft regularization (A + ε*I)^{-1} A.
            reg_gamma: Second regularization parameter for overlap matrix.
            hard_projection: If True, use hard projection to zero out small eigenvalue components
                instead of soft regularization.
            apply_reg_mat_to_step: If False, still compute the diagnostic quantities
                ``grad_norm_ratio`` and ``grad_cos_sim`` based on applying ``reg_mat`` to the
                projected gradient, but use the unregularized projected gradient for the actual
                optimization step. This is meant for debugging.
            eigenvalue_threshold: Threshold for hard projection. Eigenvalues below this value
                will have their components zeroed out. Only used if hard_projection=True and
                quantile_threshold is None.
            quantile_threshold: Quantile in [0, 1] used to determine the cutoff for small
                eigenvalues in hard projection. Eigenvalues below the given quantile are treated
                as zeroed-out components. Only used if hard_projection=True and
                eigenvalue_threshold is None.
            orthogonal_projection: If True (default), construct the projection directly in
                coefficient space, yielding an orthogonal projection that cannot amplify norms.
                If False, construct projection in original basis and transform, yielding an
                oblique projection that may amplify norms.
            ks_basis: Configuration flag indicating whether the projection and related
                regularization are applied in the Kohn–Sham basis rather than the original
                basis representation.
        """
        self.torch_optimizer = torch_optimizer
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle
        self.scheduler_cfg = scheduler
        self.reg_epsilon = reg_epsilon
        self.reg_gamma = reg_gamma
        self.hard_projection = hard_projection
        self.apply_reg_mat_to_step = apply_reg_mat_to_step
        self.eigenvalue_threshold = eigenvalue_threshold
        self.quantile_threshold = quantile_threshold
        self.orthogonal_projection = orthogonal_projection
        self.ks_basis = ks_basis
        self.overlap_tensor_block_target_bytes = overlap_tensor_block_target_bytes
        self.optimizer_kwargs = optimizer_kwargs

        assert not (
            self.eigenvalue_threshold is not None and self.quantile_threshold is not None
        ), "Cannot use both eigenvalue_threshold and quantile_threshold for hard projection."

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Optimization loop for a torch optimizer."""
        parameters = sample.coeffs.clone()
        optimizer = self.torch_optimizer([parameters], **self.optimizer_kwargs)

        reg_mat = build_reg_mat(
            sample,
            device=parameters.device,
            reg_epsilon=self.reg_epsilon,
            reg_gamma=self.reg_gamma,
            hard_projection=self.hard_projection,
            eigenvalue_threshold=self.eigenvalue_threshold,
            quantile_threshold=self.quantile_threshold,
            orthogonal_projection=self.orthogonal_projection,
            ks_basis=self.ks_basis,
            overlap_tensor_block_target_bytes=self.overlap_tensor_block_target_bytes,
        )

        converged = False
        if self.scheduler_cfg is not None:
            scheduler = self.scheduler_cfg(optimizer=optimizer)
        else:
            scheduler = None

        # Diagnostics and the pbar string force extra GPU->CPU syncs per cycle; only pay
        # for them when something displays or records them.
        compute_diagnostics = not disable_pbar or callable(callback)

        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=position,
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            if reg_mat is None:
                grad_norm_ratio = None
                grad_cos_sim = None
            else:
                projected_gradient, grad_norm_ratio, grad_cos_sim = regularize_gradient(
                    projected_gradient,
                    reg_mat,
                    apply_reg_mat_to_step=self.apply_reg_mat_to_step,
                    compute_diagnostics=compute_diagnostics,
                )
            gradient_norm = torch.norm(projected_gradient).item()

            if not disable_pbar:
                pbar.set_description(
                    get_pbar_str(
                        sample,
                        energy,
                        gradient_norm,
                        optimizer.param_groups[0]["lr"],
                        grad_norm_ratio,
                        grad_cos_sim,
                    )
                )
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                if "lr" in self.optimizer_kwargs:
                    learning_rate = self.optimizer_kwargs["lr"]
                else:
                    learning_rate = 0
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            optimizer.zero_grad()
            # NOTE: could also use normal gradient here it does not make a difference
            parameters.grad = projected_gradient
            optimizer.step()

            if scheduler is not None:
                if isinstance(
                    scheduler, (AdaptiveSigmoidScheduler, AdaptiveSigmoidSchedulerWithDecay)
                ):
                    scheduler.step(gradient_norm)
                else:
                    scheduler.step()

            update = parameters - sample.coeffs

            # projecting the update is important for, e.g., the Adam optimizer
            sample.coeffs += project_gradient(update, sample)

            # update parameters but clone to not update sample.coeffs in optimizer step
            parameters.data = sample.coeffs.detach().clone()

        pbar.close()
        return energy, converged

    def __str__(self) -> str:
        """Return a string representation of the optimizer.

        This method is overwritten since the torch optimizer can either be a class via direct
        instantiation or via hydra with a partial.
        """
        if isinstance(self.torch_optimizer, functools.partial):  # if called with config
            name = self.torch_optimizer.func.__name__
        else:
            name = self.torch_optimizer.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in self.optimizer_kwargs.items())
        return f"{name}({settings})"


class VectorAdam(Optimizer):
    """Equivariant version of the Adam optimizer."""

    def __init__(
        self,
        max_cycle: int,
        learning_rate: float,
        convergence_tolerance: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        reg_epsilon: float | None = None,
        reg_gamma: float | None = None,
        hard_projection: bool = False,
        apply_reg_mat_to_step: bool = True,
        eigenvalue_threshold: float | None = None,
        quantile_threshold: float | None = None,
        orthogonal_projection: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        overlap_tensor_block_target_bytes: int = 1_000_000_000,
    ):
        """Initialize the equivariant version of the Adam optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles.
            learning_rate: Learning rate function.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            betas: Exponential decay rates for the moment estimates.
            epsilon: Small value to avoid division by zero.
        """
        self.max_cycle = max_cycle
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        self.betas = betas
        self.epsilon = epsilon
        self.reg_epsilon = reg_epsilon
        self.reg_gamma = reg_gamma
        self.hard_projection = hard_projection
        self.apply_reg_mat_to_step = apply_reg_mat_to_step
        self.eigenvalue_threshold = eigenvalue_threshold
        self.quantile_threshold = quantile_threshold
        self.orthogonal_projection = orthogonal_projection
        self.ks_basis = ks_basis
        self.overlap_tensor_block_target_bytes = overlap_tensor_block_target_bytes

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Perform equivariant VectorAdam optimization."""
        converged = False
        m = torch.zeros_like(sample.coeffs)
        v = torch.zeros_like(sample.coeffs)
        beta1, beta2 = self.betas

        shell_beginning_mask = sample.basis_info.shell_beginning_mask
        shell_beginning_mask = torch.as_tensor(shell_beginning_mask, device=sample.coeffs.device)
        shell_indices = torch.cumsum(shell_beginning_mask[sample.basis_function_ind], dim=0) - 1

        reg_mat = build_reg_mat(
            sample,
            device=sample.coeffs.device,
            reg_epsilon=self.reg_epsilon,
            reg_gamma=self.reg_gamma,
            hard_projection=self.hard_projection,
            eigenvalue_threshold=self.eigenvalue_threshold,
            quantile_threshold=self.quantile_threshold,
            orthogonal_projection=self.orthogonal_projection,
            ks_basis=self.ks_basis,
            overlap_tensor_block_target_bytes=self.overlap_tensor_block_target_bytes,
        )

        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                dynamic_ncols=True,
                position=position,
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            # here it only works with the projected gradient unlike in the TorchOptimizer
            projected_gradient = project_gradient(gradient_vector, sample)
            if reg_mat is None:
                grad_norm_ratio, grad_cos_sim = None, None
            else:
                projected_gradient, grad_norm_ratio, grad_cos_sim = regularize_gradient(
                    projected_gradient,
                    reg_mat,
                    apply_reg_mat_to_step=self.apply_reg_mat_to_step,
                )
            gradient_norm = torch.norm(projected_gradient).item()

            m = beta1 * m + (1 - beta1) * projected_gradient
            # squared gradient per shell
            v_per_irrep = global_add_pool(projected_gradient**2, shell_indices)
            v = beta2 * v + (1 - beta2) * v_per_irrep[shell_indices]
            m_hat = m / (1 - beta1 ** (cycle + 1))
            v_hat = v / (1 - beta2 ** (cycle + 1))
            update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            update = project_gradient(update, sample)

            pbar.set_description(
                get_pbar_str(
                    sample,
                    energy,
                    gradient_norm,
                    self.learning_rate,
                    grad_norm_ratio,
                    grad_cos_sim,
                )
            )
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                learning_rate = self.learning_rate
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            sample.coeffs -= update
        pbar.close()
        return energy, converged


def scipy_functional(
    coeffs: np.ndarray,
    sample: OFData,
    energy_functional: Callable,
    convergence_tolerance: float,
    use_projected_gradient: bool,
    pbar: tqdm,
    reg_mat: torch.Tensor | None = None,
    apply_reg_mat_to_step: bool = True,
    callback: Callable | None = None,
) -> tuple[float, np.ndarray]:
    """Functional for scipy optimizers.

    Make the energy functional compatible with scipy optimizers.

    Args:
        coeffs: Input coefficients.
        sample: OFData containing the basis functions and integrals.
        energy_functional: Callable which returns the energy and gradient vector.
        convergence_tolerance: Optimization stops if the gradient norm is below this value.
        use_projected_gradient: Whether to use the projected gradient for the optimization step.
        pbar: tqdm progress bar.
        callback: Optional callback function.

    Returns:
        Energy and gradient vector.
    """
    sample.coeffs = torch.tensor(coeffs, dtype=OFDFT_DTYPE, device=sample.coeffs.device)
    energy, gradient = energy_functional(sample)
    projected_gradient = project_gradient(gradient, sample)
    if reg_mat is None:
        grad_norm_ratio, grad_cos_sim = None, None
    else:
        projected_gradient, grad_norm_ratio, grad_cos_sim = regularize_gradient(
            projected_gradient,
            reg_mat,
            apply_reg_mat_to_step=apply_reg_mat_to_step,
        )
    gradient_norm = torch.norm(projected_gradient).item()

    if callable(callback):
        learning_rate = 0  # needed for callback
        callback({**locals(), "coeffs": sample.coeffs})
    pbar.set_description(
        get_pbar_str(sample, energy, gradient_norm, None, grad_norm_ratio, grad_cos_sim)
    )

    if use_projected_gradient:
        gradient_np = projected_gradient.cpu().numpy()
    else:
        gradient_np = gradient.cpu().numpy()

    if gradient_norm < convergence_tolerance:
        raise StopIteration
    return energy.electronic_energy, gradient_np


class SLSQP(Optimizer):
    """Wrapper for the SLSQP (Sequential Least Squares Programming) optimizer from scipy."""

    def __init__(
        self,
        max_cycle: int,
        convergence_tolerance: float,
        grad_scale: float,
        use_projected_gradient: bool = True,
        reg_epsilon: float | None = None,
        reg_gamma: float | None = None,
        hard_projection: bool = False,
        apply_reg_mat_to_step: bool = True,
        eigenvalue_threshold: float | None = None,
        quantile_threshold: float | None = None,
        orthogonal_projection: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        overlap_tensor_block_target_bytes: int = 1_000_000_000,
    ):
        """Initialize the SLSQP optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles. Note that this is not the same as the
                number of functional evaluations.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            grad_scale: Scaling factor for the gradient vector.
            use_projected_gradient: Whether to use the projected gradient for the optimization
                step.
        """
        self.max_cycle = max_cycle
        self.convergence_tolerance = convergence_tolerance
        self.grad_scale = grad_scale
        self.use_projected_gradient = use_projected_gradient
        self.reg_epsilon = reg_epsilon
        self.reg_gamma = reg_gamma
        self.hard_projection = hard_projection
        self.apply_reg_mat_to_step = apply_reg_mat_to_step
        self.eigenvalue_threshold = eigenvalue_threshold
        self.quantile_threshold = quantile_threshold
        self.orthogonal_projection = orthogonal_projection
        self.ks_basis = ks_basis
        self.overlap_tensor_block_target_bytes = overlap_tensor_block_target_bytes

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Perform density optimization using the SLSQP optimizer."""
        converged = False
        normalization_vector = sample.basis_integrals.cpu().numpy()
        n_electron = sample.n_electron
        pbar = tqdm(
            range(self.max_cycle),
            leave=False,
            dynamic_ncols=True,
            position=position,
            disable=disable_pbar,
        )
        iterator = iter(pbar)

        reg_mat = build_reg_mat(
            sample,
            device=sample.coeffs.device,
            reg_epsilon=self.reg_epsilon,
            reg_gamma=self.reg_gamma,
            hard_projection=self.hard_projection,
            eigenvalue_threshold=self.eigenvalue_threshold,
            quantile_threshold=self.quantile_threshold,
            orthogonal_projection=self.orthogonal_projection,
            ks_basis=self.ks_basis,
            overlap_tensor_block_target_bytes=self.overlap_tensor_block_target_bytes,
        )

        functional = functools.partial(
            scipy_functional,
            sample=sample,
            energy_functional=energy_functional,
            convergence_tolerance=self.convergence_tolerance,
            use_projected_gradient=self.use_projected_gradient,
            pbar=pbar,
            reg_mat=reg_mat,
            apply_reg_mat_to_step=self.apply_reg_mat_to_step,
            callback=callback,
        )

        # decrease the gradient to make the optimization more stable
        def functional_scaled(coeffs: np.ndarray) -> tuple[float, np.ndarray]:
            energy, gradient = functional(coeffs)
            return energy, self.grad_scale * gradient

        def electron_number_constraint(coeffs: np.ndarray) -> float:
            return np.dot(coeffs, normalization_vector) - n_electron

        def derivative_electron_number_constraint(_) -> np.ndarray:
            return normalization_vector

        constraint = {
            "type": "eq",
            "fun": electron_number_constraint,
            "jac": derivative_electron_number_constraint,
        }

        try:
            minimize(
                functional_scaled,
                x0=sample.coeffs.cpu().numpy(),
                method="SLSQP",
                jac=True,
                constraints=constraint,
                options={"maxiter": self.max_cycle, "ftol": 0},  # disable tolerance
                callback=lambda _: next(iterator),
            )
        except StopIteration:
            converged = True
            pass  # StopIteration is raised when the gradient norm is below the tolerance
        pbar.close()
        return energy_functional(sample)[0], converged


class TrustRegionConstrained(Optimizer):
    """Wrapper for the trust region constrained optimizer from scipy."""

    def __init__(
        self,
        max_cycle: int,
        convergence_tolerance: float,
        initial_tr_radius: float,
        initial_constr_penalty: float,
        use_projected_gradient: bool = True,
        reg_epsilon: float | None = None,
        reg_gamma: float | None = None,
        hard_projection: bool = False,
        apply_reg_mat_to_step: bool = True,
        eigenvalue_threshold: float | None = None,
        quantile_threshold: float | None = None,
        orthogonal_projection: bool = True,
        ks_basis: str = "6-31G(2df,p)",
        overlap_tensor_block_target_bytes: int = 1_000_000_000,
    ):
        """Initialize the trust region constrained optimizer.

        Args:
            convergence_tolerance:
            max_cycle: Maximum number of optimization cycles.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            initial_tr_radius: Initial trust radius. Affects the size of the first steps.
            initial_constr_penalty: Initial constraint penalty.
            use_projected_gradient: Whether to use the projected gradient for the optimization step.
        """
        self.max_cycle = max_cycle
        self.convergence_tolerance = convergence_tolerance
        self.initial_tr_radius = initial_tr_radius
        self.initial_constr_penalty = initial_constr_penalty
        self.use_projected_gradient = use_projected_gradient
        self.reg_epsilon = reg_epsilon
        self.reg_gamma = reg_gamma
        self.hard_projection = hard_projection
        self.apply_reg_mat_to_step = apply_reg_mat_to_step
        self.eigenvalue_threshold = eigenvalue_threshold
        self.quantile_threshold = quantile_threshold
        self.orthogonal_projection = orthogonal_projection
        self.ks_basis = ks_basis
        self.overlap_tensor_block_target_bytes = overlap_tensor_block_target_bytes

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
        position: int = 0,
    ) -> tuple[Energies, bool]:
        """Perform trust region constrained optimization."""

        converged = False
        normalization_vector = sample.basis_integrals.cpu().numpy()
        n_electron = sample.n_electron
        pbar = tqdm(
            range(self.max_cycle),
            leave=False,
            dynamic_ncols=True,
            position=position,
            disable=disable_pbar,
        )
        iterator = iter(pbar)

        reg_mat = build_reg_mat(
            sample,
            device=sample.coeffs.device,
            reg_epsilon=self.reg_epsilon,
            reg_gamma=self.reg_gamma,
            hard_projection=self.hard_projection,
            eigenvalue_threshold=self.eigenvalue_threshold,
            quantile_threshold=self.quantile_threshold,
            orthogonal_projection=self.orthogonal_projection,
            ks_basis=self.ks_basis,
            overlap_tensor_block_target_bytes=self.overlap_tensor_block_target_bytes,
        )

        functional = functools.partial(
            scipy_functional,
            sample=sample,
            energy_functional=energy_functional,
            convergence_tolerance=self.convergence_tolerance,
            use_projected_gradient=self.use_projected_gradient,
            pbar=pbar,
            reg_mat=reg_mat,
            apply_reg_mat_to_step=self.apply_reg_mat_to_step,
            callback=callback,
        )
        constraint = LinearConstraint(normalization_vector, n_electron, n_electron)

        def scipy_callback(*_):
            next(iterator)

        try:
            minimize(
                functional,
                x0=sample.coeffs.cpu().numpy(),
                method="trust-constr",
                jac=True,
                tol=0,  # disable tolerance
                constraints=constraint,
                options={
                    "maxiter": self.max_cycle,
                    "initial_tr_radius": self.initial_tr_radius,
                    "initial_constr_penalty": self.initial_constr_penalty,
                },
                callback=scipy_callback,
            )
        except StopIteration:
            converged = True
            pass  # StopIteration is raised when the gradient norm is below the tolerance
        pbar.close()
        return energy_functional(sample)[0], converged


DEFAULT_DENSITY_OPTIMIZER = TorchOptimizer(
    torch.optim.SGD,
    convergence_tolerance=1e-4,
    max_cycle=1000,
    lr=1e-3,
    momentum=0.9,
    scheduler=functools.partial(
        AdaptiveSigmoidSchedulerWithDecay,
        min_lr=1e-3,
        max_lr=3e-3,
        offset=3.5,
        slope=1,
        decay_rate=0.995,
        min_lr_decay=1e-3,
    ),
)
