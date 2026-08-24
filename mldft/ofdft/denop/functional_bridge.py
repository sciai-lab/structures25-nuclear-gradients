"""Batched functional evaluation bridge for denop lockstep optimization.

Core architectural idea:
- Keep physical-energy composition logic centralized and explicit.
- Batch the model forward pass across active molecules to reduce dispatch overhead.
- Preserve compatibility by evaluating non-model contributions per molecule with the exact same
  formulas as the legacy functional factory path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from pyscf import dft, gto

from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft import OFDFT_DTYPE, libxc_functionals, torch_functionals
from mldft.ofdft.energies import Energies
from mldft.ofdft.functional_factory import (
    LIBXC_PREFIX,
    FunctionalFactory,
    NegativeIntegratedDensity,
    finalize_energies_to_float,
    hartree_functional,
    nuclear_attraction_functional,
)
from mldft.utils.grids import grid_setup


@dataclass
class PreparedFunctionalContext:
    """Per-molecule immutable context for functional evaluation."""

    mol: gto.Mole
    coulomb_matrix: torch.Tensor | None
    nuclear_attraction_vector: torch.Tensor | None
    grid: dft.Grids | None
    grid_weights: torch.Tensor | None
    ao: Any | None
    energy_functional: Callable
    np_ao: Any | None
    non_model_grad: torch.Tensor | None = None


@dataclass
class EnergyGradientResult:
    """Numerical output consumed by optimizer sessions."""

    energies: Energies
    gradient: torch.Tensor


class BatchedFunctionalBridge:
    """Evaluates active molecules using batched model forward + per-sample non-model terms."""

    def __init__(
        self,
        func_factory: FunctionalFactory,
        max_xc_memory: int,
    ):
        """Initialize the bridge for batched and per-sample functional evaluation.

        Args:
            func_factory (FunctionalFactory): Functional factory that defines all
                energy contributions.
            max_xc_memory (int): Memory budget hint for XC evaluations.
            compile_model (bool): Whether to compile the model for faster inference.
        """
        # Side effects:
        # - Stores runtime references to the functional factory and memory budget.
        # - Detects and caches the model contribution module from configured contributions.

        self.func_factory = func_factory
        self.max_xc_memory = max_xc_memory
        self.model_contribution = None
        for contribution in self.func_factory.contributions:
            if isinstance(contribution, torch.nn.Module):
                if isinstance(self.model_contribution, torch.nn.Module):
                    raise RuntimeError(
                        "Model contributions is already a torch.nn.Module instance."
                    )
                    # README: this breaks when there are 2 of them. Are there ever more than 1?
                self.model_contribution = contribution

    @staticmethod
    def _cast_batch_float_tensors(of_batch: OFBatch, target_dtype: torch.dtype) -> OFBatch:
        """Cast only floating-point tensors that do not already match ``target_dtype``."""

        def _cast_if_needed(tensor: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(tensor) or tensor.dtype == target_dtype:
                return tensor
            return tensor.to(dtype=target_dtype)

        return of_batch.apply(_cast_if_needed)

    def prepare(self, sample: OFData) -> PreparedFunctionalContext:
        """Prepare immutable per-sample context used during optimization steps.

        Args:
            sample (OFData): Input sample to prepare.

        Returns:
            PreparedFunctionalContext: Static context for repeated energy/gradient
            evaluations of ``sample``.
        """
        # Side effects:
        # - May remove attributes from ``sample`` (for example ``ao``) during context setup.
        # - May trigger CUDA cache cleanup when AO tensors are moved out of the sample.
        # - Allocates per-sample context objects reused across optimization steps.

        mol = sample.mol

        if hasattr(sample, "grid_level"):
            grid = grid_setup(mol, sample.grid_level, sample.grid_prune)
            grid_weights = torch.as_tensor(
                grid.weights, dtype=OFDFT_DTYPE, device=sample.coulomb_matrix.device
            )
        else:
            grid = None
            grid_weights = None

        if hasattr(sample, "ao"):
            ao = sample.ao
            if ao.device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
                ao = ao.transpose(1, 2).contiguous().transpose(1, 2)
            sample.delete_item("ao")
        else:
            ao = None

        energy_functional = self.func_factory.construct(
            mol=mol,
            coulomb_matrix=sample.coulomb_matrix,
            nuclear_attraction_vector=sample.nuclear_attraction_vector,
            grid=grid,
            ao=ao,
            max_xc_memory=self.max_xc_memory,
        )

        return PreparedFunctionalContext(
            mol=mol,
            coulomb_matrix=sample.coulomb_matrix,
            nuclear_attraction_vector=sample.nuclear_attraction_vector,
            grid=grid,
            grid_weights=grid_weights,
            ao=ao,
            np_ao=ao.cpu().numpy() if ao is not None else None,
            energy_functional=energy_functional,
        )

    def evaluate_active(
        self,
        active_samples: list[OFData],
        active_contexts: list[PreparedFunctionalContext],
    ) -> list[EnergyGradientResult]:
        """Evaluate energies and gradients for all active molecules.

        Args:
            active_samples (list[OFData]): Active samples to evaluate.
            active_contexts (list[PreparedFunctionalContext]): Precomputed contexts
                matching ``active_samples``.

        Returns:
            list[EnergyGradientResult]: One result per sample, in input order.
        """
        # Side effects:
        # - Updates transient prediction/gradient data for each active sample evaluation.
        # - Mutates local gradient tensors in place when adding non-model contributions.

        if self.model_contribution is None:
            return [
                EnergyGradientResult(*context.energy_functional(sample))
                for sample, context in zip(active_samples, active_contexts, strict=True)
            ]

        res_dict, grad_splits = self._batched_model_forward(active_samples)

        results: list[EnergyGradientResult] = []
        for i, (sample, context) in enumerate(zip(active_samples, active_contexts, strict=True)):
            energies = Energies(context.mol)
            energies[self.model_contribution.__name__] = res_dict["pred_energy"][i].item()
            gradient = grad_splits[i]

            non_model_energies, non_model_gradient = self._evaluate_non_model_contributions(
                sample=sample,
                context=context,
            )
            for key, value in non_model_energies.items():
                energies[key] = value
            gradient.add_(non_model_gradient)
            results.append(EnergyGradientResult(energies=energies, gradient=gradient))
        return results

    def _batched_model_forward(
        self, samples: list[OFData]
    ) -> tuple[dict, tuple[torch.Tensor, ...]]:
        """Run one batched model forward pass over active samples.

        Args:
            samples (list[OFData]): Active samples for one lockstep iteration.

        Returns:
            tuple[dict, tuple[torch.Tensor, ...]]: Prediction dictionary and split
            per-sample gradients.
        """

        of_batch = OFBatch.from_data_list(
            samples,
            follow_batch=["coeffs", "atomic_numbers"],
            list_keys=[
                "overlap_matrix",
                "coulomb_matrix",
                "transformation_matrix",
                "inv_transformation_matrix",
            ],
        )

        model_dtype = getattr(self.model_contribution, "dtype", None)
        if model_dtype is not None and of_batch.coeffs.dtype != model_dtype:
            # Boundary cast for model forward only; persistent optimizer/sample state remains
            # in OFDFT_DTYPE outside this temporary batched view.
            of_batch = self._cast_batch_float_tensors(of_batch, model_dtype)

        pred_energy, pred_gradients, _ = self.model_contribution.forward(of_batch)
        if pred_gradients.dtype != OFDFT_DTYPE:
            # Cast model gradients back at the same boundary so optimizer sessions always
            # consume OFDFT_DTYPE gradients.
            pred_gradients = pred_gradients.to(OFDFT_DTYPE)

        # do detach once before splitting instead of per sample after splitting
        pred_energy = pred_energy.detach()
        pred_gradients = pred_gradients.detach()

        # get per-sample results
        slices = of_batch._slice_dict["coeffs"]
        lengths = (slices[1:] - slices[:-1]).tolist()
        grad_splits = torch.split(pred_gradients, lengths)
        # diff_splits = torch.split(pred_diff, lengths) # README: not used anwywhere?

        return (
            {
                "pred_energy": pred_energy,
                "pred_gradient": grad_splits,
                # "pred_diff": diff_splits,
            },
            grad_splits,
        )

    def _evaluate_non_model_contributions(
        self, sample: OFData, context: PreparedFunctionalContext
    ) -> tuple[dict[str, float], torch.Tensor]:
        """Evaluate non-module energy terms and gradients for one sample.

        Args:
            sample (OFData): Active sample containing current coefficients.
            context (PreparedFunctionalContext): Prepared immutable per-sample context.

        Returns:
            tuple[dict[str, float], torch.Tensor]: Mapping from contribution name to
            scalar energy and accumulated gradient tensor.
        """
        # Side effects:
        # - Reuses/zeros ``context.non_model_grad`` as a mutable scratch gradient buffer.
        # - May detach ``sample.coeffs`` during external functional evaluations.
        # - Accumulates gradient contributions in place.

        energies: dict[str, float] = {}
        if (
            context.non_model_grad is None
            or context.non_model_grad.shape != sample.coeffs.shape
            or context.non_model_grad.dtype != sample.coeffs.dtype
            or context.non_model_grad.device != sample.coeffs.device
        ):
            context.non_model_grad = torch.zeros_like(sample.coeffs)
        else:
            context.non_model_grad.zero_()
        gradient = context.non_model_grad

        if len(self.func_factory.torch_functionals) > 0:
            torch_functional_outs = torch_functionals.eval_torch_functionals_blocked_fast(
                context.ao,
                context.grid_weights,
                sample.coeffs,
                self.func_factory.torch_functionals,
                max_memory=self.max_xc_memory,
            )
            for name, (energy, grad) in torch_functional_outs.items():
                gradient.add_(grad)
                energies[name] = energy.item()
            sample.coeffs = sample.coeffs.detach()

        if len(self.func_factory.libxc_functionals) > 0:
            libxc_functionals_outs = libxc_functionals.eval_libxc_functionals(
                sample.coeffs.numpy(force=True),
                self.func_factory.libxc_functionals,
                context.grid,
                context.np_ao,
                self.func_factory.max_derivative,
            )
            grad = libxc_functionals_outs[1]
            gradient.add_(torch.as_tensor(grad, dtype=OFDFT_DTYPE, device=sample.coeffs.device))
            for name, energy in zip(
                self.func_factory.libxc_functionals, libxc_functionals_outs[0]
            ):
                energies[LIBXC_PREFIX + name] = energy

        for contribution in self.func_factory.contributions:
            if isinstance(contribution, torch.nn.Module):
                continue

            if contribution == "hartree":
                energy, grad = hartree_functional(
                    sample.coeffs, coulomb_matrix=context.coulomb_matrix
                )
            elif contribution == "nuclear_attraction":
                energy, grad = nuclear_attraction_functional(
                    sample.coeffs,
                    nuclear_attraction_vector=context.nuclear_attraction_vector,
                )
            elif isinstance(contribution, str):
                if "integrated_negative_density" in contribution:
                    negative_integrated_density = NegativeIntegratedDensity(
                        context.grid_weights,
                        context.ao,
                        gamma=float(contribution[: -len("integrated_negative_density") - 1]),
                    )
                    energy, grad = negative_integrated_density(sample.detach())
                    energy = energy.item()
                    sample.coeffs = sample.coeffs.detach()
                else:
                    # Already accounted for via batched torch/libxc block.
                    continue
            else:
                # Generic callables are intentionally executed per sample in this phase.
                energy, grad = contribution(sample)

            key = contribution if isinstance(contribution, str) else contribution.__name__
            energies[key] = energy
            gradient.add_(grad)

        finalize_energies_to_float(energies)
        return energies, gradient
