from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mldft.ofdft.denop.optimizer_session as optimizer_session_module
import mldft.ofdft.optimizer as optimizer_module
from mldft.ofdft.denop.optimizer_session import build_optimizer_session
from mldft.ofdft.energies import Energies
from mldft.ofdft.optimizer import GradientDescent, TorchOptimizer, VectorAdam


class SampleStub(SimpleNamespace):
    """Tiny object with the subset of OFData container behavior used in optimizer tests."""

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class CallbackRecorder:
    """Capture the callback contract shared by legacy and lockstep optimizers."""

    def __init__(self):
        self.calls = []

    def __call__(self, minimization_locals: dict) -> None:
        self.calls.append(
            {
                "gradient_norm": minimization_locals["gradient_norm"],
                "learning_rate": minimization_locals["learning_rate"],
                "grad_norm_ratio": minimization_locals.get("grad_norm_ratio"),
                "grad_cos_sim": minimization_locals.get("grad_cos_sim"),
                "coeffs": minimization_locals["coeffs"].detach().clone(),
            }
        )


def make_sample(
    coeffs: tuple[float, ...] = (0.4, -0.2, 0.1),
    basis_integrals: tuple[float, ...] = (1.0, 1.5, 0.5),
):
    """Create a minimal sample object supporting projected-gradient sessions."""
    coeffs_tensor = torch.tensor(coeffs, dtype=torch.float64)
    return SampleStub(
        coeffs=coeffs_tensor.clone(),
        basis_integrals=torch.tensor(basis_integrals, dtype=torch.float64),
        basis_function_ind=torch.arange(coeffs_tensor.numel(), dtype=torch.long),
        basis_info=SimpleNamespace(
            shell_beginning_mask=np.array([1] + [0] * (coeffs_tensor.numel() - 1), dtype=int)
        ),
    )


def install_build_reg_mat_stub(
    monkeypatch: pytest.MonkeyPatch, reg_mat: torch.Tensor | None
) -> tuple[list[dict], list[dict]]:
    """Patch both code paths to return a deterministic cached regularization operator."""
    optimizer_calls = []
    session_calls = []

    def _make_stub(recorded_calls: list[dict]):
        def _stub(
            sample,
            *,
            device,
            reg_epsilon,
            reg_gamma,
            hard_projection,
            eigenvalue_threshold,
            quantile_threshold,
            orthogonal_projection,
            ks_basis,
            overlap_tensor_block_target_bytes=1_000_000_000,
        ):
            recorded_calls.append(
                {
                    "device": device,
                    "reg_epsilon": reg_epsilon,
                    "reg_gamma": reg_gamma,
                    "hard_projection": hard_projection,
                    "eigenvalue_threshold": eigenvalue_threshold,
                    "quantile_threshold": quantile_threshold,
                    "orthogonal_projection": orthogonal_projection,
                    "ks_basis": ks_basis,
                    "overlap_tensor_block_target_bytes": overlap_tensor_block_target_bytes,
                }
            )
            if reg_mat is None:
                return None
            return reg_mat.to(device=device, dtype=sample.coeffs.dtype)

        return _stub

    monkeypatch.setattr(optimizer_module, "build_reg_mat", _make_stub(optimizer_calls))
    monkeypatch.setattr(optimizer_session_module, "build_reg_mat", _make_stub(session_calls))
    return optimizer_calls, session_calls


def run_legacy_step(
    optimizer,
    sample,
    gradient_vector: torch.Tensor,
    callback: CallbackRecorder,
) -> tuple[Energies, bool]:
    """Run exactly one legacy optimization iteration using a fixed external gradient."""
    energy = Energies(test=1.25)

    def energy_functional(_sample):
        return energy, gradient_vector.clone()

    converged_energy, converged = optimizer.optimize(
        sample,
        energy_functional,
        callback=callback,
        disable_pbar=True,
    )
    return converged_energy, converged


def assert_callback_parity(
    legacy_callback: CallbackRecorder, session_callback: CallbackRecorder
) -> None:
    """Compare the per-step diagnostics emitted through the callback contract."""
    assert len(legacy_callback.calls) == 1
    assert len(session_callback.calls) == 1
    legacy_call = legacy_callback.calls[0]
    session_call = session_callback.calls[0]

    assert torch.allclose(session_call["coeffs"], legacy_call["coeffs"])
    assert session_call["gradient_norm"] == pytest.approx(legacy_call["gradient_norm"])
    assert session_call["learning_rate"] == pytest.approx(legacy_call["learning_rate"])
    assert session_call["grad_norm_ratio"] == pytest.approx(legacy_call["grad_norm_ratio"])
    assert session_call["grad_cos_sim"] == pytest.approx(legacy_call["grad_cos_sim"])


def test_gradient_descent_session_matches_legacy_without_regularization():
    """Lockstep gradient descent should match the legacy one-step update when reg is off."""
    optimizer = GradientDescent(learning_rate=0.2, convergence_tolerance=-1.0, max_cycle=1)
    gradient_vector = torch.tensor([1.8, -0.4, 0.9], dtype=torch.float64)
    legacy_sample = make_sample()
    session_sample = make_sample()
    legacy_callback = CallbackRecorder()
    session_callback = CallbackRecorder()

    legacy_energy, legacy_converged = run_legacy_step(
        optimizer, legacy_sample, gradient_vector, legacy_callback
    )
    session = build_optimizer_session(optimizer, session_sample)
    step_result = session.step(
        session_sample,
        energy=Energies(test=1.25),
        gradient_vector=gradient_vector.clone(),
        callback=session_callback,
    )

    assert torch.allclose(session_sample.coeffs, legacy_sample.coeffs)
    assert step_result.energy.energies_dict == legacy_energy.energies_dict
    assert step_result.converged is legacy_converged
    assert step_result.finished is True
    assert step_result.learning_rate == pytest.approx(optimizer.learning_rate)
    assert_callback_parity(legacy_callback, session_callback)


@pytest.mark.parametrize("apply_reg_mat_to_step", [True, False])
def test_torch_optimizer_session_matches_legacy_with_soft_regularization(
    monkeypatch: pytest.MonkeyPatch, apply_reg_mat_to_step: bool
):
    """Torch-backed lockstep sessions should match legacy soft-regularized updates."""
    reg_mat = torch.diag(torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64))
    optimizer_calls, session_calls = install_build_reg_mat_stub(monkeypatch, reg_mat)
    optimizer = TorchOptimizer(
        torch.optim.SGD,
        convergence_tolerance=-1.0,
        max_cycle=1,
        lr=0.15,
        reg_epsilon=0.3,
        apply_reg_mat_to_step=apply_reg_mat_to_step,
    )
    gradient_vector = torch.tensor([1.2, -0.2, 0.7], dtype=torch.float64)
    legacy_sample = make_sample()
    session_sample = make_sample()
    legacy_callback = CallbackRecorder()
    session_callback = CallbackRecorder()

    legacy_energy, legacy_converged = run_legacy_step(
        optimizer, legacy_sample, gradient_vector, legacy_callback
    )
    session = build_optimizer_session(optimizer, session_sample)
    step_result = session.step(
        session_sample,
        energy=Energies(test=1.25),
        gradient_vector=gradient_vector.clone(),
        callback=session_callback,
    )

    assert len(optimizer_calls) == 1
    assert len(session_calls) == 1
    assert optimizer_calls[0]["reg_epsilon"] == pytest.approx(0.3)
    assert session_calls[0]["reg_epsilon"] == pytest.approx(0.3)
    assert optimizer_calls[0]["hard_projection"] is False
    assert session_calls[0]["hard_projection"] is False
    assert torch.allclose(session_sample.coeffs, legacy_sample.coeffs)
    assert step_result.energy.energies_dict == legacy_energy.energies_dict
    assert step_result.converged is legacy_converged
    assert step_result.finished is True
    assert_callback_parity(legacy_callback, session_callback)


def test_vector_adam_session_matches_legacy_with_hard_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    """VectorAdam lockstep sessions should mirror the legacy hard-projection path."""
    reg_mat = torch.diag(torch.tensor([1.0, 0.25, 0.75], dtype=torch.float64))
    optimizer_calls, session_calls = install_build_reg_mat_stub(monkeypatch, reg_mat)
    optimizer = VectorAdam(
        max_cycle=1,
        learning_rate=0.05,
        convergence_tolerance=-1.0,
        hard_projection=True,
        quantile_threshold=0.3,
    )
    gradient_vector = torch.tensor([1.0, -0.3, 0.8], dtype=torch.float64)
    legacy_sample = make_sample()
    session_sample = make_sample()
    legacy_callback = CallbackRecorder()
    session_callback = CallbackRecorder()

    legacy_energy, legacy_converged = run_legacy_step(
        optimizer, legacy_sample, gradient_vector, legacy_callback
    )
    session = build_optimizer_session(optimizer, session_sample)
    step_result = session.step(
        session_sample,
        energy=Energies(test=1.25),
        gradient_vector=gradient_vector.clone(),
        callback=session_callback,
    )

    assert len(optimizer_calls) == 1
    assert len(session_calls) == 1
    assert optimizer_calls[0]["hard_projection"] is True
    assert session_calls[0]["hard_projection"] is True
    assert optimizer_calls[0]["quantile_threshold"] == pytest.approx(0.3)
    assert session_calls[0]["quantile_threshold"] == pytest.approx(0.3)
    assert torch.allclose(session_sample.coeffs, legacy_sample.coeffs)
    assert step_result.energy.energies_dict == legacy_energy.energies_dict
    assert step_result.converged is legacy_converged
    assert step_result.finished is True
    assert_callback_parity(legacy_callback, session_callback)


def test_gradient_descent_session_convergence_uses_regularized_gradient(
    monkeypatch: pytest.MonkeyPatch,
):
    """Convergence should follow the regularized projected gradient when it is applied to the
    step."""
    zero_reg_mat = torch.zeros((3, 3), dtype=torch.float64)
    install_build_reg_mat_stub(monkeypatch, zero_reg_mat)
    optimizer = GradientDescent(
        learning_rate=0.2,
        convergence_tolerance=1e-6,
        max_cycle=3,
        reg_epsilon=0.1,
        apply_reg_mat_to_step=True,
    )
    gradient_vector = torch.tensor([1.5, -0.4, 0.7], dtype=torch.float64)
    legacy_sample = make_sample()
    session_sample = make_sample()
    legacy_callback = CallbackRecorder()
    session_callback = CallbackRecorder()
    initial_coeffs = session_sample.coeffs.clone()

    _, legacy_converged = run_legacy_step(
        optimizer, legacy_sample, gradient_vector, legacy_callback
    )
    session = build_optimizer_session(optimizer, session_sample)
    step_result = session.step(
        session_sample,
        energy=Energies(test=1.25),
        gradient_vector=gradient_vector.clone(),
        callback=session_callback,
    )

    assert legacy_converged is True
    assert step_result.converged is True
    assert step_result.gradient_norm == pytest.approx(0.0)
    assert torch.allclose(session_sample.coeffs, initial_coeffs)
    assert torch.allclose(session_sample.coeffs, legacy_sample.coeffs)
    assert_callback_parity(legacy_callback, session_callback)
