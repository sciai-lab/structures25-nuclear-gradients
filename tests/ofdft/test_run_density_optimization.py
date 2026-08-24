import pytest
import torch
from pyscf import gto

from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.ofdft.energies import Energies
from mldft.ofdft.optimizer import TorchOptimizer
from mldft.ofdft.run_density_optimization import run_singlepoint_ofdft
from mldft.utils.sample_generator import SampleGenerator


def test_run_single_point_ofdft(generate_run_path):
    """Test the run_singlepoint_ofdft function with a simple molecule."""
    run_path = generate_run_path

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="angstrom")

    model_config = SampleGenerator.from_run_path(run_path, device="cpu")

    func_factory = model_config.get_functional_factory()

    optimizer = TorchOptimizer(torch.optim.SGD, 1e-4, 10, lr=3e-3, momentum=0.9)
    energies, coeffs, converged = run_singlepoint_ofdft(
        mol, model_config, func_factory, optimizer=optimizer
    )

    sample = model_config.get_sample_from_mol(mol)
    normalization_vector = torch.as_tensor(get_normalization_vector(sample.mol), dtype=OFDFT_DTYPE)

    assert not converged
    assert torch.isclose(
        normalization_vector @ coeffs,
        torch.as_tensor(mol.nelectron, dtype=OFDFT_DTYPE),
    )
    assert isinstance(energies, Energies)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.parametrize(
    "device,transform_device",
    [
        # ("cpu", "cpu") is already being tested by the above test function
        ("cuda", "cpu"),
        ("cuda", "cuda"),
        # ("cpu", "cuda") is not being tested as in my opinion there is just no point to it
    ],
)
def test_run_single_point_ofdft_cuda(device, transform_device, generate_run_path):
    """Test the run_singlepoint_ofdft function with a simple molecule."""
    run_path = generate_run_path

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="angstrom")

    model_config = SampleGenerator.from_run_path(
        run_path, device=device, transform_device=transform_device
    )

    func_factory = model_config.get_functional_factory()

    optimizer = TorchOptimizer(torch.optim.SGD, 1e-4, 10, lr=3e-3, momentum=0.9)
    energies, coeffs, converged = run_singlepoint_ofdft(
        mol, model_config, func_factory, optimizer=optimizer
    )

    sample = model_config.get_sample_from_mol(mol)
    normalization_vector = torch.as_tensor(get_normalization_vector(sample.mol), dtype=OFDFT_DTYPE)

    assert not converged
    assert torch.isclose(
        normalization_vector @ coeffs,
        torch.as_tensor(mol.nelectron, dtype=OFDFT_DTYPE),
    )
    assert isinstance(energies, Energies)
