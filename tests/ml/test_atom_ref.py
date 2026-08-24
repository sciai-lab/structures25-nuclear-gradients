import numpy as np
import pytest
import torch
import torch.nn as nn

from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.loss_function import EnergyLoss
from mldft.ml.models.components.model_output import ModelOutput


@pytest.fixture
def atom_ref_setup(dummy_basis_info):
    """Setup for the atom ref fitter tests.

    Returns:
        data: The data for the tests.
        n_atomic_numbers: The number of atom types.
        atomic_number_to_irreps: The atom type to irreps mapping.
    """

    sample = OFData.construct_new(
        basis_info=dummy_basis_info,
        pos=np.zeros((3, 3), dtype=np.float32),  # not needed for AtomRef
        atomic_numbers=np.array([1, 8, 1], dtype=np.int32),  # H2O molecule
        coeffs=np.concatenate(
            [
                np.array([1] + [0] * 5),  # H
                np.array([1] + [0] * 17),  # O
                np.array([1] + [0] * 5),  # H
            ]
        ).astype(np.float32),
        gradient_label=np.concatenate(
            [
                np.array([1] + [0] * 5),  # H
                np.array([1] + [0] * 17),  # O
                np.array([1] + [0] * 5),  # H
            ],
        ).astype(np.float32),
        energy_label=np.array([28], dtype=np.float32),
        basis_integrals="infer_from_basis",
    )
    sample = to_torch(sample)

    g_z = np.concatenate(
        [
            np.array([1] + [0] * 5),  # H
            np.array([1] + [0] * 17),  # C
            np.array([2] + [0] * 17),  # O
        ]
    ).astype(np.float32)
    t_z = torch.FloatTensor([3, 1, 4])  # H, C, O
    t_global = 1
    n_atom_types = 2

    return sample, n_atom_types, g_z, t_z, t_global


def test_atom_ref(atom_ref_setup):
    """Test the atomic reference module for an example molecule.

    Args:
        atom_ref_setup: Setup for the tests for the atomic reference module
    """
    sample, n_atom_types, g_z, t_z, t_global = atom_ref_setup
    atom_ref = AtomRef(t_z=t_z, t_global=t_global, g_z=g_z)

    reference_energy = atom_ref.sample_forward(sample)

    assert reference_energy == pytest.approx((1 + 1 + 2) + (3 + 3 + 4) + 1)


def test_atom_ref_from_statistics(dummy_dataset_statistics, dummy_loader):
    """Test the atomic reference module for batches from the dummy loader."""

    atom_ref = AtomRef.from_dataset_statistics(dummy_dataset_statistics, weigher_key="constant")

    for batch in dummy_loader:
        atom_ref_energy = atom_ref.sample_forward(batch)
        assert atom_ref_energy.shape == (batch.num_graphs,)


def test_atom_ref_float64_precision(dummy_basis_info):
    """Test whether the energy label and the atom ref energy bias are computed in float64 precision
    regardless of data transformations and model precision settings."""
    sample = OFData.construct_new(
        basis_info=dummy_basis_info,
        pos=np.zeros((3, 3), dtype=np.float32),
        atomic_numbers=np.array([1, 8, 1], dtype=np.int32),
        coeffs=np.concatenate(
            [
                np.array([1] + [0] * 5),
                np.array([1] + [0] * 17),
                np.array([1] + [0] * 5),
            ]
        ).astype(np.float32),
        gradient_label=np.concatenate(
            [
                np.array([1] + [0] * 5),
                np.array([1] + [0] * 17),
                np.array([1] + [0] * 5),
            ],
        ).astype(np.float32),
        energy_label=np.array([4e6 + 1e-6], dtype=np.float64),
        basis_integrals="infer_from_basis",
    )
    sample = to_torch(sample, float_dtype=torch.float32)

    g_z = np.concatenate(
        [
            np.array([1e-6] + [0] * 5),
            np.array([1e-6] + [0] * 17),
            np.array([2e-6] + [0] * 17),
        ]
    ).astype(np.float32)
    t_z = np.array([1e6, 0.0, 1e6], dtype=np.float64)
    atom_ref = AtomRef(t_z=t_z, t_global=1e6, g_z=g_z).to(torch.float32)

    assert sample.energy_label.dtype == torch.float64
    assert sample.energy_label.item() == pytest.approx(4e6 + 1e-6, abs=1e-8)
    assert atom_ref.t_z.dtype == torch.float64
    assert atom_ref.t_global == pytest.approx(1e6)

    pred_energy = atom_ref.sample_forward(sample)
    assert pred_energy.dtype == torch.float64

    loss = EnergyLoss(loss_function=nn.L1Loss(reduction="none"))(
        sample, model_output=ModelOutput(pred_energy=pred_energy)
    )
    expected_delta = torch.tensor([3e-6], dtype=torch.float64)
    assert torch.allclose(loss, expected_delta, atol=1e-8, rtol=0.0)


def test_atom_ref_to_float32_does_not_round_buffers():
    """Test that AtomRef buffers do not take a float32 round trip in model dtype casts."""
    t_z = torch.tensor([1_000_000.03, -2_000_000.07], dtype=torch.float64)
    t_global = torch.tensor(3_000_000.11, dtype=torch.float64)
    g_z = torch.tensor([4_000_000.13, -5_000_000.17, 6_000_000.19], dtype=torch.float64)
    atom_ref = AtomRef(t_z=t_z, t_global=t_global, g_z=g_z)

    expected_t_z = atom_ref.t_z.clone()
    expected_t_global = atom_ref.t_global
    expected_g_z = atom_ref.g_z.clone()

    atom_ref.to(torch.float32)

    assert atom_ref.t_z.dtype == torch.float64
    assert atom_ref.g_z.dtype == torch.float64
    assert torch.equal(atom_ref.t_z, expected_t_z)
    assert atom_ref.t_global == expected_t_global
    assert torch.equal(atom_ref.g_z, expected_g_z)
    assert {"t_z", "g_z"}.issubset(atom_ref.state_dict())
    assert "t_global" not in atom_ref.state_dict()
