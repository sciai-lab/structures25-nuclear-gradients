import numpy as np
import pytest

from mldft.datagen.methods.label_generation_from_ground_state import (
    calculate_labels_with_ground_state_perturbations,
)
from mldft.utils.molecules import build_mol_with_even_tempered_basis

ATOL = 1e-02


@pytest.mark.slow
@pytest.mark.parametrize(
    "perturbation_cfg",
    [
        {
            "num_perturbations": 2,
            "radius_sampling_mode": "inverse_transform",
            "l2norm_pdf": {
                "_target_": "mldft.datagen.methods.perturbation.pdf.LinearScheduleUniformPdf",
                "pert_max": 2e-3,
                "pert_min": 1e-3,
            },
        },
        {
            "num_perturbations": 2,
            "radius_sampling_mode": "inverse_transform",
            "l2norm_pdf": {
                "_target_": "mldft.datagen.methods.perturbation.pdf.LinearScheduleUniformPdf",
                "pert_max": 0.03,
                "pert_min": 0.0,
            },
        },
        {
            "num_perturbations": 2,
            "radius_sampling_mode": "inverse_transform",
            "l2norm_pdf": {
                "_target_": "mldft.datagen.methods.perturbation.pdf.CosineAnnealingPdf",
                "pert_max": 0.03,
                "pert_min": 0.0,
                "anneal_power": 2.0,
            },
        },
    ],
)
def test_label_generation_from_ground_state(ksdft_results_small, perturbation_cfg):
    """Checks label generation when perturbing around the converged ground state."""
    results, initialization, data_of_iteration, mol_orbital_basis = ksdft_results_small
    mol_density_basis = build_mol_with_even_tempered_basis(mol_orbital_basis)

    data = calculate_labels_with_ground_state_perturbations(
        results,
        initialization,
        data_of_iteration,
        mol_orbital_basis,
        mol_density_basis,
        perturbation_cfg,
    )

    num_samples = perturbation_cfg["num_perturbations"] + 1
    assert data["of_coeffs"].shape[0] == num_samples
    assert data["of_grad_kin"].shape[0] == num_samples
    assert data["of_grad_xc"].shape[0] == num_samples
    assert data["of_grad_kinapbe"].shape[0] == num_samples
    assert data["ks_e_xc"].shape[0] == num_samples
    assert np.all(data["ks_has_energy_label"])

    assert pytest.approx(data["ks_e_xc"][-1], abs=ATOL) == data["of_e_xc"][-1]
    assert pytest.approx(data["ks_e_ext"][-1], abs=ATOL) == data["of_e_ext"][-1]
    assert pytest.approx(data["ks_e_electron"][-1], abs=ATOL) == data["of_e_electron"][-1]
    assert pytest.approx(data["ks_e_kin"][-1], abs=ATOL) == data["of_e_kin"][-1]

    assert not np.any(np.isnan(data["of_grad_kin"]))
    assert not np.any(np.isnan(data["of_grad_xc"]))
    assert not np.any(np.isnan(data["of_grad_kinapbe"]))
