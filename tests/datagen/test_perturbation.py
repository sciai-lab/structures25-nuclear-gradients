import numpy as np
import pytest

from mldft.datagen.methods.perturbation import (
    build_distribution_perturbation_sampler,
)
from mldft.datagen.methods.perturbation.pdf import (
    CosineAnnealingPdf,
    LinearPdf,
    LinearScheduleUniformPdf,
)


def test_l2_distribution_sampler_respects_pdf_support_max():
    """Checks overlap-metric L2 bounds and quantile ordering."""
    overlap = np.eye(6)
    cfg = {
        "num_perturbations": 8,
        "radius_sampling_mode": "midpoint_quantile",
        "l2norm_pdf": {
            "_target_": ("mldft.datagen.methods.perturbation.pdf.LinearPdf"),
            "pert_max": 0.4,
            "pert_min": 0.0,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(3),
    )

    l2_norms = []
    for i in range(cfg["num_perturbations"]):
        perturbation = sampler.sample(i, num_coeffs_p=6)
        l2_norm = float(np.sqrt(perturbation @ overlap @ perturbation))
        l2_norms.append(l2_norm)

    max_radius = cfg["l2norm_pdf"]["pert_max"]
    assert np.all(np.array(l2_norms) <= max_radius + 1e-10)
    assert np.all(np.diff(l2_norms) <= 1e-12)


def test_l2_distribution_linear_schedule_matches_linear_levels():
    """Checks that linear_schedule reproduces descending linear amplitudes."""
    overlap = np.eye(5)
    cfg = {
        "num_perturbations": 5,
        "radius_sampling_mode": "midpoint_quantile",
        "pert_max": 0.10,
        "pert_min": 0.02,
        "l2norm_pdf": {
            "_target_": "mldft.datagen.methods.perturbation.pdf.LinearScheduleUniformPdf",
            "pert_max": 0.10,
            "pert_min": 0.02,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(11),
    )

    sampled_norms = []
    for i in range(cfg["num_perturbations"]):
        perturbation = sampler.sample(i, num_coeffs_p=5)
        sampled_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    expected = np.linspace(cfg["pert_max"], cfg["pert_min"], cfg["num_perturbations"])
    np.testing.assert_allclose(sampled_norms, expected, atol=2e-4, rtol=0.0)


def test_uniform_pdf_handles_descending_schedule_near_zero():
    """Checks LinearScheduleUniformPdf can be built when half-step extension dips below zero."""
    pdf = LinearScheduleUniformPdf(
        num_perturbations=20,
        pert_max=0.1,
        pert_min=0.002,
    )

    assert pdf.support_min == pytest.approx(0.0)
    assert pdf.support_max == pytest.approx(0.10257894736842106)


def test_l2_distribution_descending_uniform_schedule_builds_sampler():
    """Checks sampler construction for descending LinearScheduleUniformPdf schedule endpoints."""
    overlap = np.eye(5)
    cfg = {
        "num_perturbations": 20,
        "radius_sampling_mode": "midpoint_quantile",
        "pert_max": 0.1,
        "pert_min": 0.002,
        "l2norm_pdf": {
            "_target_": "mldft.datagen.methods.perturbation.pdf.LinearScheduleUniformPdf",
            "pert_max": 0.1,
            "pert_min": 0.002,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(13),
    )

    sampled_norms = []
    for i in range(cfg["num_perturbations"]):
        perturbation = sampler.sample(i, num_coeffs_p=5)
        sampled_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    assert np.all(np.array(sampled_norms) >= 0.0)
    assert sampled_norms[0] == pytest.approx(0.1, abs=3e-4)
    assert sampled_norms[-1] == pytest.approx(0.002564, abs=3e-4)


def test_build_sampler_uses_explicit_inverse_transform_mode():
    """Checks explicit config mode sets inverse-transform radius sampling."""
    overlap = np.eye(3)
    cfg = {
        "num_perturbations": 4,
        "radius_sampling_mode": "inverse_transform",
        "l2norm_pdf": {
            "_target_": "mldft.datagen.methods.perturbation.pdf.LinearScheduleUniformPdf",
            "pert_max": 0.10,
            "pert_min": 0.02,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(21),
    )

    assert sampler.radius_sampling_mode == "inverse_transform"


def test_cosine_annealing_pdf_has_compact_support_and_smooth_taper():
    """Checks cosine PDF support and endpoint behavior."""
    pdf = CosineAnnealingPdf(pert_max=0.8, pert_min=0.0, anneal_power=2.0)

    radii = np.array([-0.1, 0.0, 0.2, 0.6, 0.8, 1.0])
    values = pdf(radii)

    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)
    assert values[2] > values[3]
    assert values[4] == pytest.approx(0.0)
    assert values[5] == pytest.approx(0.0)
    assert pdf.max_l2_norm() == pytest.approx(0.8)

    eps = 1e-6
    left_slope = (pdf(np.array([eps]))[0] - pdf(np.array([0.0]))[0]) / eps
    right_slope = (pdf(np.array([0.8]))[0] - pdf(np.array([0.8 - eps]))[0]) / eps
    assert left_slope == pytest.approx(0.0, abs=1e-4)
    assert right_slope == pytest.approx(0.0, abs=1e-4)


def test_l2_distribution_cosine_annealing_biases_towards_ground_state():
    """Checks midpoint-quantile cosine schedule yields decreasing radii in support."""
    overlap = np.eye(4)
    cfg = {
        "num_perturbations": 9,
        "radius_sampling_mode": "midpoint_quantile",
        "l2norm_pdf": {
            "_target_": "mldft.datagen.methods.perturbation.pdf.CosineAnnealingPdf",
            "pert_max": 0.5,
            "pert_min": 0.0,
            "anneal_power": 2.0,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(7),
    )

    l2_norms = []
    for i in range(cfg["num_perturbations"]):
        perturbation = sampler.sample(i, num_coeffs_p=4)
        l2_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    assert np.all(np.array(l2_norms) <= 0.5 + 1e-10)
    assert np.all(np.diff(l2_norms) <= 1e-12)
    assert l2_norms[-1] < 0.2


def test_linear_pdf_has_compact_support_and_linear_taper():
    """Checks linear PDF support and linear endpoint behavior."""
    pdf = LinearPdf(pert_max=0.8, pert_min=0.2)

    radii = np.array([0.0, 0.2, 0.35, 0.5, 0.8, 1.0])
    values = pdf(radii)

    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.75)
    assert values[3] == pytest.approx(0.5)
    assert values[4] == pytest.approx(0.0)
    assert values[5] == pytest.approx(0.0)
    assert pdf.max_l2_norm() == pytest.approx(0.8)


def test_l2_distribution_linear_biases_towards_ground_state():
    """Checks midpoint-quantile linear PDF yields decreasing radii in support."""
    overlap = np.eye(4)
    cfg = {
        "num_perturbations": 9,
        "radius_sampling_mode": "midpoint_quantile",
        "l2norm_pdf": {
            "_target_": "mldft.datagen.methods.perturbation.pdf.LinearPdf",
            "pert_max": 0.5,
            "pert_min": 0.1,
        },
    }

    sampler = build_distribution_perturbation_sampler(
        cfg,
        overlap,
        rng=np.random.default_rng(17),
    )

    l2_norms = []
    for i in range(cfg["num_perturbations"]):
        perturbation = sampler.sample(i, num_coeffs_p=4)
        l2_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    assert np.all(np.array(l2_norms) <= 0.5 + 1e-10)
    assert np.all(np.array(l2_norms) >= 0.1 - 1e-10)
    assert np.all(np.diff(l2_norms) <= 1e-12)
    assert l2_norms[-1] < 0.25
