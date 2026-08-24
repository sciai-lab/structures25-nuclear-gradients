import numpy as np
import pytest

from mldft.datagen.methods.perturbation.pdf import L2NormPdf
from mldft.datagen.methods.perturbation.sampler import (
    DistributionPerturbationSampler,
)


class ConstantSupportPdf(L2NormPdf):
    """Simple compact-support constant PDF used to test sampler invariants."""

    def __init__(self, support_max: float) -> None:
        r"""Initializes a constant PDF with support on :math:`[0, \texttt{support_max}]`."""
        self.support_max = float(support_max)

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        r"""Evaluates :math:`f(r)=\mathbf{1}_{[0,r_\max]}(r)`."""
        radius = np.asarray(radius, dtype=float)
        inside = (radius >= 0.0) & (radius <= self.support_max)
        return np.where(inside, 1.0, 0.0)

    def max_l2_norm(self) -> float:
        r"""Returns :math:`r_\max` for the constant support interval."""
        return self.support_max


class NegativePdf(L2NormPdf):
    """PDF test double that returns negative values to trigger validation."""

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        return -np.ones_like(radius)

    def max_l2_norm(self) -> float:
        return 1.0


class ZeroPdf(L2NormPdf):
    """PDF test double with zero integral to trigger normalization failure."""

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        radius = np.asarray(radius, dtype=float)
        return np.zeros_like(radius)

    def max_l2_norm(self) -> float:
        return 1.0


def test_sampler_metric_norms_match_uniform_midpoint_quantiles():
    """Checks overlap-metric norms against analytic midpoint-quantile radii."""
    overlap = np.diag(np.array([1.0, 2.0, 4.0, 8.0]))
    support_max = 0.6
    num_perturbations = 6
    sampler = DistributionPerturbationSampler(
        num_perturbations=num_perturbations,
        overlap=overlap,
        l2norm_pdf=ConstantSupportPdf(support_max=support_max),
        radius_sampling_mode="midpoint_quantile",
        cdf_grid_size=32768,
        rng=np.random.default_rng(17),
    )

    sampled_norms = []
    for index in range(num_perturbations):
        perturbation = sampler.sample(index, num_coeffs_p=overlap.shape[0])
        sampled_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    expected = support_max * (
        (num_perturbations - np.arange(num_perturbations) - 0.5) / num_perturbations
    )
    np.testing.assert_allclose(sampled_norms, expected, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("index", [-1, 3])
def test_sampler_rejects_out_of_range_index(index):
    """Checks index validation in sample(index, ...)."""
    sampler = DistributionPerturbationSampler(
        num_perturbations=3,
        overlap=np.eye(2),
        l2norm_pdf=ConstantSupportPdf(support_max=1.0),
        radius_sampling_mode="midpoint_quantile",
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match="index must satisfy"):
        sampler.sample(index=index, num_coeffs_p=2)


def test_sampler_rejects_too_small_cdf_grid_size():
    """Checks that CDF construction requires at least two grid points."""
    with pytest.raises(ValueError, match="cdf_grid_size must be at least 2"):
        DistributionPerturbationSampler(
            num_perturbations=3,
            overlap=np.eye(2),
            l2norm_pdf=ConstantSupportPdf(support_max=1.0),
            radius_sampling_mode="midpoint_quantile",
            cdf_grid_size=1,
            rng=np.random.default_rng(0),
        )


def test_sampler_rejects_non_square_overlap():
    """Checks overlap validation rejects non-square matrices."""
    with pytest.raises(ValueError, match="overlap must be a square matrix"):
        DistributionPerturbationSampler(
            num_perturbations=3,
            overlap=np.ones((2, 3)),
            l2norm_pdf=ConstantSupportPdf(support_max=1.0),
            radius_sampling_mode="midpoint_quantile",
            rng=np.random.default_rng(0),
        )


def test_sampler_rejects_zero_num_perturbations():
    """Checks sampler requires a strictly positive perturbation count."""
    with pytest.raises(ValueError, match="num_perturbations must be positive"):
        DistributionPerturbationSampler(
            num_perturbations=0,
            overlap=np.eye(2),
            l2norm_pdf=ConstantSupportPdf(support_max=1.0),
            radius_sampling_mode="midpoint_quantile",
            rng=np.random.default_rng(0),
        )


def test_sampler_rejects_negative_pdf_values_on_support():
    """Checks CDF construction rejects negative density values."""
    with pytest.raises(ValueError, match="must return non-negative"):
        DistributionPerturbationSampler(
            num_perturbations=3,
            overlap=np.eye(2),
            l2norm_pdf=NegativePdf(),
            radius_sampling_mode="midpoint_quantile",
            rng=np.random.default_rng(0),
        )


def test_sampler_rejects_zero_integral_pdf():
    """Checks CDF normalization fails for identically zero PDFs."""
    with pytest.raises(ValueError, match="integral is zero"):
        DistributionPerturbationSampler(
            num_perturbations=3,
            overlap=np.eye(2),
            l2norm_pdf=ZeroPdf(),
            radius_sampling_mode="midpoint_quantile",
            rng=np.random.default_rng(0),
        )


def test_sampler_inverse_transform_respects_support_and_is_not_midpoint_schedule():
    """Checks inverse-transform radii stay in support and differ from midpoint quantiles."""
    overlap = np.diag(np.array([1.0, 2.0, 4.0, 8.0]))
    support_max = 0.6
    num_perturbations = 6
    sampler = DistributionPerturbationSampler(
        num_perturbations=num_perturbations,
        overlap=overlap,
        l2norm_pdf=ConstantSupportPdf(support_max=support_max),
        cdf_grid_size=32768,
        radius_sampling_mode="inverse_transform",
        rng=np.random.default_rng(17),
    )

    sampled_norms = []
    for index in range(num_perturbations):
        perturbation = sampler.sample(index, num_coeffs_p=overlap.shape[0])
        sampled_norms.append(float(np.sqrt(perturbation @ overlap @ perturbation)))

    midpoint_expected = support_max * (
        (num_perturbations - np.arange(num_perturbations) - 0.5) / num_perturbations
    )
    sampled_norms = np.asarray(sampled_norms)
    assert np.all(sampled_norms >= 0.0)
    assert np.all(sampled_norms <= support_max + 1e-10)
    assert not np.allclose(sampled_norms, midpoint_expected, atol=1e-4, rtol=0.0)


def test_sampler_rejects_invalid_radius_sampling_mode():
    """Checks invalid radius mode is rejected when sampling."""
    sampler = DistributionPerturbationSampler(
        num_perturbations=3,
        overlap=np.eye(2),
        l2norm_pdf=ConstantSupportPdf(support_max=1.0),
        radius_sampling_mode="invalid",
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match="Unsupported radius_sampling_mode"):
        sampler.sample(index=0, num_coeffs_p=2)


def test_sampler_rejects_num_coeffs_mismatch():
    """Checks sampled direction validates dimensional consistency with overlap."""
    sampler = DistributionPerturbationSampler(
        num_perturbations=3,
        overlap=np.eye(2),
        l2norm_pdf=ConstantSupportPdf(support_max=1.0),
        radius_sampling_mode="midpoint_quantile",
        rng=np.random.default_rng(0),
    )

    with pytest.raises(ValueError, match="num_coeffs_p does not match overlap_density"):
        sampler.sample(index=0, num_coeffs_p=3)
