from dataclasses import dataclass, field

import numpy as np
from hydra.utils import instantiate

from mldft.datagen.methods.perturbation.pdf import L2NormPdf


@dataclass(frozen=True)
class DistributionPerturbationSampler:
    r"""Samples density-basis perturbations with prescribed :math:`L^2`-radius statistics.

    Throughout, let :math:`K` denote the number of perturbations to sample
    and let :math:`k` index the perturbations: :math:`k \in \{0,\dots,K-1\}`.
    Let :math:`\mathbf{W}` denote the overlap matrix between density basis
    functions, see :meth:`~mldft.ofdft.basis_integrals.get_overlap_matrix`.

    The sampler generates perturbations as coefficient vectors :math:`\delta\mathbf{p}^{(k)}`
    in the density basis, which have to be added to the ground state density coefficients
    to yield the perturbed density coefficients.

    The distribution of the perturbations follows a user-defined radial PDF on the :math:`L^2`
    norm of the induced density difference, see :meth:`sample` and
    :class:`~mldft.datagen.methods.perturbation.pdf.L2NormPdf`. More precisely,
    the sampler generates perturbations as

    .. math::

        \delta\mathbf{p}^{(k)} = r_k\,\mathbf{u}^{(k)},
        \quad
        (\mathbf{u}^{(k)})^\top \mathbf{W}\,\mathbf{u}^{(k)} = 1

    so that the induced density difference norm satisfies

    .. math::

        \lVert \delta\rho^{(k)} \rVert_2 = r_k

    The radius sequence :math:`r_k` is generated from the configured radial PDF
    either via midpoint quantiles or inverse-transform sampling. This behavior
    is controlled by ``radius_sampling_mode``.
    """

    num_perturbations: int
    overlap: np.ndarray
    l2norm_pdf: L2NormPdf
    radius_sampling_mode: str
    cdf_grid_size: int = 4096
    rng: np.random.Generator = field(
        default_factory=np.random.default_rng,
        repr=False,
        compare=False,
    )
    _cdf_radii: np.ndarray = field(init=False, repr=False, compare=False)
    _cdf_values: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        r"""Validates inputs and precomputes a numerically normalized CDF.

        Given a non-negative probability density :math:`f(r)` from
        ``l2norm_pdf`` on :math:`[0, r_\max]`, this method builds a uniform grid
        :math:`r_i` with :math:`i \in \{0, 1, \dots, N-1\}` and computes

        .. math::

            \widetilde{F}(r_i) \approx \sum_{j=1}^{i}
            \frac{f(r_{j-1}) + f(r_j)}{2}
            (r_j-r_{j-1})

        then normalizes:

        .. math::

            F(r_i) = \frac{\widetilde{F}(r_i)}{\widetilde{F}(r_{N-1})}

        This yields a discrete CDF with :math:`F(0)=0` and :math:`F(r_\max)=1`
        used later for inverse-CDF sampling via interpolation in :meth:`_quantile_to_radius`.
        """
        if self.num_perturbations <= 0:
            raise ValueError(
                "num_perturbations must be positive to build a sampler, "
                f"but got num_perturbations={self.num_perturbations}."
            )
        if self.cdf_grid_size < 2:
            raise ValueError(
                "cdf_grid_size must be at least 2 to build a numerical CDF, "
                f"but got cdf_grid_size={self.cdf_grid_size}."
            )
        if self.overlap.ndim != 2 or self.overlap.shape[0] != self.overlap.shape[1]:
            raise ValueError("overlap must be a square matrix.")

        support_max = self.l2norm_pdf.max_l2_norm()
        radii = np.linspace(0.0, support_max, self.cdf_grid_size)
        weights = np.asarray(self.l2norm_pdf(radii), dtype=float)
        if np.any(weights < 0.0):
            raise ValueError("l2norm_pdf must return non-negative values on [0, support_max].")

        cdf_unnormalized = np.zeros_like(radii)
        increments = 0.5 * (weights[1:] + weights[:-1]) * np.diff(radii)
        cdf_unnormalized[1:] = np.cumsum(increments)
        normalization = float(cdf_unnormalized[-1])
        if normalization <= 0.0:
            raise ValueError("Failed to normalize radius distribution; integral is zero.")

        object.__setattr__(self, "_cdf_radii", radii)
        object.__setattr__(self, "_cdf_values", cdf_unnormalized / normalization)

    def _inverse_cdf(self, quantile: float) -> float:
        r"""Maps :math:`q\in[0,1]` to :math:`r=F^{-1}(q)` by linear interpolation.

        The inverse is evaluated on the precomputed discrete pair :math:`(F(r_i), r_i)`.
        """
        return float(np.interp(quantile, self._cdf_values, self._cdf_radii))

    def _sample_radius_midpoint_quantile(self, index: int) -> float:
        r"""Returns midpoint-quantile radius for perturbation index ``index`` (variable :math:`k`).

        .. math::

            q_k = \frac{K-k-\frac{1}{2}}{K},
            \quad
            r_k = F^{-1}(q_k)
        """
        quantile = (self.num_perturbations - index - 0.5) / self.num_perturbations
        return self._inverse_cdf(quantile)

    def _sample_radius_inverse_transform(self) -> float:
        r"""Samples :math:`r` from :math:`F` via inverse-transform sampling, also called Smirnov
        transform.

        .. math::

            u\sim\mathrm{Uniform}[0,1],\quad r=F^{-1}(u)
        """
        sampled_cdf_value = float(self.rng.uniform(low=0.0, high=1.0))
        return self._inverse_cdf(sampled_cdf_value)

    def _sample_radius(self, index: int) -> float:
        r"""Returns radius according to ``radius_sampling_mode``.

        ``midpoint_quantile`` reproduces the previous deterministic schedule,
        while ``inverse_transform`` samples each radius stochastically.
        """
        if self.radius_sampling_mode == "midpoint_quantile":
            return self._sample_radius_midpoint_quantile(index)
        if self.radius_sampling_mode == "inverse_transform":
            return self._sample_radius_inverse_transform()
        raise ValueError(
            "Unsupported radius_sampling_mode: "
            f"{self.radius_sampling_mode}. Supported modes are "
            "midpoint_quantile and inverse_transform."
        )

    def _sample_metric_normalized_direction(self, num_coeffs_p: int) -> np.ndarray:
        r"""Samples random direction :math:`\mathbf{u}` with
        :math:`\mathbf{u}^\top\mathbf{W}\mathbf{u}=1`.

        The method draws :math:`\mathbf{z}\sim\mathcal{N}(0, 1)` and returns

        .. math::

            \mathbf{u}=\frac{\mathbf{z}}
            {\sqrt{\mathbf{z}^\top\mathbf{W}\mathbf{z}}}

        rejecting near-degenerate draws where
        :math:`\mathbf{z}^\top\mathbf{W}\mathbf{z}` is numerically too small.
        """
        if num_coeffs_p != self.overlap.shape[0]:
            raise ValueError(
                "num_coeffs_p does not match overlap_density shape: "
                f"num_coeffs_p={num_coeffs_p}, "
                f"overlap_density.shape={self.overlap.shape}."
            )

        for _ in range(8):
            direction = self.rng.normal(loc=0.0, scale=1.0, size=num_coeffs_p)
            norm_sq = float(direction @ self.overlap @ direction)
            if norm_sq > 1e-20:
                return direction / np.sqrt(norm_sq)

        raise RuntimeError("Failed to sample a non-degenerate perturbation direction.")

    def sample(self, index: int, num_coeffs_p: int) -> np.ndarray:
        r"""Samples one perturbation as :math:`\delta\mathbf{p}^{(k)} = r_k\mathbf{u}^{(k)}`.

        This method should be called with ``index`` ranging from 0 to
        ``num_perturbations-1`` to sample the full sequence of perturbations.
        For a fixed ``index``, the direction :math:`\mathbf{u}^{(k)}` is always
        stochastic and metric-normalized with the overlap matrix.

        Radius behavior depends on ``radius_sampling_mode``:
          - ``midpoint_quantile``: deterministic :math:`r_k` via midpoint quantiles
          - ``inverse_transform``: stochastic :math:`r_k` via inverse transform sampling
        """
        if not 0 <= index < self.num_perturbations:
            raise ValueError(
                "index must satisfy 0 <= index < num_perturbations, but got "
                f"index={index} with num_perturbations={self.num_perturbations}."
            )
        radius = self._sample_radius(index)
        direction = self._sample_metric_normalized_direction(num_coeffs_p)
        return radius * direction


def build_distribution_perturbation_sampler(
    perturbation_cfg: dict,
    overlap: np.ndarray,
    rng: np.random.Generator | None = None,
) -> DistributionPerturbationSampler:
    """Builds a distribution perturbation sampler from user config.

    For ``LinearScheduleUniformPdf``, ``num_perturbations`` is injected into the PDF for convenience.
    """
    num_perturbations = int(perturbation_cfg["num_perturbations"])

    l2norm_pdf_cfg = dict(perturbation_cfg["l2norm_pdf"])
    target = str(l2norm_pdf_cfg.get("_target_", ""))
    if target.endswith(".LinearScheduleUniformPdf") and "num_perturbations" not in l2norm_pdf_cfg:
        l2norm_pdf_cfg["num_perturbations"] = num_perturbations
    l2norm_pdf: L2NormPdf = instantiate(l2norm_pdf_cfg)

    radius_sampling_mode = str(perturbation_cfg["radius_sampling_mode"])

    rng_processed = rng if rng is not None else np.random.default_rng()

    return DistributionPerturbationSampler(
        num_perturbations=num_perturbations,
        overlap=overlap,
        l2norm_pdf=l2norm_pdf,
        radius_sampling_mode=radius_sampling_mode,
        rng=rng_processed,
    )
