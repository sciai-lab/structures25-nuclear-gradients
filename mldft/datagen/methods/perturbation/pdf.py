from abc import ABC, abstractmethod

import numpy as np


class L2NormPdf(ABC):
    r"""Interface for :math:`L^2`-norm radial PDFs (probability density functions) used for
    perturbation magnitudes.

    Let :math:`r = \lVert \delta\rho \rVert_2` be the target density distance.
    Implementations of this base class provide a non-negative function :math:`f(r)` on
    :math:`[0, r_\max]`. The sampler later normalizes it to

    .. math::

        f(r) = \frac{f(r)}{\int_0^{r_\max} f(s)\,ds}

    Hence, this interface intentionally allows unnormalized values.
    """

    @abstractmethod
    def __call__(self, radius: np.ndarray) -> np.ndarray:
        r"""Evaluates :math:`f(r)` at given radii."""

    @abstractmethod
    def max_l2_norm(self) -> float:
        r"""Returns :math:`r_\max` with :math:`f(r)=0` for :math:`r > r_\max`."""


class LinearScheduleUniformPdf(L2NormPdf):
    r"""Uniform raw PDF reproducing a linear midpoint-quantile radius schedule.

    Historically, radii were generated linearly as

    .. math::

        r_k = r_\mathrm{max} + k\,\Delta,
        \quad \Delta = \frac{r_\mathrm{min} - r_\mathrm{max}}{K-1},
        \quad k \in \{0,\dots,K-1\},

    To reproduce this with a distribution, we can use a uniform PDF on the interval
    covering the linear schedule. Specifically, given :math:`K` perturbations and
    endpoints :math:`r_\mathrm{max}` and :math:`r_\mathrm{min}`, this class
    sets a constant density on the interval

    .. math::

        \left[r_\mathrm{max} - \frac{\Delta}{2},
        \,r_\mathrm{min} + \frac{\Delta}{2}\right]

    Mid-quantiles :math:`q_k=(k + \frac{1}{2})/K` of a uniform CDF on that interval recover
    the same linear :math:`r_k` sequence.
    """

    def __init__(
        self,
        num_perturbations: int,
        pert_max: float,
        pert_min: float,
    ):
        r"""Initializes a uniform radial density from linear schedule endpoints.

        The support is inferred from the half-step extension around
        :math:`(r_\mathrm{min},r_\mathrm{max})` to ensure that midpoint quantiles of the resulting
        CDF match the linear schedule radii.
        """
        pert_max = float(pert_max)
        pert_min = float(pert_min)

        if num_perturbations <= 1:
            raise ValueError("num_perturbations must be greater than 1 for a linear schedule.")
        if pert_max < pert_min:
            raise ValueError(
                "pert_max must be greater than or equal to pert_min, but got "
                f"pert_max={pert_max} and pert_min={pert_min}."
            )

        support_min, support_max = self._compute_linear_schedule_support(
            pert_max=pert_max,
            pert_min=pert_min,
            num_perturbations=num_perturbations,
        )

        # L2 norms are non-negative by definition, so truncate the support at zero
        # if the half-step extension slightly crosses into negative radii.
        support_min = max(0.0, support_min)

        if support_max < 0.0:
            raise ValueError(
                "support_max must be non-negative, but got " f"support_max={support_max}."
            )

        self.support_min = support_min
        self.support_max = support_max

    def _compute_linear_schedule_support(
        self,
        num_perturbations: int,
        pert_max: float,
        pert_min: float,
    ) -> tuple[float, float]:
        r"""Returns :math:`[r_\mathrm{max}- \frac{\Delta}{2}, \,r_\mathrm{min}+
        \frac{\Delta}{2}]`."""
        step = (pert_min - pert_max) / (num_perturbations - 1)
        left = pert_max - 0.5 * step
        right = pert_min + 0.5 * step
        support_min = min(left, right)
        support_max = max(left, right)
        return support_min, support_max

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        r"""Evaluates :math:`f(r)=\mathbf{1}_{[r_\min,r_\max]}(r)`."""
        radius = np.asarray(radius, dtype=float)
        inside = (radius >= self.support_min) & (radius <= self.support_max)
        return np.where(inside, 1.0, 0.0)

    def max_l2_norm(self) -> float:
        r"""Returns :math:`r_\max` for the uniform support interval."""
        return self.support_max


class CosineAnnealingPdf(L2NormPdf):
    r"""Cosine-annealed radial PDF with compact support.

    This PDF emphasizes small perturbation radii (near the ground state) and
    smoothly tapers to zero at the maximum radius.

    For :math:`r \in [r_\mathrm{min}, r_\mathrm{max}]`, define

    .. math::

        x(r) = \frac{r - r_\mathrm{min}}{r_\mathrm{max} - r_\mathrm{min}}

    and the unnormalized density

    .. math::

        f(r) =
        \begin{cases}
        \Bigl[\frac{1}{2} \Bigl(1 + \cos\!\left(\pi x(r)\right) \Bigr)\Bigr]^{p},
        & r_\mathrm{min} \le r \le r_\mathrm{max}\\
        0, & \text{otherwise}
        \end{cases}

    where :math:`p=\texttt{anneal_power} > 0` controls concentration near
    :math:`r_\mathrm{min}`. This uses half of a cosine period on
    :math:`x\in[0,1]`, so :math:`f'(r_\mathrm{min})=f'(r_\mathrm{max})=0`
    (smooth endpoints).
    """

    def __init__(
        self,
        pert_max: float,
        pert_min: float,
        anneal_power: float,
    ):
        r"""Initializes cosine-annealed support and shape parameters.

        Args:
            pert_max: Upper support bound :math:`r_\mathrm{max}`.
            pert_min: Lower support bound :math:`r_\mathrm{min}`.
            anneal_power: Cosine exponent :math:`p`.
        """
        pert_max = float(pert_max)
        pert_min = float(pert_min)
        anneal_power = float(anneal_power)

        if pert_min < 0.0:
            raise ValueError(f"pert_min must be non-negative, but got pert_min={pert_min}.")
        if pert_max <= pert_min:
            raise ValueError(
                "pert_max must be strictly greater than pert_min, but got "
                f"pert_max={pert_max} and pert_min={pert_min}."
            )
        if anneal_power <= 0.0:
            raise ValueError(
                "anneal_power must be positive, but got " f"anneal_power={anneal_power}."
            )

        self.pert_max = pert_max
        self.pert_min = pert_min
        self.anneal_power = anneal_power

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        r"""Evaluates the unnormalized cosine-annealed profile :math:`f(r)`."""
        radius = np.asarray(radius, dtype=float)
        values = np.zeros_like(radius)

        support = (radius >= self.pert_min) & (radius <= self.pert_max)
        if np.any(support):
            scaled = (radius[support] - self.pert_min) / (self.pert_max - self.pert_min)
            values[support] = (0.5 * (1.0 + np.cos(np.pi * scaled))) ** self.anneal_power

        return values

    def max_l2_norm(self) -> float:
        r"""Returns :math:`r_\mathrm{max}` for compact support."""
        return self.pert_max


class LinearPdf(L2NormPdf):
    r"""Linear radial PDF with compact support.

    This PDF places larger probability density near small radii and decreases linearly
    towards the largest radius.

    For :math:`r \in [r_\mathrm{min}, r_\mathrm{max}]`, define

    .. math::

        x(r) = \frac{r - r_\mathrm{min}}{r_\mathrm{max} - r_\mathrm{min}}

    The unnormalized density is

    .. math::

        f(r) =
        \begin{cases}
        1 - x(r), & r_\mathrm{min} \le r \le r_\mathrm{max}, \\
        0, & \text{otherwise}
        \end{cases}

    So :math:`f(r_\mathrm{min})=1` and :math:`f(r_\mathrm{max})=0`.
    """

    def __init__(
        self,
        pert_max: float,
        pert_min: float,
    ):
        r"""Initializes linear support bounds.

        Args:
            pert_max: Upper support bound :math:`r_\mathrm{max}`.
            pert_min: Lower support bound :math:`r_\mathrm{min}`.
        """
        pert_max = float(pert_max)
        pert_min = float(pert_min)

        if pert_min < 0.0:
            raise ValueError(f"pert_min must be non-negative, but got pert_min={pert_min}.")
        if pert_max <= pert_min:
            raise ValueError(
                "pert_max must be strictly greater than pert_min, but got "
                f"pert_max={pert_max} and pert_min={pert_min}."
            )

        self.pert_max = pert_max
        self.pert_min = pert_min

    def __call__(self, radius: np.ndarray) -> np.ndarray:
        r"""Evaluates the unnormalized linear profile :math:`f(r)`."""
        radius = np.asarray(radius, dtype=float)
        values = np.zeros_like(radius)

        support = (radius >= self.pert_min) & (radius <= self.pert_max)
        if np.any(support):
            scaled = (radius[support] - self.pert_min) / (self.pert_max - self.pert_min)
            values[support] = 1.0 - scaled

        return values

    def max_l2_norm(self) -> float:
        r"""Returns :math:`r_\mathrm{max}` for compact support."""
        return self.pert_max
