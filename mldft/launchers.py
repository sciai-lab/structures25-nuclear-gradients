"""GPU-aware console-script entry points.

These wrappers first auto-allocate GPUs via
:func:`mldft.utils.gpu_allocation.ensure_gpu_allocation` and only then import the
heavy hydra/torch ``main``, such that ``CUDA_VISIBLE_DEVICES`` is set before any
CUDA context is created.
"""

from mldft.utils.gpu_allocation import ensure_gpu_allocation


def train() -> None:
    """Launch training with automatic GPU selection."""
    ensure_gpu_allocation()
    from mldft.ml.train import main

    main()


def run_density_optimization() -> None:
    """Launch density optimization with automatic GPU selection."""
    ensure_gpu_allocation()
    from mldft.ofdft.run_density_optimization import main

    main()


def run_geometry_optimization() -> None:
    """Launch geometry optimization with automatic GPU selection."""
    ensure_gpu_allocation()
    from mldft.ofdft.run_geometry_optimization import main

    main()
