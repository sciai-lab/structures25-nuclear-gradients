"""Automatic GPU allocation for the console-script entry points."""

from mldft.utils.gpu_allocation.allocation import ensure_gpu_allocation

__all__ = ["ensure_gpu_allocation"]
