"""Utilities for deterministic RNG seeding across libraries."""

import random

import numpy as np
import torch
from loguru import logger


def seed_all_rngs(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed to apply. If ``None``, this function is a no-op.
    """
    if seed is None:
        logger.warning("No seed provided; RNGs are left unseeded.")
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Seeded Python/NumPy/PyTorch RNGs with seed={seed}.")
