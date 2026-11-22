"""Small helpers shared by the operators (same behaviour as pyFDN's private ones)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_2d(block: ArrayLike) -> np.ndarray:
    """Coerce a signal block to a ``(num_samples, channels)`` float array."""
    x = np.asarray(block, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.ndim != 2:
        raise ValueError("signal block must be 1-D or 2-D")
    return x


def s_fade(n: int) -> np.ndarray:
    """0 -> 1 over ``n`` samples along a raised cosine, reaching exactly 1 at the end."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
