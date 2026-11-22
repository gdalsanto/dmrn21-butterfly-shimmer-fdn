"""Kronecker feedback matrix driven by an external, per-sample angle trajectory."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from pyFDN import kronecker_transform
from pyFDN.td import TimeOperator

from ._utils import as_2d


class ControlledKroneckerMatrix(TimeOperator):
    """Kronecker matrix ``K_M(theta_M) kron ... kron K_1(theta_1)`` with angles set from outside.

    Each ``K_i`` is a 2 x 2 rotation kernel, ``[[cos, -sin], [sin, cos]]``. ``K_1`` mixes
    adjacent pairs of lines and ``K_M`` the first half with the second half; an angle
    of 0 decouples the lines it addresses and ``pi/4`` mixes them equally.

    The fast ``O(N log2 N)`` butterfly of :func:`pyFDN.kronecker_transform`, like
    :class:`pyFDN.td.KroneckerMatrix`, but each block may carry its own angles, one set
    per sample. :class:`pyFDN.td.TimeVaryingKroneckerMatrix` computes its angles from a
    fixed-rate LFO; this operator leaves the trajectory to the caller, so angles can
    follow smoothed parameters, LFOs whose rate is automated, or other control signals
    (in FDN_H: ``theta_H,1`` scaled by ``1 - a_H``). The matrix is orthogonal for every
    angle, so any trajectory is lossless.

    Parameters
    ----------
    angles
        ``M`` kernel angles in radians, innermost (``theta_1``) first. Used for blocks
        that do not pass their own.
    kernel_type
        ``"rotation"`` or ``"reflection"``, once for all kernels or one per kernel.
    """

    def __init__(
        self,
        angles: ArrayLike,
        kernel_type: str | Sequence[str] = "rotation",
    ) -> None:
        self.angles = np.asarray(angles, dtype=float).reshape(-1)
        self.kernel_type = kernel_type
        self.M = self.angles.size
        self.N = 1 << self.M
        self.in_channels = self.out_channels = self.N
        kronecker_transform(np.zeros(self.N), self.angles, kernel_type)

    def process_block(
        self, block: ArrayLike, *, angles: ArrayLike | None = None
    ) -> np.ndarray:
        """Mix one block ``(n, N)``.

        ``angles`` is ``(M,)`` for a fixed matrix over the block or ``(n, M)`` for one
        set per sample; ``None`` uses the angles given at construction.
        """
        x = as_2d(block)
        if x.shape[1] != self.N:
            raise ValueError(
                f"ControlledKroneckerMatrix expects {self.N} input channels"
            )
        theta = self.angles if angles is None else np.asarray(angles, dtype=float)
        if theta.shape[-1] != self.M:
            raise ValueError(
                f"expected {self.M} angles per sample, got {theta.shape[-1]}"
            )
        return kronecker_transform(x, theta, self.kernel_type)
