"""Per-sample control signals: parameter smoothers and a phase-accumulating LFO.

Every control signal is computed per sample and in closed form per block, so parameter
automation is click-free and the trajectories do not depend on how a signal is cut
into blocks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class OnePole:
    """Per-sample one-pole smoother ``x[n] = x[n-1] + k (target - x[n-1])``.

    :meth:`block` returns the next ``n`` values in closed form,
    ``x[j] = t + (x0 - t) p^(j+1)`` with ``p = exp(-1 / (tau fs))``. It snaps to the
    target once within ``atol``, which makes steady states exact.

    Parameters
    ----------
    value
        Initial value, scalar or array (the shape is kept).
    tau
        Time constant in seconds; ``0`` jumps to every new target.
    fs
        Sample rate in Hz.
    atol
        Distance below which the value snaps to the target.
    """

    def __init__(
        self, value: ArrayLike, tau: float, fs: float, atol: float = 1e-12
    ) -> None:
        self.value = np.array(value, dtype=float)
        self.target = self.value.copy()
        self.pole = float(np.exp(-1.0 / (tau * fs))) if tau > 0 else 0.0
        self.atol = atol

    def set(self, target: ArrayLike, snap: bool = False) -> None:
        """New target; ``snap`` jumps to it."""
        self.target = np.broadcast_to(
            np.asarray(target, dtype=float), self.value.shape
        ).copy()
        if snap:
            self.value = self.target.copy()

    def block(self, n: int) -> np.ndarray:
        """The next ``n`` values, shape ``(n, *value.shape)``."""
        shape = (n,) + self.value.shape
        if np.array_equal(self.value, self.target):
            return np.broadcast_to(self.value, shape).copy()
        decay = (self.pole ** np.arange(1, n + 1)).reshape(
            (n,) + (1,) * self.value.ndim
        )
        trajectory = self.target + (self.value - self.target) * decay
        last = trajectory[-1]
        self.value = np.where(
            np.abs(last - self.target) <= self.atol, self.target, last
        )
        return trajectory


class Ramp:
    """Moves toward its target at a fixed rate (0 -> 1 in ``time`` s), reaching it exactly."""

    def __init__(self, value: float, time: float, fs: float) -> None:
        self.value = float(value)
        self.target = float(value)
        self.step = 1.0 / max(time * fs, 1.0)

    def set(self, target: float, snap: bool = False) -> None:
        """New target; ``snap`` jumps to it."""
        self.target = float(target)
        if snap:
            self.value = self.target

    def block(self, n: int) -> np.ndarray:
        """The next ``n`` values, shape ``(n,)``."""
        distance = self.target - self.value
        if distance == 0.0:
            return np.full(n, self.value)
        moved = np.minimum(self.step * np.arange(1, n + 1), abs(distance))
        trajectory = self.value + np.sign(distance) * moved
        self.value = (
            self.target if moved[-1] >= abs(distance) else float(trajectory[-1])
        )
        return trajectory


class PhaseLFO:
    """A bank of sine LFOs ``sin(2 pi f_i n / fs + phi_i)`` whose rate may change per sample.

    The phase is accumulated (not computed as ``2 pi f_i n / fs``), so a rate change
    bends the frequency without a jump in phase. LFO ``i`` runs at
    ``f_i = f_mod * ratios[i]``, with ``f_mod`` the rate passed to :meth:`block`.

    Parameters
    ----------
    phase
        Initial phases ``phi_i`` in radians, one per LFO.
    ratios
        Rate multiplier ``f_i / f_mod`` of each LFO.
    fs
        Sample rate in Hz.
    """

    def __init__(self, phase: ArrayLike, ratios: ArrayLike, fs: float) -> None:
        self.phase0 = np.array(phase, dtype=float).reshape(-1)
        self.ratios = np.array(ratios, dtype=float).reshape(-1)
        if self.ratios.shape != self.phase0.shape:
            raise ValueError("need one rate ratio per LFO phase")
        self.fs = float(fs)
        self.reset()

    def reset(self) -> None:
        """Back to the initial phases."""
        self.phase = self.phase0.copy()

    def block(self, rate: ArrayLike) -> np.ndarray:
        """``sin`` of the phases over one block, ``(n, K)``, for a rate trajectory ``f_mod`` ``(n,)`` in Hz."""
        rate = np.asarray(rate, dtype=float)
        dphi = 2 * np.pi * rate[:, None] * self.ratios[None, :] / self.fs
        phase = self.phase + np.cumsum(dphi, axis=0) - dphi
        self.phase = (self.phase + dphi.sum(axis=0)) % (2 * np.pi)
        return np.sin(phase)
