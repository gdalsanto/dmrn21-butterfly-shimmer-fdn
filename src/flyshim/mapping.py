"""Parameter mappings: shift time T_S -> handover angles theta_SO, T60 -> per-pass gains.

The output of every line ``i`` of FDN_S is split in two: ``gamma_S,i = cos theta_SO,i``
recirculates through the PS and ``Psi_S``, and ``gamma_O,i = sin theta_SO,i`` enters the
``i``-th line of the open half of FDN_H. As ``gamma_S,i^2 + gamma_O,i^2 = 1``, the split
dissipates no energy but hands it over, and FDN_S decays by 60 dB in ``T_S``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def handover_angle(shift_time: float, tau: ArrayLike, fs: float) -> np.ndarray:
    """Per-line handover angle ``theta_SO,i = arccos 10^(-3 tau_i / (T_S fs))``.

    ``shift_time`` is the shift time ``T_S`` in seconds. ``tau`` holds the pass times
    ``tau_i`` of FDN_S in samples: delay-line length ``m_S,i`` plus PS latency.
    ``T_S = 0`` gives ``pi/2`` (the input passes through FDN_S once, untransposed) and
    ``T_S = inf`` gives 0 (nothing is handed over: the classic shimmer FDN).
    """
    tau = np.asarray(tau, dtype=float)
    if shift_time <= 0:
        return np.full(tau.shape, np.pi / 2)
    if np.isinf(shift_time):
        return np.zeros(tau.shape)
    return np.arccos(10.0 ** (-3.0 * tau / (shift_time * fs)))


def t60_gain(t60: float, delay_samples: ArrayLike, fs: float) -> np.ndarray:
    """Per-pass amplitude gain ``10^(-3 m / (T60 fs))`` of a line of ``m`` samples (1 at T60 = inf)."""
    m = np.asarray(delay_samples, dtype=float)
    if np.isinf(t60):
        return np.ones(m.shape)
    if t60 <= 0:
        return np.zeros(m.shape)
    return 10.0 ** (-3.0 * m / (t60 * fs))


def extra_loss_gain(
    shift_time: float, t60_dc: float, tau: ArrayLike, fs: float
) -> np.ndarray:
    """Gains ``g_i = min(1, 10^(-3 tau_i / (T60,dc fs)) / gamma_S,i)`` of FDN_S.

    FDN_S has no attenuation filters, so ``g_i`` ensures it never decays more slowly
    than FDN_H. The total per-pass gain of line ``i`` is then
    ``min(gamma_S,i, 10^(-3 tau_i / (T60,dc fs)))``: ``g_i = 1`` while
    ``T_S <= T60,dc`` (pure handover, nothing dissipated), and FDN_S decays in
    ``T60,dc`` otherwise, which includes the classic shimmer at ``T_S = inf``.
    """
    gamma_S = np.cos(handover_angle(shift_time, tau, fs))
    target = t60_gain(t60_dc, tau, fs)
    safe = np.where(gamma_S > 0, gamma_S, 1.0)
    return np.where(gamma_S > 0, np.minimum(1.0, target / safe), 1.0)
