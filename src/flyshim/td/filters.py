"""Passive filters for feedback loops: ``|H(e^jw)| <= 1`` at every frequency, in every state.

NormalizedDCBlocker        pyFDN's dc blocker with its peak gain normalised to exactly 1
AntiAliasLowpass           lowpass ahead of a pitch shifter (PS), retuned as the ratio r moves
HoldableAttenuationFilter  per-line attenuation filters Gamma(z), faded to unity by a hold amount

Retuning a recursive filter by swapping its coefficients under a running state gives a
small transient. The retunable filters therefore run the old and the new coefficients
from the same state over one block and crossfade the two outputs; a convex combination
of passive filters is passive.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from pyFDN.td import DCBlocker, TimeOperator
from scipy.signal import butter, sosfilt

from ..control import Ramp
from ._utils import as_2d, s_fade


class NormalizedDCBlocker(DCBlocker):
    """pyFDN's :class:`~pyFDN.td.DCBlocker` scaled to a peak gain of exactly 1.

    ``y(n) = (1 + R) / 2 (x(n) - x(n-1)) + R y(n-1)``, with pole ``R``. Without the
    fixed gain ``(1 + R) / 2``, the blocker peaks at ``2 / (1 + R)`` at the Nyquist
    frequency: +0.003 dB at a 5 Hz cutoff, +0.02 dB at pyFDN's default pole 0.995. That
    is harmless once, but inside a loop it is applied on every pass. With the gain the
    filter is passive at every frequency, so a loop that contains it stays a
    contraction by construction.

    The loss-compensating envelope follower of the parent (``correct_loss``) is always
    off: a level-dependent make-up gain inside a loop is not passive.

    Parameters
    ----------
    channels
        Number of channels processed independently.
    fs
        Sample rate in Hz.
    cutoff
        Cutoff frequency in Hz; sets the pole ``R = exp(-2 pi cutoff / fs)``. The 5 Hz
        default costs a recirculating path -0.03 dB per pass at 60 Hz.
    """

    def __init__(self, channels: int, fs: float, cutoff: float = 5.0) -> None:
        R = float(np.exp(-2.0 * np.pi * cutoff / fs))
        super().__init__(channels, pole=R, correct_loss=False)
        self.cutoff = float(cutoff)
        self.scale = 0.5 * (1.0 + R)  # (1 + R) / 2

    def process_block(self, block: ArrayLike) -> np.ndarray:
        return self.scale * super().process_block(block)


class AntiAliasLowpass(TimeOperator):
    """Second-order Butterworth lowpass at ``0.45 fs / r`` ahead of a pitch shifter.

    Shifting up by a ratio ``r = 2^(T_st / 12)`` makes the read pointers advance ``r``
    samples per sample, so content above ``fs / (2 r)`` would fold back as aliasing.
    The cutoff follows the current ratio of each group of channels (one group per
    transposition, e.g. ``T_st,L`` and ``T_st,R``), with coefficient changes
    crossfaded over one block. Shifting down
    does not alias, so a group's filter is faded out over ``fade`` seconds while its
    ratio is at most 1. The fade ``y = (1 - b) x + b H x`` is passive for every ``b``
    because ``|H| <= 1``.

    Parameters
    ----------
    groups
        Channel indices of each group; together they cover the channels to filter.
    fs
        Sample rate in Hz.
    fade
        Fade time of the filter in seconds when a group starts or stops shifting up.
    """

    def __init__(
        self, groups: Sequence[ArrayLike], fs: float, fade: float = 0.01
    ) -> None:
        self.groups = [np.asarray(g, dtype=int) for g in groups]
        self.in_channels = self.out_channels = int(
            max(int(g.max()) for g in self.groups) + 1
        )
        self.fs = float(fs)
        self.ratios = np.ones(len(self.groups))
        self.zi = [np.zeros((1, 2, len(g))) for g in self.groups]
        self.sos: list[np.ndarray | None] = [None] * len(self.groups)
        self.mix = [Ramp(0.0, fade, fs) for _ in self.groups]

    def _design(self, r: float) -> np.ndarray:
        return np.asarray(
            butter(2, 0.45 * self.fs / max(r, 1.0), fs=self.fs, output="sos")
        )

    def process_block(
        self, block: ArrayLike, *, ratios: ArrayLike | None = None
    ) -> np.ndarray:
        """Filter one block ``(n, C)``.

        ``ratios`` holds the current pitch ratio of each group; it is kept for later
        blocks that do not pass one.
        """
        x = as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"AntiAliasLowpass expects {self.in_channels} input channels"
            )
        if ratios is not None:
            self.ratios = np.broadcast_to(
                np.asarray(ratios, dtype=float), (len(self.groups),)
            ).copy()
        y = x.copy()
        for k, (lines, r) in enumerate(zip(self.groups, self.ratios, strict=True)):
            self.mix[k].set(1.0 if r > 1.0 + 1e-6 else 0.0)
            b = self.mix[k].block(len(x))
            if not b.any():
                self.zi[k][:], self.sos[k] = 0.0, None
                continue
            sos = self._design(float(r))
            xk = x[:, lines]
            low, zi = sosfilt(sos, xk, axis=0, zi=self.zi[k])
            old_sos = self.sos[k]
            if old_sos is not None and not np.array_equal(sos, old_sos):
                old, _ = sosfilt(old_sos, xk, axis=0, zi=self.zi[k])
                low = old + s_fade(len(x))[:, None] * (low - old)
            self.zi[k], self.sos[k] = zi, sos
            y[:, lines] = xk + b[:, None] * (low - xk)
        return y

    def reset(self) -> None:
        for k, mix in enumerate(self.mix):
            self.zi[k][:], self.sos[k] = 0.0, None
            mix.set(0.0, snap=True)


class HoldableAttenuationFilter(TimeOperator):
    """Per-line attenuation filters ``Gamma_i(z)``, each faded to unity by a hold amount ``a``.

    ``Gamma~_i(z) = (1 - a) Gamma_i(z) + a``. With ``|Gamma_i(e^jw)| <= 1`` every ``a``
    in [0, 1] is passive, and ``a = 1`` is exactly unity: a lossless line, which is what
    holding a delay line needs. The filters keep running while ``a = 1``, so releasing
    the hold is click-free as well.

    The filters are an SOS bank in pyFDN's canonical ``(n_sections, 6, N)`` layout, e.g.
    from :func:`pyFDN.decay_to_one_pole`. A new bank set with :meth:`retune` takes over
    with a one-block crossfade from the old one.

    Parameters
    ----------
    channels
        Number of lines ``N``.
    sos
        Initial bank, ``(n_sections, 6, N)``; ``None`` starts at the identity.
    """

    def __init__(self, channels: int, sos: ArrayLike | None = None) -> None:
        self.in_channels = self.out_channels = int(channels)
        identity = np.zeros((1, 6, self.in_channels))
        identity[0, 0], identity[0, 3] = 1.0, 1.0
        self.sos = self._layout(identity)
        self.pending: np.ndarray | None = None
        self._state = np.zeros((self.in_channels, 1, 2))
        if sos is not None:
            self.sos = self._layout(sos)
            self._state = np.zeros((self.in_channels, self.sos.shape[1], 2))

    def _layout(self, sos: ArrayLike) -> np.ndarray:
        """Canonical ``(n_sections, 6, N)`` -> ``(N, n_sections, 6)`` for sosfilt."""
        bank = np.asarray(sos, dtype=float)
        if bank.ndim != 3 or bank.shape[1] != 6 or bank.shape[2] != self.in_channels:
            raise ValueError(
                f"sos must have shape (n_sections, 6, {self.in_channels}); got {bank.shape}"
            )
        return np.ascontiguousarray(bank.transpose(2, 0, 1))

    def retune(self, sos: ArrayLike) -> None:
        """New filters, ``(n_sections, 6, N)`` with as many sections as the current bank.

        Applied with a crossfade over the next block.
        """
        bank = self._layout(sos)
        if bank.shape != self.sos.shape:
            raise ValueError("a retuned bank must keep the number of sections")
        self.pending = bank

    def _run(self, bank: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y, state = np.empty_like(x), np.empty_like(self._state)
        for i in range(self.in_channels):
            y[:, i], state[i] = sosfilt(
                bank[i], np.ascontiguousarray(x[:, i]), zi=self._state[i]
            )
        return y, state

    def process_block(self, block: ArrayLike, *, amount: ArrayLike = 0.0) -> np.ndarray:
        """Filter one block ``(n, N)``.

        ``amount`` is the hold amount ``a``: a scalar, ``(n, 1)`` for all lines, or
        ``(n, N)`` per sample and line. ``0`` is the plain filter ``Gamma_i(z)``, ``1``
        is unity.
        """
        x = as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"HoldableAttenuationFilter expects {self.in_channels} input channels"
            )
        a = np.asarray(amount, dtype=float)
        attenuated, state = self._run(
            self.sos if self.pending is None else self.pending, x
        )
        if self.pending is not None:
            old, _ = self._run(self.sos, x)
            attenuated = old + s_fade(len(x))[:, None] * (attenuated - old)
            self.sos, self.pending = self.pending, None
        self._state = state
        # (1 - a) Gamma(z) x + a x
        return x + (1.0 - a) * (attenuated - x)

    def reset(self) -> None:
        """Zero the filter states; the filters are kept."""
        self._state[:] = 0.0
