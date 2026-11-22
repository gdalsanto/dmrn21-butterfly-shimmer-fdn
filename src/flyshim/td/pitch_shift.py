"""A pitch shifter (PS) with two read pointers that adds no gain inside a feedback loop.

Time compression on a ring buffer: two read pointers, offset by half a modulation period,
advance by ``r = 2^(T_st / 12)`` samples per sample (``T_st`` the transposition in
semitones) and sweep a delay range of ``L_w`` samples, i.e. their delay moves at the
slope ``(1 - r)`` per sample, with Catmull-Rom (cubic) spline reads. pyFDN's
:class:`pyFDN.td.PitchShift` uses the same principle but diverges in long-decay loops, for
two separate reasons: the make-up gain of its built-in ``DCBlocker(correct_loss=True)``,
and its equal-power (sine) crossfade, which adds up to +3 dB per pass when the outputs of
the two read pointers are correlated. :class:`GainSafePitchShift` removes both:

1. **Windows.** Trapezoidal windows ``w_1(n)`` and ``w_2(n)`` with raised-cosine ramps and
   ``w_1(n) + w_2(n) = 1``: each read pointer is silent around its wrap-around, alone on
   a plateau, and active together with the other one for a fraction ``crossfade`` of
   each half period.
2. **Crossfade law.** While both are active, the read pointers are rescaled by
   ``1 / sqrt(w_1^2 + w_2^2 + 2 rho w_1 w_2)``, with ``rho`` the measured short-term
   correlation of their outputs, clipped to [0, 1] and biased high (the larger of a fast
   and a slow estimate). This keeps the energy when ``rho`` is right, never exceeds the
   equal-power level, and is exactly amplitude-complementary (``|y| <= max |s_i|``) for
   fully correlated outputs.

There is no dc blocker inside; a loop places a passive one after the PS
(:class:`flyshim.td.NormalizedDCBlocker`).

At unison the read pointers park on a plateau, where the PS is an exact delay of
``3 + L_w/2`` samples, its mean latency at every transposition. Each channel may have its
own ``L_w``, so that the crossfades of different lines drift apart. The phase of the read
pointers is a fixed-point (int64) accumulator, so the output does not depend on the block
size. A whole block is written to the ring buffer before it is read, which is exact
because every read is at least 3 samples old.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pyFDN.td import TimeOperator
from scipy.signal import lfilter

from ..control import OnePole
from ._utils import as_2d

# fixed-point phase: integer sums are exact, so the output does not depend on the block size
PHASE_ONE = 2**48


def trapezoid(phase: ArrayLike, crossfade: float) -> np.ndarray:
    """Window ``w_1`` of a read pointer at ``phase`` in [0, 1): 0 near the wrap-around, 1 on the plateau around 1/2.

    Raised-cosine ramps of width ``crossfade`` (in periods) centred at 1/4 and 3/4, so
    that ``w_1 + w_2 = trapezoid(p) + trapezoid(p + 1/2) = 1`` for every ``p``.
    """
    phase = np.asarray(phase, dtype=float)
    x = crossfade
    rise = np.clip((phase - (0.25 - 0.5 * x)) / x, 0.0, 1.0)
    fall = np.clip((phase - (0.75 - 0.5 * x)) / x, 0.0, 1.0)
    return np.sin(0.5 * np.pi * rise) ** 2 - np.sin(0.5 * np.pi * fall) ** 2


class GainSafePitchShift(TimeOperator):
    """Gain-safe pitch shifter (PS) with two read pointers, one transposition per channel.

    Parameters
    ----------
    channels
        Number of independent channels.
    fs
        Sample rate in Hz (sets the glide and correlation time constants).
    window
        ``L_w``, the window length: the delay range a read pointer sweeps, in samples;
        even and at least 16. A scalar, or one value per channel.
    crossfade
        Fraction of each half period during which both read pointers are active, in
        (0, 0.5].
    glide
        Time constant in seconds of the transposition smoother.
    corr_times
        Time constants in seconds of the fast and slow correlation estimates.
    park_cents
        Largest detune, in cents, used to steer the read pointers onto a plateau when a
        channel is set to unison.
    max_block
        Largest block handled in one step. Longer blocks are split, which is exact
        except for the glide smoother's rounding.

    Attributes
    ----------
    L_w
        Window length of each channel in samples.
    latency
        Mean delay of each channel in samples, ``3 + L_w/2``; the exact delay at unison.
    """

    MIN_DELAY = (
        3  # keeps the cubic reads, which look two samples ahead, off unwritten samples
    )

    def __init__(
        self,
        channels: int,
        fs: float,
        *,
        window: ArrayLike = 2048,
        crossfade: float = 0.25,
        glide: float = 0.03,
        corr_times: tuple[float, float] = (0.002, 0.02),
        park_cents: float = 25.0,
        max_block: int = 4096,
    ) -> None:
        if not 0.0 < crossfade <= 0.5:
            raise ValueError("crossfade must be in (0, 0.5]")
        self.in_channels = self.out_channels = int(channels)
        self.L_w = np.broadcast_to(
            np.asarray(window, dtype=int), (self.in_channels,)
        ).copy()
        if np.any(self.L_w % 2) or np.any(self.L_w < 16):
            raise ValueError("window lengths must be even and at least 16")
        self.fs, self.crossfade = float(fs), float(crossfade)
        self.max_block = int(max_block)
        # oldest read stays clear of the new block
        self.size = int(self.L_w.max()) + self.max_block + 8
        self.latency = self.MIN_DELAY + self.L_w // 2
        self.park_cents = float(park_cents)
        self.park_inc = (2.0 ** (park_cents / 1200.0) - 1.0) / self.L_w
        self.stagger = (0.5 + np.arange(self.in_channels) / self.in_channels) % 1.0
        self.cents = OnePole(np.zeros(self.in_channels), glide, fs)
        self.k_corr = [1.0 - np.exp(-1.0 / (t * fs)) for t in corr_times]
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def phase(self) -> np.ndarray:
        """Phase of read pointer 1 in periods, [0, 1); read pointer 2 is half a period ahead."""
        return self._phase / PHASE_ONE

    @phase.setter
    def phase(self, value: ArrayLike) -> None:
        value = np.broadcast_to(np.asarray(value, dtype=float), (self.in_channels,))
        self._phase = np.round(value * PHASE_ONE).astype(np.int64) % PHASE_ONE

    def reset(self) -> None:
        """Empty the buffer, stagger the read pointers; the transposition target is kept."""
        self.buffer = np.zeros((self.size, self.in_channels))
        self.write = 0
        # staggered, so that the crossfades of the lines do not line up
        self.phase = self.stagger
        self.zi_corr = np.zeros((len(self.k_corr), 1, 3 * self.in_channels))
        self.set_cents(self.cents.target, snap=True)

    def set_cents(self, cents: ArrayLike, snap: bool = False) -> None:
        """Target transposition in cents, scalar or one per channel; ``snap`` skips the glide.

        Snapping also resets the read-pointer phases: on a plateau centre for channels at
        unison, staggered otherwise. That is only click-free while the buffer is empty,
        e.g. right after construction or :meth:`reset`.
        """
        self.cents.set(cents, snap)
        if snap:
            unison = np.abs(self.cents.target) < 0.5
            self.phase = np.where(
                unison, np.round(2.0 * self.stagger) % 2 / 2.0, self.stagger
            )

    @property
    def ratio(self) -> np.ndarray:
        """Current (smoothed) pitch ratio ``r`` of each channel."""
        return 2.0 ** (self.cents.value / 1200.0)

    def inflight_energy(self) -> float:
        """Energy of the last ``latency`` input samples: what is still in transit at unison."""
        total = 0.0
        for c, lat in enumerate(self.latency):
            idx = (self.write - 1 - np.arange(lat)) % self.size
            total += float(np.sum(self.buffer[idx, c] ** 2))
        return total

    # ------------------------------------------------------------------ processing
    def _increments(self, n: int) -> np.ndarray:
        """Per-sample phase increments (n, C): (1 - r) / L_w, steered onto a plateau at unison."""
        cents = self.cents.block(n)
        inc = (1.0 - 2.0 ** (cents / 1200.0)) / self.L_w
        for c in np.flatnonzero(np.abs(self.cents.target) < 0.5):
            near = np.abs(cents[:, c]) <= self.park_cents
            if not near.any():
                continue
            j0 = int(np.argmax(near))
            phase0 = (self.phase[c] + inc[:j0, c].sum()) % 1.0
            # to the nearest plateau centre, 0 or 1/2
            move = -(((phase0 + 0.25) % 0.5) - 0.25)
            path = np.sign(move) * np.minimum(
                self.park_inc[c] * np.arange(1, n - j0 + 1), abs(move)
            )
            inc[j0:, c] = np.diff(path, prepend=0.0)
        return inc

    def _read(self, position: np.ndarray) -> np.ndarray:
        """Catmull-Rom read of the ring buffer at fractional positions (n, C)."""
        position = position % self.size
        i = np.floor(position).astype(int)
        f = position - i
        cols = np.arange(self.in_channels)[None, :]
        y0 = self.buffer[(i - 1) % self.size, cols]
        y1 = self.buffer[i, cols]
        y2 = self.buffer[(i + 1) % self.size, cols]
        y3 = self.buffer[(i + 2) % self.size, cols]
        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        return ((a * f + b) * f + c) * f + y1

    def _correlation(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Correlation ``rho`` of the read-pointer outputs: max of a fast and a slow one-pole estimate, clipped to [0, 1]."""
        C = self.in_channels
        products = np.concatenate([s1 * s1, s2 * s2, s1 * s2], axis=1)
        rho: np.ndarray | None = None
        for j, k in enumerate(self.k_corr):
            avg, self.zi_corr[j] = lfilter(
                [k], [1.0, k - 1.0], products, axis=0, zi=self.zi_corr[j]
            )
            r = avg[:, 2 * C :] / np.sqrt(avg[:, :C] * avg[:, C : 2 * C] + 1e-30)
            rho = r if rho is None else np.maximum(rho, r)
        assert rho is not None  # at least one correlation time
        return np.clip(rho, 0.0, 1.0)

    def _step(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        self.buffer[(self.write + np.arange(n)) % self.size] = x

        inc = np.round(self._increments(n) * PHASE_ONE).astype(np.int64)
        travelled = np.cumsum(inc, axis=0)
        phase1 = ((self._phase + travelled - inc) % PHASE_ONE) / PHASE_ONE
        phase2 = (phase1 + 0.5) % 1.0
        self._phase = (self._phase + travelled[-1]) % PHASE_ONE

        now = (self.write + np.arange(n))[:, None].astype(float)
        s1 = self._read(now - (self.MIN_DELAY + self.L_w * phase1))
        s2 = self._read(now - (self.MIN_DELAY + self.L_w * phase2))
        self.write = (self.write + n) % self.size

        w1 = trapezoid(phase1, self.crossfade)
        w2 = 1.0 - w1
        y = w1 * s1 + w2 * s2
        rho = self._correlation(s1, s2)
        # = y / sqrt(w1^2 + w2^2 + 2 rho w1 w2)
        return y / np.sqrt(1.0 - 2.0 * (1.0 - rho) * w1 * w2)

    def process_block(self, block: ArrayLike) -> np.ndarray:
        """Shift one block ``(n, C) -> (n, C)``."""
        x = as_2d(block)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"GainSafePitchShift expects {self.in_channels} input channels"
            )
        if len(x) == 0:
            return x.copy()
        if len(x) <= self.max_block:
            return self._step(x)
        return np.concatenate(
            [
                self._step(x[i : i + self.max_block])
                for i in range(0, len(x), self.max_block)
            ]
        )
