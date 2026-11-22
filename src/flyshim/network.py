"""The butterfly shimmer FDN: two coupled FDNs, the shimmer FDN, FDN_S, and the open-hold
FDN, FDN_H.

Notation follows the paper. FDN_H has ``N = 2^M`` lines and FDN_S has ``N/2``. The lines
of FDN_H split into the open half O and the hold half H, interleaved so that the
innermost kernel ``K_1`` of ``Psi_H`` pairs every open line with a hold line: in 0-based
Python indexing, H is lines 0, 2, 4, ... and O is lines 1, 3, 5, .... In both networks
the first half of the lines carries the left channel and the second half the right.

Names: ``_S`` marks FDN_S. ``_O`` and ``_H`` mark the open and hold halves of FDN_H
(``m_O``, ``s_H``, ``Gamma_H``, ``a_H``), except in ``Psi_H`` and ``theta_H``, which
belong to the whole of FDN_H, as in the paper. ``_OH`` marks a vector over all lines of
FDN_H, in the interleaved order ``Psi_H`` acts on.

Per sample, with ``s_S(n)``, ``s_O(n)`` and ``s_H(n)`` the delay-line outputs of FDN_S and
of the two halves of FDN_H, and ``*`` the element-wise product:

    s_S(n + m_S) = Psi_S PS(g * gamma_S * s_S(n)) + B x(n)

    [s_O(n + m_O)]           [Gamma_O s_O(n) + gamma_O * s_S(n)]
    [s_H(n + m_H)] = Psi_H(n) [Gamma~_H s_H(n)                 ]

    y(n) = C [l s_S(n); s_O(n); s_H(n)]

with ``gamma_S = cos theta_SO``, ``gamma_O = sin theta_SO``,
``Gamma~_H(z) = (1 - a_H) Gamma_H(z) + a_H``, and

    Psi_S    = K_{M-1}(theta_S,M-1) kron ... kron K_1(theta_S,1)
    Psi_H(n) = K_M(theta_H,M) kron K_{M-1}(theta_H,M-1(n)) kron ... kron K_1((1 - a_H) theta_H,1(n))
    theta_H,i(n) = pi/4 + d sin(2 pi f_i n / fs + phi_i),  i = 1, ..., M-1

As in the paper, the FDN_S equation leaves out two filters of its loop, which here runs
``Psi_S DC(PS(AA(g * gamma_S * s_S(n))))``: an anti-aliasing lowpass AA ahead of the
PS, and the dc blocker DC after it.

``theta_SO,i = arccos 10^(-3 tau_i / (T_S fs))``, with ``T_S`` the shift time and
``tau_i`` the pass time of line ``i`` of FDN_S in samples (``m_S,i`` plus the PS
latency): on every pass line ``i`` keeps ``gamma_S,i^2`` of its energy and hands
``gamma_O,i^2`` to the ``i``-th line of the open half, so FDN_S decays by 60 dB in
``T_S`` and nothing is dissipated. The gains ``g_i < 1`` only when ``T_S > T60,dc``.
FDN_H has no path back into FDN_S, so held tones are never transposed again.

``a_H`` is the hold amount (parameter ``hold_amount``), moved along a 0.5 s S-shaped ramp.
At ``a_H = 1``, ``K_1 = I``: H and O exchange nothing, and H receives no input and loses
nothing, so H sustains its content indefinitely (Coppola, DAFx26, Section 5.3). O keeps
accumulating new tones and decays with ``T60,dc``, so the network stays bounded with no
level detector. Every intermediate ``a_H`` is passive: ``Psi_H(n)`` is orthogonal for
any angles, and ``|(1 - a_H) Gamma_H,i(e^jw) + a_H| <= 1``.

The loop is written out explicitly on two :class:`pyFDN.td.RecursionState` delay banks,
as :func:`pyFDN.process_dss` does; the operators inside it are the ``flyshim.td`` ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pyFDN.td import RecursionState, TimeOperator

from pyFDN import decay_to_one_pole, sample_delay_lengths

from .control import OnePole, PhaseLFO, Ramp
from .mapping import extra_loss_gain, handover_angle
from .td import (
    AntiAliasLowpass,
    ControlledKroneckerMatrix,
    GainSafePitchShift,
    HoldableAttenuationFilter,
    NormalizedDCBlocker,
)
from .td._utils import as_2d

DEFAULTS: dict[str, Any] = {
    # s, shift time T_S: FDN_S decays by 60 dB in T_S; 0 = no shift, inf = classic shimmer
    "shift_time": 3.0,
    # semitones, transposition per pass (T_st,L, T_st,R); a scalar sets both
    "transposition": (12.0, 12.0),
    # s, T60,dc: reverberation time of FDN_H at dc (and of FDN_S when T_S > T60,dc)
    "t60_dc": 8.0,
    # s, T60,Ny: reverberation time of FDN_H at the Nyquist frequency
    "t60_ny": 4.0,
    # hold amount a_H, 0 to 1: closes off the hold half H of FDN_H
    "hold_amount": 0.0,
    # rad, d: depth of the LFOs on theta_H,1..M-1
    "mod_depth": 0.05 * np.pi,
    # Hz, f_mod: rate of the LFOs
    "mod_rate": 0.3,
    # rad, theta_S,M-1: stereo coupling in FDN_S (0 = a separate cascade per channel)
    "stereo_coupling_S": 0.0,
    # rad, theta_H,M: stereo coupling in FDN_H
    "stereo_coupling_H": np.pi / 8,
    # shimmer level l: output level of FDN_S
    "shimmer_level": 1.0,
}


def _num_kernels(N: int) -> int:
    """Number of kernels ``M`` of an ``N x N`` Kronecker matrix, ``N = 2^M >= 2``."""
    M = int(np.log2(N))
    if N < 2 or 2**M != N:
        raise ValueError(f"line counts must be powers of two >= 2, got {N}")
    return M


def _stereo_gains(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Input gains ``B`` ``(n, 2)`` and output gains ``C`` ``(2, n)``.

    The left channel goes to and from the first half of the lines, the right channel to
    and from the second half, with random signs: ``+-1/sqrt(n/2)``.
    """
    h = n // 2
    signs = rng.choice([-1.0, 1.0], size=(2, n)) / np.sqrt(h)
    B, C = np.zeros((n, 2)), np.zeros((2, n))
    B[:h, 0], B[h:, 1] = signs[0, :h], signs[0, h:]
    C[0, :h], C[1, h:] = signs[1, :h], signs[1, h:]
    return B, C


class ButterflyShimmerFDN(TimeOperator):
    """Stereo shimmer FDN with a shift time ``T_S`` and a hold amount ``a_H``.

    A :class:`pyFDN.td.TimeOperator` from 2 input to 2 output channels (wet only). A mono
    input ``(T,)`` or ``(T, 1)`` feeds both channels.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    n_shift, n_hold
        Lines in FDN_S and FDN_H, ``N/2`` and ``N`` in the paper: powers of two with
        ``n_hold >= 2 n_shift`` (every line of FDN_S feeds its own line of the open
        half). In each network the first half of the lines carries the left channel and
        the second half the right channel.
    shift_pass
        ``(min, max)`` pass time ``tau`` of FDN_S in seconds, delay line ``m_S`` plus PS
        latency: the time between two successive transpositions.
    hold_delays
        ``(min, max)`` delay-line lengths ``m_O`` and ``m_H`` of FDN_H in seconds.
    window
        ``L_w``, the PS window length: the delay range the read pointers sweep, in
        samples. Each line of FDN_S gets an ``L_w`` within +-12 % of it, so that their
        crossfades drift apart.
    crossfade
        Fraction of each half period during which both read pointers of the PS are
        active.
    block_size
        Loop block, capped at the shortest delay line. The output does not depend on
        it: every control signal is computed per sample.
    seed
        Delays, gain signs and LFO phases ``phi_i``.
    **params
        Initial values of the automatable parameters (see ``DEFAULTS``), applied without
        smoothing. Change them later with :meth:`set`, or schedule them with
        :func:`render`.

    Attributes
    ----------
    tau
        Pass times ``tau_i`` of FDN_S in samples: ``m_S,i`` plus the PS latency.
    m_S, m_O, m_H
        Delay-line lengths in samples of FDN_S and of the open and hold halves of FDN_H.
    m_OH
        Delay-line lengths of all lines of FDN_H, interleaved as ``Psi_H`` sees them.
    lines_O, lines_H
        Indices of the open and hold halves among the lines of FDN_H.
    lines_SO
        The line of the open half that each line of FDN_S feeds, through ``gamma_O``.
    B, C
        Input gains ``(N/2, 2)``, and output gains ``(2, 3N/2)`` on
        ``[l s_S; s_O; s_H]``.
    Psi_S, Psi_H, PS
        Feedback matrices of FDN_S and FDN_H, and the pitch shifter of FDN_S.
    """

    def __init__(
        self,
        fs: float = 48000,
        *,
        n_shift: int = 8,
        n_hold: int = 16,
        shift_pass: tuple[float, float] = (0.09, 0.16),
        hold_delays: tuple[float, float] = (0.025, 0.09),
        window: int = 2048,
        crossfade: float = 0.25,
        block_size: int = 1024,
        seed: int = 0,
        **params: Any,
    ) -> None:
        # kernels in Psi_S and Psi_H: M_S = M - 1 when n_hold = 2 n_shift
        M_S, M = _num_kernels(n_shift), _num_kernels(n_hold)
        if n_hold < 2 * n_shift:
            raise ValueError(
                "n_hold must be at least 2 * n_shift (one open line per FDN_S line)"
            )
        rng = np.random.default_rng(seed)
        self.fs, self.n_shift, self.n_hold = fs, n_shift, n_hold
        self.in_channels = self.out_channels = 2

        def lengths(n: int, seconds: tuple[float, float]) -> np.ndarray:
            low, high = (int(round(t * fs)) for t in seconds)
            return sample_delay_lengths(
                n, (low, high), distribution="geometric", coprime=True, rng=rng
            )

        # tau_i = m_S,i + PS latency
        self.tau = lengths(n_shift, shift_pass)
        self.m_OH = lengths(n_hold, hold_delays)
        spread = np.linspace(-0.12, 0.12, n_shift)[rng.permutation(n_shift)]
        L_w = 2 * np.round(window * (1 + spread) / 2).astype(int)
        self.m_S = self.tau - (GainSafePitchShift.MIN_DELAY + L_w // 2)
        if self.m_S.min() < 1:
            raise ValueError("FDN_S pass times must be longer than the PS latency")
        self.block_size = int(min(block_size, self.m_S.min(), self.m_OH.min()))

        # FDN_S: delay lines z^-m_S; in the loop anti-aliasing, PS, dc blocker, Psi_S
        h_s, h_h = n_shift // 2, n_hold // 2
        self.delays_S = RecursionState(self.m_S, self.block_size)
        self.anti_alias = AntiAliasLowpass(
            [np.arange(h_s), np.arange(h_s, n_shift)], fs
        )
        self.PS = GainSafePitchShift(
            n_shift, fs, window=L_w, crossfade=crossfade, max_block=self.block_size
        )
        self.dc_blocker = NormalizedDCBlocker(n_shift, fs)
        self.Psi_S = ControlledKroneckerMatrix(np.full(M_S, np.pi / 4))
        # FDN_H: delay lines z^-m_O and z^-m_H (interleaved), attenuation filters, Psi_H
        self.delays_OH = RecursionState(self.m_OH, self.block_size)
        self.attenuation = HoldableAttenuationFilter(n_hold)
        self.Psi_H = ControlledKroneckerMatrix(np.full(M, np.pi / 4))

        self.B, C_S = _stereo_gains(n_shift, rng)
        # FDN_H takes no input (its input gains are drawn for the seed's sake)
        _, C_OH = _stereo_gains(n_hold, rng)
        self.lines_H = np.arange(0, n_hold, 2)  # hold half H: closed off by a_H
        self.lines_O = np.arange(1, n_hold, 2)  # open half O: always live, fed by FDN_S
        self.m_O, self.m_H = self.m_OH[self.lines_O], self.m_OH[self.lines_H]
        self.mask_H = np.zeros(n_hold)
        self.mask_H[self.lines_H] = 1.0
        # C acts on [l s_S; s_O; s_H]
        self.C = np.hstack([C_S, C_OH[:, self.lines_O], C_OH[:, self.lines_H]])
        rows = 2 * np.arange(h_s) + 1
        # line i of FDN_S feeds the i-th line of the open half on its side
        self.lines_SO = np.concatenate([rows, h_h + rows])

        # Psi_S: theta_S,1..M_S-1 = pi/4; outermost theta_S,M_S = stereo_coupling_S
        self.theta_S_inner = np.full(M_S - 1, np.pi / 4)
        # Psi_H: theta_H,1..M-1 = pi/4 + LFO; outermost theta_H,M = stereo_coupling_H
        self.theta_H_inner = np.full(M - 1, np.pi / 4)
        # f_i / f_mod, spread over one octave around f_mod; random initial phases phi_i
        f_ratios = 2.0 ** np.linspace(-0.5, 0.5, max(M - 1, 1))[: M - 1]
        self.lfo = PhaseLFO(2 * np.pi * rng.random(M - 1), f_ratios, fs)

        t_smooth = 0.05  # s, time constant of the parameter smoothers
        self._theta_SO = OnePole(np.zeros(n_shift), t_smooth, fs)
        self._g = OnePole(np.ones(n_shift), t_smooth, fs)
        self._hold_ramp = Ramp(0.0, 0.5, fs)  # linear; a_H is its S-shaped map
        # 1 / T60,dc and 1 / T60,Ny, so that inf is 0
        self._inv_t60_dc = OnePole(0.0, t_smooth, fs)
        self._inv_t60_ny = OnePole(0.0, t_smooth, fs)
        self._d = OnePole(0.0, 0.1, fs)
        self._f_mod = OnePole(0.0, 0.1, fs)
        self._theta_S_outer = OnePole(0.0, t_smooth, fs)
        self._theta_H_outer = OnePole(0.0, t_smooth, fs)
        self._ell = OnePole(1.0, t_smooth, fs)
        # (T60,dc, T60,Ny) of the attenuation filters
        self._tuned = (np.inf, np.inf)
        self.params = dict(DEFAULTS)
        self.reset()
        self.set(snap=True, **params)

    # ------------------------------------------------------------------ parameters
    def set(self, snap: bool = False, **params: Any) -> None:
        """Set parameter targets (see ``DEFAULTS``), reached smoothly unless ``snap``."""
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown parameters {sorted(unknown)}")
        p = {**self.params, **params}
        if (
            p["shift_time"] < 0
            or p["t60_dc"] <= 0
            or p["t60_ny"] <= 0
            or not 0 <= p["hold_amount"] <= 1
        ):
            raise ValueError(
                "need shift_time >= 0, t60_dc > 0, t60_ny > 0 and 0 <= hold_amount <= 1"
            )
        self.params = p
        T_st = np.broadcast_to(np.asarray(p["transposition"], dtype=float), (2,))
        self.PS.set_cents(100.0 * np.repeat(T_st, self.n_shift // 2), snap)
        self._theta_SO.set(handover_angle(p["shift_time"], self.tau, self.fs), snap)
        self._g.set(
            extra_loss_gain(p["shift_time"], p["t60_dc"], self.tau, self.fs), snap
        )
        self._hold_ramp.set(float(p["hold_amount"]), snap)
        self._inv_t60_dc.set(1.0 / p["t60_dc"], snap)  # 1 / inf = 0
        self._inv_t60_ny.set(1.0 / p["t60_ny"], snap)
        self._d.set(p["mod_depth"], snap)
        self._f_mod.set(p["mod_rate"], snap)
        self._theta_S_outer.set(p["stereo_coupling_S"], snap)
        self._theta_H_outer.set(p["stereo_coupling_H"], snap)
        self._ell.set(p["shimmer_level"], snap)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Empty every delay line and filter; the parameters are kept."""
        for part in (
            self.delays_S,
            self.anti_alias,
            self.PS,
            self.dc_blocker,
            self.delays_OH,
            self.attenuation,
            self.lfo,
        ):
            part.reset()
        if hasattr(self, "params"):
            self.set(snap=True)

    def stored_energy(self) -> dict[str, float]:
        """Energy stored in FDN_S (delay lines and PS) and in the two halves of FDN_H.

        Returns ``{"S", "O", "H"}``: FDN_S, the open half and the hold half.
        """
        e_OH = np.sum(self.delays_OH.buffer**2, axis=1)
        return {
            "S": float(np.sum(self.delays_S.buffer**2)) + self.PS.inflight_energy(),
            "O": float(e_OH[self.lines_O].sum()),
            "H": float(e_OH[self.lines_H].sum()),
        }

    # ------------------------------------------------------------------ processing
    def process_block(self, block: ArrayLike) -> np.ndarray:
        """Run the network on a block of any length, ``(T,)``, ``(T, 1)`` or ``(T, 2)``.

        Returns the wet signal, ``(T, 2)``. The loop runs in blocks of ``block_size``.
        """
        x = as_2d(block)
        if x.shape[1] == 1:
            x = np.column_stack([x[:, 0], x[:, 0]])
        if x.shape[1] != 2:
            raise ValueError("ButterflyShimmerFDN expects 1 or 2 input channels")
        y = np.empty((len(x), 2))
        for start in range(0, len(x), self.block_size):
            y[start : start + self.block_size] = self._loop(
                x[start : start + self.block_size]
            )
        return y

    def _retune(self, n: int) -> None:
        """Redesign Gamma_O(z) and Gamma_H(z) when T60,dc or T60,Ny has moved."""
        t60_dc, t60_ny = (
            np.inf if inv == 0 else 1.0 / inv
            for inv in (
                float(self._inv_t60_dc.block(n)[0]),
                float(self._inv_t60_ny.block(n)[0]),
            )
        )
        if (t60_dc, t60_ny) != self._tuned:
            self.attenuation.retune(
                decay_to_one_pole(t60_dc, self.m_OH, self.fs, rt_nyquist=t60_ny)
            )
            self._tuned = (t60_dc, t60_ny)

    def _loop(self, x: np.ndarray) -> np.ndarray:
        n, h = len(x), self.n_shift // 2
        theta_SO = self._theta_SO.block(n)  # (n, N/2) handover angles
        g = self._g.block(n)  # (n, N/2), < 1 only when T_S > T60,dc
        # hold amount a_H: S-shaped, exactly 0 and 1 at the ends
        a_H = 0.5 - 0.5 * np.cos(np.pi * self._hold_ramp.block(n))
        self._retune(n)

        # FDN_S: s_S(n + m_S) = Psi_S DC(PS(AA(g * gamma_S * s_S(n)))) + B x(n)
        s_S = self.delays_S.get_values(n)
        r = self.PS.ratio  # the ratio before this block's glide sets the cutoff
        z = self.anti_alias.process_block(
            np.cos(theta_SO) * g * s_S, ratios=(r[:h].max(), r[h:].max())
        )
        z = self.dc_blocker.process_block(self.PS.process_block(z))
        z = self.Psi_S.process_block(z, angles=self._theta_S(n))
        self.delays_S.set_values(z + x @ self.B.T)
        self.delays_S.advance(n)

        # FDN_H: Gamma_O on O, Gamma~_H = (1 - a_H) Gamma_H + a_H on H, gamma_O * s_S into O
        s_OH = self.delays_OH.get_values(n)
        z = self.attenuation.process_block(s_OH, amount=a_H[:, None] * self.mask_H)
        z[:, self.lines_SO] += np.sin(theta_SO) * s_S
        z = self.Psi_H.process_block(z, angles=self._theta_H(n, a_H))
        self.delays_OH.set_values(z)
        self.delays_OH.advance(n)

        # y(n) = C [l s_S(n); s_O(n); s_H(n)]
        ell = self._ell.block(n)[:, None]
        s = np.hstack([ell * s_S, s_OH[:, self.lines_O], s_OH[:, self.lines_H]])
        return s @ self.C.T

    def _theta_S(self, n: int) -> np.ndarray:
        """Kernel angles of Psi_S, ``(n, M_S)``: ``pi/4``, then the stereo coupling outermost.

        With ``M_S = M - 1`` these are ``theta_S,1..M-2 = pi/4`` and ``theta_S,M-1``.
        """
        inner = np.broadcast_to(self.theta_S_inner, (n, len(self.theta_S_inner)))
        return np.column_stack([inner, self._theta_S_outer.block(n)])

    def _theta_H(self, n: int, a_H: np.ndarray) -> np.ndarray:
        """Kernel angles of Psi_H(n), ``(n, M)``.

        ``theta_H,i(n) = pi/4 + d sin(2 pi f_i n / fs + phi_i)`` for ``i = 1, ..., M-1``,
        with ``theta_H,1`` (O against H) scaled by ``1 - a_H``, exactly 0 at
        ``a_H = 1``; then the stereo coupling ``theta_H,M`` outermost.
        """
        lfo = self.lfo.block(self._f_mod.block(n))
        theta_H = self.theta_H_inner + self._d.block(n)[:, None] * lfo
        theta_H[:, 0] *= 1.0 - a_H
        return np.column_stack([theta_H, self._theta_H_outer.block(n)])


def render(
    fdn: ButterflyShimmerFDN,
    x: ArrayLike,
    events: Iterable[tuple[float, Mapping[str, Any]]] = (),
) -> np.ndarray:
    """Process ``x`` through ``fdn`` with parameter changes: ``events = [(time in s, {name: value}), ...]``.

    The changes go through :meth:`ButterflyShimmerFDN.set`, so they are smoothed exactly
    as live automation is.
    """
    x = np.asarray(x, dtype=float)
    cuts = sorted(
        ((int(round(t * fdn.fs)), p) for t, p in events), key=lambda cut: cut[0]
    )
    out, start = [], 0
    for at, params in [*cuts, (len(x), None)]:
        at = min(max(at, start), len(x))
        if at > start:
            out.append(fdn.process_block(x[start:at]))
        if params:
            fdn.set(**params)
        start = at
    return np.concatenate(out) if out else np.zeros((0, 2))
