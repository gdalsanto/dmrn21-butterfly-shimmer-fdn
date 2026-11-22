"""flyshim: an evolving shimmer reverb from two coupled feedback delay networks, on pyFDN.

In the shimmer FDN, FDN_S, a pitch shifter (PS) transposes every line at every pass,
and every pass hands part of the energy over to the open-hold FDN, FDN_H, so that
FDN_S decays by 60 dB in the shift time T_S. Tones in FDN_H are never transposed again.
The hold amount a_H closes off the hold half H of FDN_H (Coppola's selective freeze),
which then sustains its content indefinitely while the open half O keeps accumulating
new tones and decaying.

network   ButterflyShimmerFDN (a pyFDN TimeOperator), render() for parameter automation
mapping   shift time T_S -> handover angles theta_SO, T60 -> per-pass gains g
control   per-sample parameter smoothers and a phase-accumulating LFO
td        pyFDN-style operators: GainSafePitchShift, NormalizedDCBlocker, AntiAliasLowpass,
          HoldableAttenuationFilter, ControlledKroneckerMatrix
"""

from __future__ import annotations

from . import td
from .control import OnePole, PhaseLFO, Ramp
from .mapping import extra_loss_gain, handover_angle, t60_gain
from .network import DEFAULTS, ButterflyShimmerFDN, render

__version__ = "0.1.0"

__all__ = [
    "DEFAULTS",
    "ButterflyShimmerFDN",
    "OnePole",
    "PhaseLFO",
    "Ramp",
    "extra_loss_gain",
    "handover_angle",
    "render",
    "t60_gain",
    "td",
]
