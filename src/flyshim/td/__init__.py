"""Time-domain operators in pyFDN's ``td`` style, candidates for upstreaming.

Each one is a :class:`pyFDN.td.TimeOperator` (``in_channels``, ``out_channels``,
``process_block``, ``reset``), so it also works inside pyFDN graphs
(:class:`pyFDN.td.Series`, :class:`pyFDN.td.Recursion`). Control signals that change
per block are optional keyword arguments of ``process_block``. See ``docs/upstream.md``.

GainSafePitchShift         pitch shifter (PS), two read pointers, adds no gain inside a loop
NormalizedDCBlocker        pyFDN's dc blocker with a peak gain of exactly 1
AntiAliasLowpass           lowpass ahead of the PS, retuned with the pitch ratio r
HoldableAttenuationFilter  per-line attenuation filters Gamma(z), faded to unity by a hold amount
ControlledKroneckerMatrix  Kronecker butterfly with a per-sample angle trajectory
"""

from __future__ import annotations

from .filters import AntiAliasLowpass, HoldableAttenuationFilter, NormalizedDCBlocker
from .kronecker import ControlledKroneckerMatrix
from .pitch_shift import GainSafePitchShift, trapezoid

__all__ = [
    "AntiAliasLowpass",
    "ControlledKroneckerMatrix",
    "GainSafePitchShift",
    "HoldableAttenuationFilter",
    "NormalizedDCBlocker",
    "trapezoid",
]
