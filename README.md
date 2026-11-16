# Giving Wings to Shimmer Feedback Delay Networks

> [WIP] Source code and notebooks are on their way... -.-" 

Companion repository for the abstract submitted to the Digital Music Research Network one-day workshop (DMRN+21), King's College London, 11 December 2026. See [`abstract.pdf`](abstract.pdf).

A stereo algorithm that combines time-varying operations and lossless parametric matrices in two coupled feedback delay networks (FDNs) [1] to produce an evolving shimmer reverberation effect. It brings together two ideas presented at DAFx26:

- shimmer reverb via time compression and expansion in the FDN feedback loop [2];
- creative modulation of the feedback matrix based on recursive Kronecker products [3].

New to FDNs? An FDN is an artificial reverberation algorithm based on recirculating delay lines. If you want to learn more, you might find this useful: [delay-network-reverbs](https://github.com/gdalsanto/delay-network-reverbs).

## Audio examples

See [`clips/`](clips/). For now, there is only one example, showcasing pitch shifting, modulation, and freeze.


## Setup

The implementation builds on [pyFDN](https://github.com/artificial-audio/pyFDN), included as a submodule:

```bash
git submodule update --init
```

## References

1. J.-M. Jot and A. Chaigne, "Digital delay networks for designing artificial reverberators," in *Proc. 90th Audio Eng. Soc. Conv.*, 1991.
2. G. Dal Santo, X. Pi, K. Prawda, S. J. Schlecht, and V. Välimäki, "Shimmer reverberation with nonlinear feedback delay networks," in *Proc. 29th Int. Conf. Digital Audio Effects (DAFx26)*, 2026, pp. 40–47.
3. A. Coppola, "Fast parametric matrices for lossless feedback delay networks," in *Proc. 29th Int. Conf. Digital Audio Effects (DAFx26)*, 2026, pp. 32–39.
