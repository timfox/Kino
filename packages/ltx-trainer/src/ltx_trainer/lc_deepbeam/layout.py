"""Limitations / notes for LC-DeepBeam stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "This is a reference stub: it does not implement the paper's U-Net/attention architecture.",
    "RTF/interference subspace estimation is a compact numpy CW demo; the paper uses annotated frame sets.",
    "Loss terms follow Eq. (12) shape (pass + null penalties) but are simplified and not differentiable here.",
    "Beampattern plots are not generated; only a toy wideband beampower helper is included.",
)

