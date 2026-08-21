"""Limitations for EDC prediction stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "This is a reference stub: no Pyroomacoustics simulation or trained ConvNet weights.",
    "Loss and RSS helpers are numpy toys for orientation; not a training loop.",
    "Full RIR reconstruction and perceptual validation are out of scope here.",
)
