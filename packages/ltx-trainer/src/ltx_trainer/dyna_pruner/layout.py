"""Scope notes for GOPEX integration."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Mask generator uses temporal-variance proxy, not the paper's learned CNN head.",
    "No SimVP / ConvLSTM / TAU backbone training or Jetson latency measurement in-repo.",
    "DiT block skip maps spatial importance to block indices — not per-filter conv group pruning.",
    "WeatherBench / SEVIR table anchors remain reference numbers; upstream repos for full eval.",
)
