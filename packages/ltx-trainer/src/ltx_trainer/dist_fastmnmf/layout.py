"""Distributed FastMNMF stub limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — no full FastMNMF IP/MM iteration loop or Pyroomacoustics simulation in-repo.",
    "Centralized algorithm (all subarray observations); not decentralized IVA with communication constraints.",
    "Block-diagonal SCM is a computational approximation; inter-subarray phase discarded.",
    "Experiments assume synchronized, noiseless conditions; calibration/async errors not evaluated.",
)
