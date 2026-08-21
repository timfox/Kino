"""Evaluation datasets (Sec. 5)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "sionna_conference": {
            "scene": "14 m × 10 m × 4 m indoor, rich multipath",
            "positions": "5142 train / 1543 test RX (fixed TX)",
            "output": "hemispherical spectrum @ receiver",
            "toolchain": "Sionna ray tracer",
        },
        "rfid_nerf2": {
            "source": "NeRF2 RFID spectrum dataset",
            "tx_locations": 6123,
            "rx": "fixed 4×4 array @ 915 MHz",
            "loss": "L_sp (L1 + SSIM, λ1=0.2)",
        },
        "argos": {
            "description": "64-antenna massive MIMO outdoor @ 2.4 GHz",
            "split": "3200 train / 800 test positions",
            "output": "complex downlink CSI (26 subcarriers)",
            "downstream": ["MCS selection", "5G NR PUSCH BER/BLER"],
        },
        "baselines": ["NeRF2", "NeWRF", "WRFGS+", "GSRF", "FIRE"],
    }
