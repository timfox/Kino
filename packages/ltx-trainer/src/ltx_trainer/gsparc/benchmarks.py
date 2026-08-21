"""Reference metrics (Nukapotula et al., arXiv:2511.22793)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gsparc.config import (
    CONFIDENCE_THRESHOLD,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PILOT_FREE_FRACTION,
)
from ltx_trainer.gsparc.downstream import downstream_card, mcs_throughput_anchor

# Table 1 — Sionna conference room (representative)
SIONNA_CONFERENCE_GSRF = {
    "train_hrs": 1.22,
    "render_ms": 23.0,
    "ssim": 0.70,
    "psnr_db": 25.76,
    "confidence": False,
}

SIONNA_CONFERENCE_GSPARC = {
    "train_hrs": 1.17,
    "render_ms": 3.8,
    "ssim": 0.78,
    "psnr_db": 27.16,
    "confidence": True,
}

# Table 3 — Sionna conference room full comparison (means)
SIONNA_TABLE3 = {
    "NeRF2": {"ssim_mean": 0.69, "mse_mean": 0.00513, "train_hrs": 4.15},
    "WRFGS+": {"ssim_mean": 0.71, "mse_mean": 0.00550, "train_hrs": 3.67},
    "GSRF": {"ssim_mean": 0.70, "mse_mean": 0.00512, "train_hrs": 1.1},
    "GSpaRC": {"ssim_mean": 0.78, "mse_mean": 0.00417, "train_hrs": 1.0},
}

# Table 4 — RFID
RFID_TABLE4 = {
    "WRFGS+": {"ssim": 0.78, "render_ms": 8.0},
    "NeRF2": {"ssim": 0.75, "render_ms": 230.0},
    "GSRF": {"ssim": 0.82, "mse": 0.0094, "render_ms": 23.0},
    "GSpaRC": {"ssim": 0.82, "mse": 0.0134, "render_ms": 0.8},
}

# Table 5 — Argos CSI
ARGOS_TABLE5 = {
    "GSRF": {"snr_1sc_db": 23.62, "snr_26sc_db": 20.87, "nmse": 0.0247, "render_ms": 22.76},
    "GSpaRC": {"snr_1sc_db": 23.52, "snr_26sc_db": 19.59, "nmse": 0.0110, "render_ms": 3.86},
}

ARGOS_CONFIDENCE = {
    "threshold": CONFIDENCE_THRESHOLD,
    "pilot_free_fraction": PILOT_FREE_FRACTION,
    "test_positions": 800,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_sionna": {
            "GSRF": SIONNA_CONFERENCE_GSRF,
            "GSpaRC": SIONNA_CONFERENCE_GSPARC,
        },
        "table3_sionna_conference": SIONNA_TABLE3,
        "table4_rfid": RFID_TABLE4,
        "table5_argos": ARGOS_TABLE5,
        "argos_confidence": ARGOS_CONFIDENCE,
        "downstream_sec6": downstream_card(),
        "mcs_throughput": mcs_throughput_anchor(),
    }
