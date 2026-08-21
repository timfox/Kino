"""Downstream PHY metrics (Sec. 6)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.gsparc.config import CONFIDENCE_THRESHOLD, PILOT_FREE_FRACTION


def pilot_free_coverage(
    confidence_normalized: Tensor,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Fraction of positions with C̃ ≥ τ (Fig. 2)."""
    mask = confidence_normalized >= threshold
    frac = float(mask.float().mean())
    return {
        "threshold": threshold,
        "pilot_free_fraction": frac,
        "pilot_required_fraction": 1.0 - frac,
        "count": int(mask.sum()),
        "total": int(mask.numel()),
    }


def mcs_throughput_anchor() -> dict[str, float]:
    """Fig. 7(b) effective throughput (Mbps)."""
    return {
        "gt_csi_mbps": 39.7,
        "gsparc_mbps": 39.3,
        "gap_mbps": 0.36,
        "gap_percent": 0.9,
    }


def downstream_card() -> dict[str, Any]:
    return {
        "link_adaptation": {
            **mcs_throughput_anchor(),
            "target_bler": 0.1,
            "nsc": 624,
            "fs_symbols_per_s": 28000,
        },
        "communication": {
            "simulator": "Sionna 5G NR PUSCH",
            "confidence_filter_tau": CONFIDENCE_THRESHOLD,
            "pilot_free_fraction_paper": PILOT_FREE_FRACTION,
            "note": "BER/BLER with C≥τ tracks perfect CSI; unfiltered degrades",
        },
        "argos_mcs_errors": {
            "error_0": 412,
            "error_2": 383,
            "error_gt_2": 5,
        },
    }


def filter_by_confidence(
    values: Tensor,
    confidence_normalized: Tensor,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Tensor:
    """Keep samples with C̃ ≥ τ for confidence-aware PHY."""
    return values[confidence_normalized >= threshold]
