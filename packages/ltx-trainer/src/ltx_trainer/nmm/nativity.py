"""Cross-modal fusion nativity (Sec. 2.1)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class FusionRegime(str, Enum):
    """Architectural nativity by integration depth."""

    LATE = "late-fusion"
    MID = "mid-fusion"
    EARLY = "early-fusion"


def fusion_formulas() -> dict[str, str]:
    """Paper formalization of late / mid / early fusion."""
    return {
        "late": "F_late = G_LLM({ P_i(E_i(m_i)) }_i)",
        "mid": "F_mid = Backbone(C(E_1(m_1), ..., E_n(m_n)))",
        "early": "F_early = Transformer(⊕_i T(m_i))",
    }


def describe_fusion(regime: FusionRegime) -> dict[str, Any]:
    """Short industrial summary per fusion regime."""
    cards: dict[FusionRegime, dict[str, Any]] = {
        FusionRegime.LATE: {
            "native": False,
            "summary": "Modular encoders + frozen LLM + grafted output head",
            "training_signature": "Degenerate: single LR, text-only CE, encoder frozen",
        },
        FusionRegime.MID: {
            "native": True,
            "summary": "Joint multimodal backbone with modality-aware boundaries",
            "training_signature": "Progressive unfreeze, differential LR, decoupled losses",
        },
        FusionRegime.EARLY: {
            "native": True,
            "summary": "Born-native unified embedding space / single transformer",
            "training_signature": "Joint-from-start, unified NTP, z-loss + QK-Norm, modality mixture",
        },
    }
    key = regime.value.split("-")[0]
    return {"regime": regime.value, "formula": fusion_formulas()[key], **cards[regime]}
