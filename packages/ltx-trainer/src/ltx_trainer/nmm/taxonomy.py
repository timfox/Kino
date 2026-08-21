"""Input–output duality taxonomy (Sec. 2.2)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class IOCategory(str, Enum):
    """Multi-to-{Text, Target, Multi} functional paradigms."""

    M2T = "multi-to-text"
    M2G = "multi-to-target"
    M2M = "multi-to-multi"


def io_formulas() -> dict[str, str]:
    return {
        "M2T": "F_M2T : M → T",
        "M2G": "F_M2G : M → y_k",
        "M2M": "F_M2M : M_in → M_out",
    }


def describe_io(category: IOCategory) -> dict[str, Any]:
    """Capability focus per IO paradigm."""
    cards: dict[IOCategory, dict[str, Any]] = {
        IOCategory.M2T: {
            "output": "text only",
            "focus": "cross-modal comprehension and reasoning",
            "examples": ["Qwen3-VL", "Kimi K2.5", "InternVL-3.5", "MiniCPM-V-4.6"],
        },
        IOCategory.M2G: {
            "output": "single target modality (image / audio / video)",
            "focus": "scenario-oriented native generation",
            "examples": ["LTX-2.3", "Wan2.2", "Seedream3.0", "OmniVoice", "Qwen3-Omni"],
        },
        IOCategory.M2M: {
            "output": "arbitrary multimodal combinations",
            "focus": "symmetric understanding + generation in one network",
            "examples": ["BAGEL", "Emu3.5", "Show-o2", "Transfusion", "Chameleon", "AnyGPT"],
        },
    }
    return {"category": category.value, **cards[category], "formula": io_formulas()[category.name]}
