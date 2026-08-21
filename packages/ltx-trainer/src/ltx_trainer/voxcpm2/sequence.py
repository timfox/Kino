"""Unified sequence organization (Table 2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voxcpm2.config import GenerationMode, Voxcpm2Config

SEQUENCE_LAYOUTS: dict[GenerationMode, str] = {
    "basic_tts": "⟨text⟩ → ⟨target audio⟩",
    "voice_design": "⟨(voice description) text⟩ → ⟨target audio⟩",
    "reference_cloning": "⟨reference audio⟩ | ⟨text⟩ → ⟨target audio⟩",
    "controllable_cloning": "⟨reference audio⟩ | ⟨(style description) text⟩ → ⟨target audio⟩",
    "continuation_cloning": "⟨prompt text + target text⟩ | ⟨prompt audio⟩ → ⟨target audio⟩",
}


def sequence_table() -> list[dict[str, Any]]:
    return [
        {"mode": mode, "layout": layout}
        for mode, layout in SEQUENCE_LAYOUTS.items()
    ]


def format_voice_design_text(description: str, text: str) -> str:
    desc = description.strip()
    if not desc.startswith("("):
        desc = f"({desc})"
    return f"{desc}{text}"


def sequence_demo(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    example = format_voice_design_text(
        "A young woman, gentle and sweet voice",
        "Hello, welcome to VoxCPM2!",
    )
    return {
        "modes": list(c.generation_modes),
        "layouts": sequence_table(),
        "voice_design_example": example,
        "ref_markers": ("REF_START", "REF_END"),
    }
