"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lora_hd_attn.paper import paper_card
from ltx_trainer.lora_hd_attn.pipeline import run_active_ft_demo, run_demo, run_reused_sequences_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    active = run_active_ft_demo()
    reused = run_reused_sequences_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "active_ft": active,
        "reused_sequences": reused,
        "status": "ok"
        if demo["lora_beats_frozen"]
        and demo["delta_eff_decreases_with_alignment"]
        and active["active_improves"]
        and reused["delta_eff_smaller_when_reused"]
        else "fail",
    }
