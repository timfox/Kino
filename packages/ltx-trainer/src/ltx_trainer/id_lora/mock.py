"""CPU smoke gates for ID-LoRA integration."""

from __future__ import annotations

from typing import Any

from ltx_trainer.id_lora.pipeline import evaluation_demo, knowledge_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    kb = knowledge_card()
    ok = (
        demo["parsed"]["identity_ready"]
        and demo["inference_plan"]["script"].endswith(".py")
        and kb["training"]["strategy"] == "audio_ref_only_ic"
        and kb["framework"]["training_strategy"] == "audio_ref_only_ic"
    )
    return {
        "status": "ok" if ok else "fail",
        "demo": demo,
        "knowledge_keys": sorted(kb.keys()),
    }
