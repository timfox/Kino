"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.qwen3_tts.attention import detect_best_attention, resolve_attention
from ltx_trainer.qwen3_tts.config import Qwen3TtsConfig
from ltx_trainer.qwen3_tts.dialogue import RoleBank, parse_dialogue_script, validate_script_roles
from ltx_trainer.qwen3_tts.pipeline import evaluation_demo, framework_card, speaker_table
from ltx_trainer.qwen3_tts.upstream import build_infer_argv, doctor, install_plan


def evaluation_smoke(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    card = framework_card(c)
    assert card["framework"] == "Qwen3-TTS"
    assert len(speaker_table(c)) == len(c.speakers)

    script = "Alice: Hello there.\nBob: Hi Alice!"
    lines = parse_dialogue_script(script)
    assert len(lines) == 2
    bank = RoleBank(roles={"Alice": {"stub": 1}, "Bob": {"stub": 2}})
    assert validate_script_roles(script, bank) == []

    _, attn_impl, _ = resolve_attention("auto")
    assert attn_impl in ("flash_attention_2", "sdpa", "eager")

    plan = build_infer_argv(mode="custom_voice", text="Hello", speaker="Ryan")
    assert plan["mode"] == "custom_voice"
    assert "--model" in plan["command"]

    demo = evaluation_demo(seed=0, cfg=c)
    assert demo["speakers"][0]["speaker"] == c.speakers[0]

    doc = doctor(c)
    assert "issues" in doc
    assert install_plan(c)["transformers_warning"]

    return {
        "status": "ok",
        "framework": card["framework"],
        "speakers": len(c.speakers),
        "languages": len(c.languages),
        "auto_attention": detect_best_attention(),
        "transformers_pin": c.transformers_pin,
    }
