"""Skill composition modes (§4.5, Table 3)."""
from __future__ import annotations

from enum import Enum
from typing import Any

from ltx_trainer.latentskill.benchmarks import table3_composition_look
from ltx_trainer.latentskill.compiler import SkillCompiler
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.lora import LatentSkillAdapter, compose_adapters, compile_skill_lora
from ltx_trainer.latentskill.skills import decompose_skill_components, skill_document


class CompositionMode(str, Enum):
    LOOK_ONLY = "look_only"
    PICK_ONLY = "pick_only"
    DIRECT_MERGE = "direct_merge"
    TEXT_MERGE = "text_merge"
    COMPONENT_MERGE = "component_merge"


def compile_skill_by_name(name: str, cfg: LatentSkillConfig, compiler: SkillCompiler | None = None) -> LatentSkillAdapter:
    compiler = compiler or SkillCompiler(cfg)
    return compiler.compile(skill_document(name), skill_id=name.replace(" ", "_").lower())


def compose_skills(
    mode: CompositionMode,
    *,
    cfg: LatentSkillConfig | None = None,
    alpha_look: float = 1.0,
    alpha_pick: float = 1.0,
) -> LatentSkillAdapter:
    cfg = cfg or LatentSkillConfig()
    look_name = "Look At Obj In Light Skill"
    pick_name = "Pick And Place Skill"
    if mode == CompositionMode.LOOK_ONLY:
        return compile_skill_by_name(look_name, cfg)
    if mode == CompositionMode.PICK_ONLY:
        return compile_skill_by_name(pick_name, cfg)
    if mode == CompositionMode.DIRECT_MERGE:
        look = compile_skill_by_name(look_name, cfg)
        pick = compile_skill_by_name(pick_name, cfg)
        return compose_adapters([look, pick], [alpha_look, alpha_pick])
    if mode == CompositionMode.TEXT_MERGE:
        merged_text = skill_document(look_name) + "\n" + skill_document(pick_name)
        return compile_skill_lora(merged_text, "text_merged", cfg)
    # COMPONENT_MERGE: shared components once + task-specific
    shared = decompose_skill_components(look_name)
    look_spec = [c for c in decompose_skill_components(look_name) if "look" in c or "lamp" in c]
    pick_spec = [c for c in decompose_skill_components(pick_name) if "search" in c]
    parts = shared[:2] + look_spec + pick_spec
    adapters = [compile_skill_lora(p, f"comp_{i}", cfg) for i, p in enumerate(parts)]
    return compose_adapters(adapters, [1.0] * len(adapters))


def composition_eval_report(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    modes = {}
    for mode in CompositionMode:
        adapter = compose_skills(mode, cfg=cfg)
        modes[mode.value] = {"n_updates": len(adapter.updates), "skill_id": adapter.skill_id}
    tab = table3_composition_look()
    return {
        "modes": modes,
        "table3": tab,
        "best_seen": max(tab.items(), key=lambda kv: kv[1]["seen"])[0],
        "best_unseen": max(tab.items(), key=lambda kv: kv[1]["unseen"])[0],
    }
