"""Token efficiency vs in-context skills (§4.2)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.inference import AgentTurnContext
from ltx_trainer.latentskill.skills import skill_document


def estimate_skill_tokens(skill_name: str, *, tokens_per_word: float = 1.3) -> float:
    doc = skill_document(skill_name)
    return len(doc.split()) * tokens_per_word


def compare_modes_on_history(
    history: str,
    skill_name: str,
    *,
    cfg: LatentSkillConfig | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    latent_k = AgentTurnContext(history, skill_name, mode="latent", injection_alpha=cfg.default_injection_alpha)
    inctx_k = AgentTurnContext(history, skill_name, mode="in_context")
    latent_per_step = latent_k.estimate_prefill_tokens() / 1000.0
    inctx_per_step = inctx_k.estimate_prefill_tokens() / 1000.0
    skill_only = estimate_skill_tokens(skill_name) / 1000.0
    reduction = 1.0 - (latent_per_step / inctx_per_step if inctx_per_step else 1.0)
    return {
        "latent_prefill_k_per_step": round(latent_per_step, 3),
        "in_context_prefill_k_per_step": round(inctx_per_step, 3),
        "skill_text_k": round(skill_only, 3),
        "relative_reduction": round(reduction, 3),
        "steps": steps,
        "paper_alfworld_reduction": cfg.prefill_reduction_alfworld,
        "paper_search_qa_reduction": cfg.prefill_reduction_search_qa,
    }


def efficiency_demo(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    sample_history = "You are in the kitchen. You see apple 1 on counter 1 and fridge 1."
    alf = compare_modes_on_history(sample_history, "Clean Skill", cfg=cfg)
    search = compare_modes_on_history("Question: capital of France?", "direct_retrieval", cfg=cfg)
    return {"alfworld_clean": alf, "search_qa_direct": search}
