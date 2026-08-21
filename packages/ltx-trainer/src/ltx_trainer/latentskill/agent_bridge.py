"""Bridge LatentSkill to GOPEX role_agent + vLLM messenger stack."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.inference import inference_notes
from ltx_trainer.latentskill.messenger_bridge import messenger_integration_plan, messenger_status
from ltx_trainer.latentskill.rollout import rollout_demo
from ltx_trainer.latentskill.skills import skill_library_manifest


def role_agent_integration_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    """How LatentSkill complements Role-Agent (arXiv:2606.10917) rollouts."""
    cfg = cfg or LatentSkillConfig()
    return {
        "pairing": "Role-Agent WIA/AIW training + LatentSkill skill encoding at inference",
        "envs": ["role_agent.envs.replay (ALFWorld/WebShop fixtures)", "role_agent.envs.search_qa_env"],
        "skill_matching": skill_library_manifest(),
        "latent_mode": {
            "prompt": "history + admissible actions only",
            "adapter": "mount compiled LoRA from SkillCompiler cache",
            "alpha": cfg.default_injection_alpha,
        },
        "in_context_baseline": {
            "prompt": "skill markdown + history",
            "overhead_k_per_step": "~1.1–1.2 (paper Table 1–2)",
        },
        "smoke": "./scripts/gopex-latentskill.sh rollout",
    }


def vllm_lora_sidecar_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "compile": "latentskill_compile tool → deterministic stub LoRA (replace with trained G_phi)",
        "export": "safetensors per skill_id under $WORK_ROOT/adapters/latentskill/",
        "serve": "vLLM --enable-lora --lora-modules skill_pick=..., skill_look=...",
        "select": "match_alfworld_skill(task_type) or match_search_qa_skill(dataset)",
        "notes": inference_notes(cfg),
    }


def bridge_status(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    demo = rollout_demo(cfg)
    return {
        "role_agent": role_agent_integration_plan(cfg),
        "vllm": vllm_lora_sidecar_plan(cfg),
        "messenger": messenger_integration_plan(cfg),
        "messenger_status": messenger_status(cfg),
        "rollout_smoke": demo,
        "github_upstream": cfg.github,
    }
