"""GOPEX stack integration for LatentSkill (arXiv:2606.06087)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig


def gopex_stack_card(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "latentskill_role": (
            "Hypernetwork G_phi maps textual skills to cached LoRA adapters; "
            "zero skill tokens at inference with α-scaled injection on attn_o + mlp_down."
        ),
        "scripts": {
            "gopex": "./scripts/gopex-latentskill.sh",
            "role_agent": "./scripts/gopex-role-agent.sh",
        },
        "backbone": cfg.backbone,
        "benchmarks": ["ALFWorld", "Search-QA"],
        "export_root": "WORK_ROOT/adapters/latentskill/",
    }


def joint_role_agent_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    """LatentSkill inference + Role-Agent WIA/AIW training stack."""
    cfg = cfg or LatentSkillConfig()
    return {
        "pairing": "Role-Agent trains dual-role policy; LatentSkill encodes skills as in-weight adapters at serve time",
        "train": {
            "role_agent": "./scripts/gopex-role-agent.sh train-toy",
            "role_agent_search": "./scripts/gopex-role-agent.sh train-search-qa",
            "latentskill_pretrain": "upstream G_phi document pretrain (171K skills)",
            "latentskill_sft": "trajectory SFT on SkillRL teacher traces",
        },
        "inference": {
            "latent_skills": "latentskill_compile → vLLM LoRA module per task type",
            "alpha": cfg.default_injection_alpha,
            "in_context_baseline": "skill markdown in prompt (Table 1–2 cost anchor)",
        },
        "env": {
            "GOPEX_LATENTSKILL_ALPHA": str(cfg.default_injection_alpha),
            "GOPEX_LATENTSKILL_MODE": "latent",
            "GOPEX_ROLE_AGENT_ENV": "replay",
        },
        "smoke": [
            "./scripts/gopex-latentskill.sh smoke",
            "./scripts/gopex-role-agent.sh smoke",
            "./scripts/gopex-latentskill.sh joint",
        ],
    }


def native_evolve_plan() -> dict[str, Any]:
    """Suggested GOPEX steps when pairing with agent stack (no LTX video fold)."""
    return {
        "preprocess": [
            {"id": "export_skills", "command": "./scripts/gopex-latentskill.sh export"},
            {"id": "role_agent_env", "command": "./scripts/gopex-role-agent.sh env-probe"},
        ],
        "serve": [
            {"id": "vllm_lora", "command": "./scripts/gopex-latentskill.sh messenger"},
            {"id": "llm_rollout", "command": "./scripts/gopex-role-agent.sh llm-rollout"},
        ],
        "eval": [
            {"id": "latentskill_eval", "command": "./scripts/gopex-latentskill.sh eval"},
            {"id": "latentskill_ablation", "command": "./scripts/gopex-latentskill.sh ablation"},
            {"id": "joint_demo", "command": "./scripts/gopex-latentskill.sh joint"},
        ],
        "env": 'eval "$(./scripts/gopex-latentskill.sh env)"',
    }
