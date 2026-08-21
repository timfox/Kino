"""GOPEX agent-stack profile env for LatentSkill + Role-Agent (no LTX video fold)."""
from __future__ import annotations

import shlex
from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.export import default_export_root
from ltx_trainer.latentskill.messenger_bridge import messenger_env_plan


def agent_skills_profile(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    messenger = messenger_env_plan(cfg)
    return {
        "name": "agent_skills",
        "description": (
            "LLM agent skill encoding: LatentSkill hypernetwork LoRA + Role-Agent WIA/AIW "
            "training on replay/search fixtures (no LTX av_fold hooks)."
        ),
        "hooks": [],
        "env": {
            **messenger,
            "GOPEX_ROLE_AGENT_ENV": "replay",
            "GOPEX_LATENTSKILL_ENABLE": "1",
        },
        "scripts": {
            "latentskill_smoke": "./scripts/gopex-latentskill.sh smoke",
            "latentskill_eval": "./scripts/gopex-latentskill.sh eval",
            "latentskill_joint": "./scripts/gopex-latentskill.sh joint",
            "role_agent_smoke": "./scripts/gopex-role-agent.sh smoke",
            "role_agent_env": "./scripts/gopex-role-agent.sh env-catalog",
        },
        "export_root": str(default_export_root()),
        "backbone": cfg.backbone,
    }


def export_env_shell(cfg: LatentSkillConfig | None = None) -> str:
    profile = agent_skills_profile(cfg)
    return "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in sorted(profile["env"].items()))


def agent_skills_plan() -> list[str]:
    return [
        'eval "$(./scripts/gopex-latentskill.sh env)"',
        "./scripts/gopex-latentskill.sh export",
        "./scripts/gopex-latentskill.sh eval",
        "./scripts/gopex-role-agent.sh env-catalog",
        "./scripts/gopex-latentskill.sh joint",
    ]
