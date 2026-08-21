"""Joint LatentSkill + Role-Agent evaluation stack."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.ablation import run_ablation_smoke
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.integration import joint_role_agent_plan
from ltx_trainer.latentskill.rollout import rollout_demo


def joint_evaluation_demo(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    roll = rollout_demo(cfg)
    ablation = run_ablation_smoke(cfg)
    role_agent_ok = False
    role_agent_dual: dict[str, Any] = {}
    try:
        from ltx_trainer.role_agent.examples import demo_dual_role_loop

        role_agent_dual = demo_dual_role_loop()
        role_agent_ok = bool(role_agent_dual.get("ok"))
    except Exception as exc:  # noqa: BLE001
        role_agent_dual = {"error": str(exc)}
    alf = roll.get("alfworld_replay", {})
    search = roll.get("search_qa", {})
    return {
        "latentskill_rollout": roll,
        "latentskill_ablation": ablation,
        "role_agent_dual_role": role_agent_dual,
        "joint_plan": joint_role_agent_plan(cfg),
        "all_ok": (
            ablation.get("ok")
            and isinstance(alf, dict)
            and alf.get("latent", {}).get("success")
            and isinstance(search, dict)
            and search.get("latent", {}).get("success")
            and role_agent_ok
        ),
    }


def joint_stack_status(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    demo = joint_evaluation_demo(cfg)
    env_probe: dict[str, Any] = {}
    try:
        from ltx_trainer.role_agent.envs.registry import probe_all_envs

        env_probe = probe_all_envs()
    except Exception as exc:  # noqa: BLE001
        env_probe = {"error": str(exc)}
    return {
        "demo": demo,
        "env_probe": env_probe,
        "alpha": cfg.default_injection_alpha,
        "backbone_latentskill": cfg.backbone,
    }
