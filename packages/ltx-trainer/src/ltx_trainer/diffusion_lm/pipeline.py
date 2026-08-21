"""Framework card, evaluation demo, smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.diffusion_lm.benchmarks import (
    agentboard_excerpt,
    bfcl_excerpt,
    paper_anchors,
    search_agent_latency,
)
from ltx_trainer.diffusion_lm.config import DiffusionLmPaperConfig
from ltx_trainer.diffusion_lm.integration import integration_bundle
from ltx_trainer.diffusion_lm.taxonomy import taxonomy_bundle


def framework_card(cfg: DiffusionLmPaperConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DiffusionLmPaperConfig()
    tax = taxonomy_bundle()
    return {
        "papers": paper_anchors(),
        "config": cfg.__dict__,
        "taxonomy": tax,
        "integration": integration_bundle(),
        "headline_guidance": tax["deployment_ladder"][1]["mode"],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "agentboard": agentboard_excerpt(),
        "bfcl": bfcl_excerpt(),
        "search_latency": search_agent_latency(),
        "papers": paper_anchors(),
        "framework": framework_card(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    from gopex_diffusion_lm.decode import synthetic_decode_demo
    from gopex_diffusion_lm.engine import DiffuAgentEngine
    from gopex_diffusion_lm.p_react import plan_p_react

    engine = DiffuAgentEngine()
    plan = engine.plan_turn(
        user_message="Plan a multi-tool research answer with web search.",
        history_chars=8000,
        expects_tools=True,
        expects_slow_tool=True,
    )
    return {
        "hybrid_plan": plan.to_dict(),
        "decode_round": synthetic_decode_demo(seed=seed),
        "p_react": plan_p_react(expects_slow_tool=True).to_dict(),
        "framework": framework_card(),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    plan = demo["hybrid_plan"]
    return {
        "paper": "diffusion_lm",
        "profile": plan["profile"],
        "stage_count": len(plan["stages"]),
        "has_backbone": any(s["role"] == "backbone_ar" for s in plan["stages"]),
        "p_react_gain": demo["p_react"]["estimated_latency_gain"],
        "decode_commits": len(demo["decode_round"]["committed_positions"]),
        "papers": paper_anchors(),
    }
