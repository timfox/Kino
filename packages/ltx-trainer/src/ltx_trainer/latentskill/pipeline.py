"""LatentSkill pipeline and evaluation (arXiv:2606.06087)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.analysis import (
    alpha_performance_curve,
    compose_look_pick,
    mds_semantic_geometry,
    token_efficiency_report,
)
from ltx_trainer.latentskill.benchmarks import benchmarks_bundle, table1_alfworld, table2_search_qa
from ltx_trainer.latentskill.compiler import SkillCompiler
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.lora import compile_skill_lora, mount_delta, stable_rank
from ltx_trainer.latentskill.skills import skill_library_manifest
from ltx_trainer.latentskill.ablation import run_ablation_smoke
from ltx_trainer.latentskill.agent_bridge import bridge_status
from ltx_trainer.latentskill.integration import gopex_stack_card, joint_role_agent_plan
from ltx_trainer.latentskill.role_agent_joint import joint_evaluation_demo, joint_stack_status
from ltx_trainer.latentskill.run_plan import run_plan
from ltx_trainer.latentskill.composition import composition_eval_report
from ltx_trainer.latentskill.efficiency import efficiency_demo
from ltx_trainer.latentskill.rollout import rollout_demo
from ltx_trainer.latentskill.sensitivity import sensitivity_suite
from ltx_trainer.latentskill.skills import skill_document
from ltx_trainer.latentskill.eval import evaluate_harness
from ltx_trainer.latentskill.training import training_ladder


def framework_card(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "name": "LatentSkill",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "task": "In-weight latent skills via hypernetwork LoRA for LLM agents",
        "backbone": cfg.backbone,
        "paradigm": "G_phi(skill_text) -> LoRA adapter; zero skill tokens at inference",
        "components": [
            "skill_compiler_hypernetwork (Eq. 1)",
            "document_pretrain L_pre (Eq. 4)",
            "trajectory_sft L_sft (Eq. 5)",
            "adapter_cache + alpha scaling (Eq. 3)",
            "parameter_space_composition (Eq. 6)",
        ],
        "benchmarks": ["ALFWorld", "Search-QA"],
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }


def evaluation_demo(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    compiler = SkillCompiler(cfg)
    skill_text = "Pick And Place Skill: search shelves and drawers systematically."
    adapter = compiler.compile(skill_text, "pick")
    u0 = adapter.updates[0]
    d = u0.a.shape[1]
    base_w = __import__("numpy").random.default_rng(0).standard_normal((d, d)) * 0.01
    mounted = mount_delta(base_w, u0, alpha=cfg.default_injection_alpha, rank=u0.a.shape[0])
    delta = u0.delta(cfg.default_injection_alpha, u0.a.shape[0])
    return {
        "compiled_skill_id": adapter.skill_id,
        "n_lora_updates": len(adapter.updates),
        "mount_shape_ok": mounted.shape == base_w.shape,
        "stable_rank": round(stable_rank(delta), 3),
        "semantic_geometry": mds_semantic_geometry(cfg),
        "composition": compose_look_pick(cfg),
        "token_efficiency": token_efficiency_report(cfg),
        "training_ladder": training_ladder(cfg),
        "skill_library": skill_library_manifest(),
        "composition_modes": composition_eval_report(cfg),
        "sensitivity": sensitivity_suite(skill_document("Clean Skill"), "clean", cfg=cfg),
        "efficiency": efficiency_demo(cfg),
        "rollout": rollout_demo(cfg),
        "bridge": bridge_status(cfg),
        "ablation": run_ablation_smoke(cfg),
        "joint_stack": joint_stack_status(cfg),
        "run_plan": run_plan(cfg),
        "eval_harness": evaluate_harness(cfg),
    }


def paper_checks() -> dict[str, bool]:
    t1 = table1_alfworld()
    t2 = table2_search_qa()
    comp = compose_look_pick()
    sens = sensitivity_suite(skill_document("Clean Skill"), "clean")
    roll = rollout_demo()
    alf = roll.get("alfworld_replay", {})
    latent_ok = isinstance(alf, dict) and alf.get("latent", {}).get("success") is True
    search = roll.get("search_qa", {})
    search_ok = isinstance(search, dict) and search.get("latent", {}).get("success") is True
    return {
        "latentskill_beats_incontext_alfworld_seen": t1["LatentSkill"]["seen_avg"] > t1["In-Context Skill"]["seen_avg"],
        "latentskill_beats_incontext_alfworld_unseen": t1["LatentSkill"]["unseen_avg"] > t1["In-Context Skill"]["unseen_avg"],
        "latentskill_beats_incontext_search": t2["LatentSkill"]["avg"] > t2["In-Context Skill"]["avg"],
        "component_merge_beats_look_only": comp["component_merging_best"],
        "alpha_inverted_u_exists": alpha_performance_curve("seen")[str(0.6)] > alpha_performance_curve("seen")[str(1.2)],
        "prefill_reduction_alfworld": LatentSkillConfig().prefill_reduction_alfworld > 0.6,
        "latent_robust_under_hijack": sens["latent_robust_under_hijack"],
        "replay_rollout_success": latent_ok,
        "search_qa_rollout_success": search_ok,
        "latent_zero_skill_tokens": isinstance(alf, dict) and alf.get("latent_zero_skill_tokens") is True,
        "ablation_smoke_ok": run_ablation_smoke().get("ok"),
        "joint_stack_ok": joint_stack_status().get("demo", {}).get("all_ok"),
        "eval_harness_ok": evaluate_harness().get("ok"),
    }


def knowledge_card(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "framework": framework_card(cfg),
        "gopex_stack": gopex_stack_card(cfg),
        "joint_plan": joint_role_agent_plan(cfg),
        "benchmarks": benchmarks_bundle(),
        "checks": paper_checks(),
    }


def evaluation_smoke(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    demo = evaluation_demo(cfg)
    checks = paper_checks()
    return {
        "package": "latentskill",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": all(checks.values()),
        "checks": checks,
        "compiled_updates": demo["n_lora_updates"],
        "alfworld_gain_seen_pp": benchmarks_bundle()["alfworld_gain_seen_pp"],
    }
