"""Unified LatentSkill eval harness — all replay fixtures × modes (CPU stub)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.rollout import run_replay_rollout
from ltx_trainer.latentskill.skills import match_alfworld_skill, match_search_qa_skill

EvalMode = Literal["latent", "in_context", "vanilla"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    benchmark: str
    skill_name: str
    kind: str  # replay | search_qa


# ALFWorld Clean replay + WebShop + Search-QA Table 6 coverage
EVAL_SUITE: tuple[EvalCase, ...] = (
    EvalCase("alfworld_put_apple", "alfworld", match_alfworld_skill("Clean"), "replay"),
    EvalCase("webshop_mug_blue", "webshop", "direct_retrieval", "replay"),
    EvalCase("search_capital_france", "search_qa", match_search_qa_skill("NQ"), "search_qa"),
    EvalCase("search_python_creator", "search_qa", match_search_qa_skill("TriviaQA"), "search_qa"),
    EvalCase("search_hotpot_bridge", "search_qa", match_search_qa_skill("HotpotQA"), "search_qa"),
)


def _make_env(case: EvalCase):
    if case.kind == "search_qa":
        from ltx_trainer.role_agent.envs.search_qa_env import SearchQAEnv

        return SearchQAEnv.create(case.case_id)
    from ltx_trainer.role_agent.envs.replay import make_replay_env

    return make_replay_env(case.case_id)


def evaluate_case(
    case: EvalCase,
    *,
    mode: EvalMode = "latent",
    cfg: LatentSkillConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    env = _make_env(case)
    out = run_replay_rollout(env, skill_name=case.skill_name, mode=mode, cfg=cfg)
    return {"case_id": case.case_id, "benchmark": case.benchmark, **out}


def evaluate_suite(
    *,
    modes: tuple[EvalMode, ...] = ("latent", "in_context", "vanilla"),
    cfg: LatentSkillConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    rows: list[dict[str, Any]] = []
    for case in EVAL_SUITE:
        for mode in modes:
            try:
                row = evaluate_case(case, mode=mode, cfg=cfg)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "case_id": case.case_id,
                    "benchmark": case.benchmark,
                    "mode": mode,
                    "skill": case.skill_name,
                    "success": False,
                    "error": str(exc),
                }
            rows.append(row)

    def _rate(mode: EvalMode) -> float:
        subset = [r for r in rows if r.get("mode") == mode and "error" not in r]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.get("success")) / len(subset)

    latent_rows = [r for r in rows if r.get("mode") == "latent" and "error" not in r]
    inctx_rows = [r for r in rows if r.get("mode") == "in_context" and "error" not in r]
    token_wins = sum(
        1
        for lat, ic in zip(latent_rows, inctx_rows, strict=False)
        if lat.get("prefill_k", 0) <= ic.get("prefill_k", 0)
    )
    return {
        "n_cases": len(EVAL_SUITE),
        "modes": list(modes),
        "rows": rows,
        "success_rate": {m: round(_rate(m), 3) for m in modes},
        "latent_beats_incontext_tokens": token_wins == len(EVAL_SUITE),
        "all_latent_success": all(r.get("success") for r in latent_rows),
    }


def evaluate_harness(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    """Full CPU eval: fixture sweep + paper anchor cross-check."""
    from ltx_trainer.latentskill.ablation import ablation_table_anchors

    suite = evaluate_suite(cfg=cfg)
    anchors = ablation_table_anchors()
    return {
        "suite": suite,
        "paper_anchors": anchors,
        "ok": suite["all_latent_success"] and suite["latent_beats_incontext_tokens"],
    }
