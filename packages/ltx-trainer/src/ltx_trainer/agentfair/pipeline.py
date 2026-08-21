"""AgentFAIR evaluation pipeline (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.agentfair.agents import EvaluationResult, evaluate_all
from ltx_trainer.agentfair.config import DIMENSION_MEMBERS, AgentFAIRConfig, SUB_PRINCIPLES
from ltx_trainer.agentfair.critic import CriticDecision, consistency_flags, critic_review
from ltx_trainer.agentfair.extract import demo_records, extract_metadata


def dimension_score(results: dict[str, EvaluationResult], dimension: str) -> float:
    """Score(d) = mean(s_p)/3 * 100 for sub-principles in dimension."""
    members = DIMENSION_MEMBERS[dimension]
    total = sum(results[p].score for p in members)
    return (total / (3.0 * len(members))) * 100.0


def overall_score(results: dict[str, EvaluationResult]) -> float:
    total = sum(results[p].score for p in SUB_PRINCIPLES)
    return (total / (3.0 * len(SUB_PRINCIPLES))) * 100.0


@dataclass
class AgentFAIRReport:
    url: str
    repository: str
    results: dict[str, EvaluationResult]
    dimension_scores: dict[str, float]
    fair_score: float
    critic_decisions: list[CriticDecision] = field(default_factory=list)
    consistency: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "repository": self.repository,
            "scores": {k: v.as_dict() for k, v in self.results.items()},
            "dimension_scores": dict(self.dimension_scores),
            "fair_score": self.fair_score,
            "critic_retries": sum(1 for d in self.critic_decisions if d.retry),
            "consistency_flags": list(self.consistency),
            "recommendations": list(self.recommendations),
        }


def generate_recommendations(results: dict[str, EvaluationResult]) -> list[str]:
    out: list[str] = []
    for p in SUB_PRINCIPLES:
        for rec in results[p].recommendations:
            if rec not in out:
                out.append(rec)
    return out


def evaluate_dataset(
    url: str,
    raw: dict[str, Any] | None = None,
    *,
    repository: str = "",
    config: AgentFAIRConfig | None = None,
) -> AgentFAIRReport:
    """Run Stages 1–3 for one landing page (CPU stub, no network)."""
    cfg = config or AgentFAIRConfig()
    meta = extract_metadata(url, raw, repository=repository)
    results = evaluate_all(meta)
    decisions: list[CriticDecision] = []

    if cfg.enable_critic:
        # First pass consistency context
        for p in SUB_PRINCIPLES:
            decision = critic_review(p, results[p], meta, config=cfg, all_results=results)
            decisions.append(decision)
            if decision.retry and decision.revised is not None:
                results[p] = decision.revised

    dims = {d: dimension_score(results, d) for d in DIMENSION_MEMBERS}
    return AgentFAIRReport(
        url=url,
        repository=str(meta.get("repository") or repository or ""),
        results=results,
        dimension_scores=dims,
        fair_score=overall_score(results),
        critic_decisions=decisions,
        consistency=consistency_flags(results),
        recommendations=generate_recommendations(results),
    )


def evaluate_demo_suite(config: AgentFAIRConfig | None = None) -> list[AgentFAIRReport]:
    cfg = config or AgentFAIRConfig()
    reports = []
    for meta in demo_records():
        reports.append(
            evaluate_dataset(
                meta.url,
                meta.fields,
                repository=str(meta.get("repository") or ""),
                config=cfg,
            )
        )
    return reports


def evaluation_demo() -> dict[str, Any]:
    reports = evaluate_demo_suite()
    return {
        "n": len(reports),
        "datasets": [
            {
                "repository": r.repository,
                "fair_score": round(r.fair_score, 1),
                "dimension_scores": {k: round(v, 1) for k, v in r.dimension_scores.items()},
                "critic_retries": sum(1 for d in r.critic_decisions if d.retry),
            }
            for r in reports
        ],
        "mean_fair": round(sum(r.fair_score for r in reports) / len(reports), 1),
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.agentfair.benchmarks import PAPER_ANCHORS
    from ltx_trainer.agentfair.rubric import assert_rubric_complete

    assert_rubric_complete()
    reports = evaluate_demo_suite(AgentFAIRConfig(enable_critic=True))
    no_critic = evaluate_demo_suite(AgentFAIRConfig(enable_critic=False))
    zenodo = next(r for r in reports if r.repository == "Zenodo")
    aurin = next(r for r in reports if r.repository == "AURIN")
    checks = {
        "rubric_13": len(SUB_PRINCIPLES) == 13,
        "demo_n3": len(reports) == 3,
        "zenodo_above_aurin": zenodo.fair_score > aurin.fair_score,
        "interoperability_weakest_on_aurin": aurin.dimension_scores["I"]
        <= min(aurin.dimension_scores[d] for d in ("F", "A", "R")),
        "critic_can_trigger": any(d.retry for r in reports for d in r.critic_decisions),
        "no_critic_runs": len(no_critic) == 3,
        "paper_findability_anchor": abs(float(PAPER_ANCHORS["findability_mean"]) - 79.7) < 0.05,
        "paper_interop_anchor": abs(float(PAPER_ANCHORS["interoperability_mean"]) - 45.3) < 0.05,
        "paper_cost_anchor": abs(float(PAPER_ANCHORS["mean_cost_usd"]) - 0.054) < 1e-6,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_fix() -> dict[str, bool]:
    return evaluation_smoke()
