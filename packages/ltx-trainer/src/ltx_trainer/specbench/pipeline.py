"""SpecBench pipeline cards and demos (arXiv:2605.30314)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.specbench.benchmark import (
    benchmark_card,
    evaluate_agent,
    evaluate_agent_on_task,
    paper_agent_leaderboard,
)
from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.mock import evaluation_smoke
from ltx_trainer.specbench.scoring import aggregate_accuracy, score_task
from ltx_trainer.specbench.judging import MatchPair
from ltx_trainer.specbench.tasks import all_tasks, codex54_scoring_predictions, paper_table5_match_pairs, task_by_id
from ltx_trainer.specbench.taxonomy import REPOSITORY_LABELS


def framework_card() -> dict[str, Any]:
    cfg = SpecBenchConfig()
    return {
        "name": "SpecBench",
        "arxiv": cfg.paper_arxiv,
        "github": cfg.github_url,
        "task": "Identify specification deficiencies in RFC design proposals",
        "inputs": ["initial RFC proposal", "codebase snapshot", "prior RFC history"],
        "outputs": ["ranked deficiency predictions (SPI-decomposed)"],
        "deficiency_classes": ["omission", "ambiguous", "inconsistent", "incorrect"],
        "repositories": {k: v for k, v in REPOSITORY_LABELS.items()},
        "scoring": {
            "prediction_budget": f"ceil({cfg.prediction_budget_multiplier} × |G|)",
            "core_weight": cfg.core_weight,
            "extended_weight": cfg.extended_weight,
            "matching": "SPI subject+predicate ensemble judge (4 trials, majority 3/4)",
        },
        "paper_best": {"agent": cfg.paper_best_agent, "accuracy": cfg.paper_best_accuracy},
    }


def knowledge_card() -> dict[str, Any]:
    cfg = SpecBenchConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": "SpecBench: Evaluating Specification-Level Reasoning for Software Engineering LLM Agents",
        "authors": "Hamblin, Song, Zhu, Jayarajan, Liu, Vijaykumar, Pekhimenko",
        "institutions": ["University of Toronto", "Vector Institute", "University of Waterloo", "NVIDIA"],
        "motivation": (
            "SWE-Bench assumes perfect specifications; real systems need expert RFC review "
            "before implementation."
        ),
        "capability": "specification-level reasoning (complete, unambiguous, consistent, correct specs)",
        "version_note": "This stub covers deficiency identification; revision tasks are future work.",
        "leaderboard": paper_agent_leaderboard(cfg),
    }


def benchmarks_bundle() -> dict[str, Any]:
    tasks = all_tasks()
    return {
        "table1_kep4671_gold_count": 15,
        "table1_core_extended_split": "11 core / 4 extended",
        "table6_kep4671_weighted_example": {
            "core_matches": "7/11",
            "extended_matches": "0/4",
            "weighted_score": 7.0,
            "max_weighted_score": 13.0,
            "accuracy": 0.5385,
        },
        "figure3_best_agent": {"name": "codex-5.4", "accuracy": 0.444},
        "tasks": [
            {
                "task_id": t.task_id,
                "repository": t.repository,
                "rfc_id": t.rfc_id,
                "title": t.title,
                "n_gold": len(t.gold),
            }
            for t in tasks
        ],
        "agent_results": paper_agent_leaderboard(),
    }


def evaluation_demo(*, task_id: str = "kubernetes_kep4671") -> dict[str, Any]:
    cfg = SpecBenchConfig()
    task = task_by_id(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    preds, pred_spis = codex54_scoring_predictions() if task_id == "kubernetes_kep4671" else (
        [g.text for g in task.gold[:2]],
        None,
    )
    forced = (
        [MatchPair(pred_index=p, gold_id=g) for p, g in paper_table5_match_pairs()]
        if task_id == "kubernetes_kep4671"
        else None
    )
    score = score_task(task, preds, cfg, pred_spis=pred_spis, forced_pairs=forced)
    return {
        "task_id": task_id,
        "repository": task.repository,
        "rfc_id": task.rfc_id,
        "title": task.title,
        "n_predictions": len(preds),
        "score": score.to_dict(),
        "sample_predictions": preds[:3],
        "leaderboard": paper_agent_leaderboard(cfg)[:3],
    }


def full_benchmark_demo() -> dict[str, Any]:
    cfg = SpecBenchConfig()
    preds_map = {
        t.task_id: (
            codex54_scoring_predictions()[0]
            if t.task_id == "kubernetes_kep4671"
            else [g.text for g in t.gold]
        )
        for t in all_tasks()
    }
    # score_task uses heuristic SPI unless pred_spis passed; full eval uses per-task scoring
    per_task = []
    for t in all_tasks():
        if t.task_id == "kubernetes_kep4671":
            texts, spis = codex54_scoring_predictions()
            forced = [MatchPair(pred_index=p, gold_id=g) for p, g in paper_table5_match_pairs()]
            per_task.append(score_task(t, texts, cfg, pred_spis=spis, forced_pairs=forced))
        else:
            per_task.append(score_task(t, [g.text for g in t.gold], cfg))
    overall = aggregate_accuracy(per_task)
    by_repo: dict[str, list[float]] = {}
    for s in per_task:
        by_repo.setdefault(s.repository, []).append(s.accuracy)
    repo_mean = {k: sum(v) / len(v) for k, v in by_repo.items()}
    return {
        "overall_accuracy": round(overall, 4),
        "per_task": [s.to_dict() for s in per_task],
        "by_repository": {k: round(v, 4) for k, v in repo_mean.items()},
    }
