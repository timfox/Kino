"""SpecBench benchmark catalog and evaluation helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.scoring import TaskScore, aggregate_accuracy, score_task
from ltx_trainer.specbench.tasks import SpecBenchTask, all_tasks, codex54_predictions_kep4671, task_by_id


def benchmark_card() -> dict[str, Any]:
    cfg = SpecBenchConfig()
    tasks = all_tasks()
    return {
        "name": "SpecBench",
        "arxiv": cfg.paper_arxiv,
        "github": cfg.github_url,
        "focus": "specification-level reasoning (deficiency identification)",
        "repositories": list(cfg.repositories),
        "n_tasks": len(tasks),
        "deficiency_classes": ["omission", "ambiguous", "inconsistent", "incorrect"],
        "scoring": "tiered IR with 1.25× prediction budget; core weight 2× extended",
    }


def evaluate_agent_on_task(
    task: SpecBenchTask,
    predictions: list[str],
    cfg: SpecBenchConfig | None = None,
) -> TaskScore:
    return score_task(task, predictions, cfg or SpecBenchConfig())


def evaluate_agent(
    predictions_by_task: dict[str, list[str]],
    cfg: SpecBenchConfig | None = None,
) -> dict[str, Any]:
    c = cfg or SpecBenchConfig()
    per_task: list[TaskScore] = []
    for task in all_tasks():
        preds = predictions_by_task.get(task.task_id, [])
        per_task.append(score_task(task, preds, c))
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


def paper_agent_leaderboard(cfg: SpecBenchConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SpecBenchConfig()
    return [{"agent": name, "overall_accuracy": acc} for name, acc in c.agent_results]
