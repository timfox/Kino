"""SpecBench evaluation smoke (arXiv:2605.30314)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.specbench.benchmark import benchmark_card, paper_agent_leaderboard
from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.scoring import prediction_budget, score_task
from ltx_trainer.specbench.tasks import codex54_scoring_predictions, paper_table5_match_pairs, task_by_id
from ltx_trainer.specbench.judging import MatchPair
from ltx_trainer.specbench.taxonomy import REPOSITORY_LABELS


def evaluation_smoke(cfg: SpecBenchConfig | None = None) -> dict[str, Any]:
    c = cfg or SpecBenchConfig()
    task = task_by_id("kubernetes_kep4671")
    assert task is not None
    texts, spis = codex54_scoring_predictions()
    forced = [MatchPair(pred_index=p, gold_id=g) for p, g in paper_table5_match_pairs()]
    kep_score = score_task(task, texts, c, pred_spis=spis, forced_pairs=forced)
    budget = prediction_budget(len(task.gold), c)

    return {
        "paper": c.paper_arxiv,
        "github": c.github_url,
        "repositories": list(REPOSITORY_LABELS.keys()),
        "n_repositories": c.n_repositories,
        "deficiency_classes": 4,
        "prediction_budget_formula": f"ceil({c.prediction_budget_multiplier} × |G|)",
        "kep4671_prediction_budget": budget,
        "kep4671_accuracy": round(kep_score.accuracy, 4),
        "kep4671_core_matches": f"{kep_score.core_matches}/{kep_score.n_core}",
        "kep4671_extended_matches": f"{kep_score.extended_matches}/{kep_score.n_extended}",
        "paper_best_agent": c.paper_best_agent,
        "paper_best_accuracy": c.paper_best_accuracy,
        "leaderboard_top": paper_agent_leaderboard(c)[0],
        "benchmark_card": benchmark_card(),
    }
