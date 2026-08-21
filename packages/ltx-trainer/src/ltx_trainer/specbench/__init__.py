"""SpecBench: RFC specification-deficiency benchmark (arXiv:2605.30314)."""

from ltx_trainer.specbench.benchmark import (
    benchmark_card,
    evaluate_agent,
    evaluate_agent_on_task,
    paper_agent_leaderboard,
)
from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.judging import MatchPair, ensemble_match, match_predictions
from ltx_trainer.specbench.mock import evaluation_smoke
from ltx_trainer.specbench.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    full_benchmark_demo,
    knowledge_card,
)
from ltx_trainer.specbench.scoring import TaskScore, aggregate_accuracy, prediction_budget, score_task
from ltx_trainer.specbench.spi import SPITriple, decompose_deficiency, spi_subject_predicate_match
from ltx_trainer.specbench.tasks import (
    GoldDeficiency,
    SpecBenchTask,
    all_tasks,
    codex54_predictions_kep4671,
    codex54_prediction_spis,
    codex54_scoring_predictions,
    paper_table5_match_pairs,
    task_by_id,
)
from ltx_trainer.specbench.taxonomy import (
    DEFICIENCY_CLASSES,
    DeficiencyClass,
    GoldTier,
    REPOSITORY_LABELS,
    classify_deficiency_text,
)

__all__ = [
    "DEFICIENCY_CLASSES",
    "DeficiencyClass",
    "GoldDeficiency",
    "GoldTier",
    "MatchPair",
    "REPOSITORY_LABELS",
    "SPITriple",
    "SpecBenchConfig",
    "SpecBenchTask",
    "TaskScore",
    "aggregate_accuracy",
    "all_tasks",
    "benchmark_card",
    "benchmarks_bundle",
    "codex54_predictions_kep4671",
    "codex54_prediction_spis",
    "codex54_scoring_predictions",
    "paper_table5_match_pairs",
    "classify_deficiency_text",
    "decompose_deficiency",
    "ensemble_match",
    "evaluate_agent",
    "evaluate_agent_on_task",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "full_benchmark_demo",
    "knowledge_card",
    "match_predictions",
    "paper_agent_leaderboard",
    "prediction_budget",
    "score_task",
    "spi_subject_predicate_match",
    "task_by_id",
]
