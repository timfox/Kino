"""SWE-Mutation: LLM test-suite evaluation via agentic mutants (Sun et al., arXiv:2605.22175)."""

from ltx_trainer.swemutation.config import (
    MULTILINGUAL_LANGS,
    MUTATION_STRATEGY_GROUPS,
    SWEMutationConfig,
    TaskName,
)
from ltx_trainer.swemutation.framework import AgenticMutationPipeline, LocateScope, MutantCandidate
from ltx_trainer.swemutation.metrics import (
    absolute_mutation_score,
    pass_at_1,
    relative_detection_rate,
    verified_reproduction_rate,
)
from ltx_trainer.swemutation.mutation import (
    compare_mutation_methods,
    judge_mutant,
    mutation_strategy_catalog,
    self_play_select,
)
from ltx_trainer.swemutation.pipeline import (
    benchmark_table_multilingual,
    benchmark_table_test_generation,
    benchmark_table_test_repair,
    dataset_card,
    demo_instance_evaluation,
    demo_micro_rdr,
    evaluate_task_metrics,
    locate_ablation_table,
    mutant_quality_table,
    mutation_strategy_rdr_table,
)

__all__ = [
    "MULTILINGUAL_LANGS",
    "MUTATION_STRATEGY_GROUPS",
    "AgenticMutationPipeline",
    "LocateScope",
    "MutantCandidate",
    "SWEMutationConfig",
    "TaskName",
    "absolute_mutation_score",
    "benchmark_table_multilingual",
    "benchmark_table_test_generation",
    "benchmark_table_test_repair",
    "compare_mutation_methods",
    "dataset_card",
    "demo_instance_evaluation",
    "demo_micro_rdr",
    "evaluate_task_metrics",
    "judge_mutant",
    "locate_ablation_table",
    "mutation_strategy_catalog",
    "mutant_quality_table",
    "mutation_strategy_rdr_table",
    "pass_at_1",
    "relative_detection_rate",
    "self_play_select",
    "verified_reproduction_rate",
]
