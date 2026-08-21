"""ReceiptBench: receipt VIE benchmark + metric-aware GRPO (Chen et al., arXiv:2605.22413)."""

from ltx_trainer.receiptbench.config import (
    DOCUMENT_TYPES,
    SUBTASKS,
    ReceiptBenchConfig,
)
from ltx_trainer.receiptbench.evaluation import (
    evaluate_receipt,
    f1_from_counts,
    field_similarity,
)
from ltx_trainer.receiptbench.matching import list_field_similarity
from ltx_trainer.receiptbench.pipeline import (
    dataset_summary,
    demo_gold_receipt,
    demo_prediction_errors,
    grpo_group_rewards,
)
from ltx_trainer.receiptbench.rewards import field_reward, invoice_reward
from ltx_trainer.receiptbench.schema import (
    ALL_FIELDS,
    FIELD_SPEC,
    NORMALIZATION_FIELDS,
    PERCEPTION_FIELDS,
    REASONING_FIELDS,
    STRUCTURE_FIELDS,
    fields_for_subtask,
)
from ltx_trainer.receiptbench.similarity import composite_similarity, levenshtein_ratio

__all__ = [
    "ALL_FIELDS",
    "DOCUMENT_TYPES",
    "FIELD_SPEC",
    "NORMALIZATION_FIELDS",
    "PERCEPTION_FIELDS",
    "REASONING_FIELDS",
    "STRUCTURE_FIELDS",
    "SUBTASKS",
    "ReceiptBenchConfig",
    "composite_similarity",
    "dataset_summary",
    "demo_gold_receipt",
    "demo_prediction_errors",
    "evaluate_receipt",
    "f1_from_counts",
    "field_reward",
    "field_similarity",
    "grpo_group_rewards",
    "invoice_reward",
    "levenshtein_ratio",
    "list_field_similarity",
    "fields_for_subtask",
]
