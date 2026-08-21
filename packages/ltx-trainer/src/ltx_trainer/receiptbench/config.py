"""Configuration for ReceiptBench (Chen et al., arXiv:2605.22413)."""

from __future__ import annotations

from dataclasses import dataclass


SUBTASKS: tuple[str, ...] = (
    "perception",
    "normalization",
    "reasoning",
    "structure",
)

DOCUMENT_TYPES: tuple[str, ...] = (
    "purchase_dining",
    "plane_ticket",
    "hotel_bill",
    "taxi_receipt",
    "train_ticket",
    "bus_ticket",
    "ship_ferry",
    "fuel_receipt",
    "metro_ticket",
    "other",
)


@dataclass
class ReceiptBenchConfig:
    """Defaults from paper Sec. 3–5 and Appendix B."""

    dataset_size: int = 10_656
    num_fields: int = 19
    test_size: int = 2_000
    numeric_epsilon: float = 1e-6
    list_match_threshold: float = 0.25
    amount_mismatch_cost: float = 0.05
    # Composite similarity weights (Eq. 1, Config A)
    weight_levenshtein: float = 0.3
    weight_token_sort: float = 0.2
    weight_lcs: float = 0.1
    weight_semantic: float = 0.4
    # Metric-aware GRPO (Eq. 2)
    reward_tn: float = 0.3
    reward_fp: float = -0.5
    reward_fn: float = 0.0
    expense_types: tuple[str, ...] = (
        "plane",
        "train",
        "ship",
        "bus",
        "taxi",
        "metro",
        "hotel",
        "other",
    )
