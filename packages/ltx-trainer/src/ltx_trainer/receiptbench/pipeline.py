"""Training and evaluation glue for ReceiptBench."""

from __future__ import annotations

from typing import Any

from ltx_trainer.receiptbench.config import ReceiptBenchConfig
from ltx_trainer.receiptbench.evaluation import evaluate_receipt
from ltx_trainer.receiptbench.rewards import invoice_reward
from ltx_trainer.receiptbench.schema import ALL_FIELDS


def dataset_summary(cfg: ReceiptBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ReceiptBenchConfig()
    from ltx_trainer.receiptbench.config import DOCUMENT_TYPES, SUBTASKS
    from ltx_trainer.receiptbench.schema import (
        NORMALIZATION_FIELDS,
        PERCEPTION_FIELDS,
        REASONING_FIELDS,
        STRUCTURE_FIELDS,
    )

    return {
        "name": "ReceiptBench",
        "size": cfg.dataset_size,
        "num_fields": cfg.num_fields,
        "test_size": cfg.test_size,
        "subtasks": list(SUBTASKS),
        "document_types": list(DOCUMENT_TYPES),
        "fields_per_subtask": {
            "perception": len(PERCEPTION_FIELDS),
            "normalization": len(NORMALIZATION_FIELDS),
            "reasoning": len(REASONING_FIELDS),
            "structure": len(STRUCTURE_FIELDS),
        },
        "grpo_rewards": {
            "lambda_tn": cfg.reward_tn,
            "lambda_fp": cfg.reward_fp,
            "lambda_fn": cfg.reward_fn,
        },
    }


def demo_gold_receipt() -> dict[str, Any]:
    """Synthetic ground-truth JSON for smoke tests."""
    return {
        "type": "train",
        "std_start_time": "2024-07-06",
        "orig_start_time": "06 Jul 2024",
        "std_end_time": "",
        "orig_end_time": "",
        "std_invoice_time": "2024-06-03",
        "orig_invoice_time": "03 Jun 2024",
        "place": "Australia-Sydney",
        "departure": "Australia-Sydney",
        "arrival": "Australia-Canberra",
        "std_curr": "AUD",
        "orig_curr": ["$", "Sydney"],
        "std_total": "50.58",
        "orig_total": "50.58",
        "detail": [
            {"content": "Trip Fare", "amount": "45.00", "ifTax": False},
            {"content": "Tax fee", "amount": "5.58", "ifTax": True},
        ],
        "seller_name": ["NSW TrainLink"],
        "seller_address": ["Australia-Sydney"],
        "invoice_number": "0306202450122",
        "tax_number": "50 325 560 455",
    }


def demo_prediction_errors(gold: dict[str, Any]) -> dict[str, Any]:
    """Prediction with typical errors (missing + hallucination)."""
    pred = dict(gold)
    pred["invoice_number"] = ""
    pred["place"] = "United States-Carlsbad"
    pred["std_total"] = "0.00"
    return pred


def grpo_group_rewards(
    predictions: list[dict[str, Any]],
    gold: dict[str, Any],
    *,
    cfg: ReceiptBenchConfig | None = None,
) -> list[float]:
    """Rewards for a GRPO sample group."""
    return [invoice_reward(p, gold, cfg=cfg) for p in predictions]
