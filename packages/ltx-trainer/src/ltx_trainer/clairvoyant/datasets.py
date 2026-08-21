"""Training dataset Long-class statistics (Table 2, Sec. 4.2)."""

from __future__ import annotations

from typing import Any


def dataset_long_class_table() -> list[dict[str, Any]]:
    """Paper Table 2: Long-class representation across seven datasets."""
    return [
        {"dataset": "ShareGPT", "total": 48312, "pct_long": 14.8, "usable_for_training": True},
        {"dataset": "LMSYS-Chat-1M", "total": 876412, "pct_long": 12.1, "usable_for_training": True},
        {"dataset": "OASST1", "total": 8792, "pct_long": 6.3, "usable_for_training": True},
        {"dataset": "Alpaca 52K", "total": 52002, "n_long": 4, "pct_long": 0.008, "usable_for_training": False},
        {"dataset": "CodeAlpaca 20K", "total": 20022, "n_long": 3, "pct_long": 0.015, "usable_for_training": False},
        {"dataset": "Dolly 15K", "total": 15011, "n_long": 88, "pct_long": 0.6, "usable_for_training": "test_only"},
        {"dataset": "CNN/DailyMail", "total": 11490, "n_long": 1, "pct_long": 0.009, "usable_for_training": "test_only"},
    ]


def model_splits_table() -> list[dict[str, Any]]:
    """Paper Table 3: balanced training splits per model variant."""
    return [
        {"model": "A", "dataset": "ShareGPT", "train": 4800, "val": 600, "test": 600},
        {"model": "B", "dataset": "LMSYS-Chat-1M", "train": 4800, "val": 600, "test": 600},
        {"model": "C", "dataset": "OASST1", "train": 660, "val": 83, "test": 84},
    ]


def datasets_card() -> dict[str, Any]:
    return {
        "long_class_minimum_for_training": 200,
        "instruction_datasets_degenerate": ["Alpaca 52K", "CodeAlpaca 20K"],
        "natural_conversation_sources": ["ShareGPT", "LMSYS-Chat-1M", "OASST1"],
        "table2": dataset_long_class_table(),
        "table3_splits": model_splits_table(),
        "filtering_note": "English langdetect p>0.95; Llama-2 tokenizer lengths; stratified balance",
    }
