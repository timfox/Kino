"""MUSTBENCH five temporally grounded QA tasks (Sec. 3.2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mustbench.config import MustBenchConfig, MustTask


def task_registry(cfg: MustBenchConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or MustBenchConfig()
    return [
        {
            "task": MustTask.TSG.value,
            "abbr": "TSG",
            "name": "Temporal Source Grounding",
            "description": "When does an instrument/vocal first enter or finally exit?",
            "answer_format": "single timestamp",
            "test_n": cfg.tsg_n,
            "metric": "Hit@3s",
        },
        {
            "task": MustTask.LTR.value,
            "abbr": "LTR",
            "name": "Local Transition Recognition",
            "description": "Multiple-choice transition at a given timestamp",
            "answer_format": "A/B/C/D",
            "test_n": cfg.ltr_n,
            "metric": "Accuracy",
        },
        {
            "task": MustTask.TAD.value,
            "abbr": "TAD",
            "name": "Transition-Aware Description",
            "description": "Open-ended musical change at timestamp",
            "answer_format": "free text",
            "test_n": cfg.tad_n,
            "metric": "METEOR + CLAP Score",
        },
        {
            "task": MustTask.GTO.value,
            "abbr": "GTO",
            "name": "Global Temporal Ordering",
            "description": "Chronological order of three transition events",
            "answer_format": "permutation (6 choices)",
            "test_n": cfg.gto_n,
            "metric": "Accuracy",
        },
        {
            "task": MustTask.MTR.value,
            "abbr": "MTR",
            "name": "Mood Trajectory Reasoning",
            "description": "Highest/lowest arousal temporal intervals",
            "answer_format": "interval list [start-end]",
            "test_n": cfg.mtr_n,
            "metric": "Temporal IoU + F1",
        },
    ]


def split_statistics(cfg: MustBenchConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — train/val/test per task."""
    cfg = cfg or MustBenchConfig()
    rows = [
        ("TSG", 8_000, 378, cfg.tsg_n),
        ("LTR", 8_000, 764, cfg.ltr_n),
        ("TAD", 8_000, 683, cfg.tad_n),
        ("GTO", 8_000, 767, cfg.gto_n),
        ("MTR", 8_000, 1_975, cfg.mtr_n),
    ]
    return [
        {
            "task": abbr,
            "train": train,
            "val": val,
            "test": test,
            "total": train + val + test,
        }
        for abbr, train, val, test in rows
    ]
