"""Mitigation categories (Sec. 5)."""

from __future__ import annotations

from typing import Any


def defense_catalog() -> list[dict[str, Any]]:
    return [
        {
            "category": "dynamic_benchmark",
            "examples": ["LiveBench", "LatestEval", "LiveCodeBench", "VarBench"],
            "tradeoff": "Reduces contamination; may hurt reproducibility.",
        },
        {
            "category": "private_secure_benchmarking",
            "examples": ["encrypted test sets", "confidential computing"],
            "tradeoff": "Strong integrity; limited open evaluation.",
        },
        {
            "category": "automated_decontamination",
            "examples": ["AntiLeak-Bench", "C2LEVA", "inference-time decontamination"],
            "tradeoff": "Scalable; weak on paraphrased overlap.",
        },
        {
            "category": "watermarking",
            "examples": ["TextMarker", "copyright traps", "Mosaic Memory"],
            "tradeoff": "IP enforcement; training complexity and legal uncertainty.",
        },
        {
            "category": "machine_unlearning",
            "examples": ["Harry Potter unlearning", "RWKU", "minority-aware unlearning"],
            "tradeoff": "Post-hoc removal; limited effectiveness today.",
        },
    ]
