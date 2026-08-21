"""Paper benchmark tables (arXiv:2605.28722)."""

from __future__ import annotations

from typing import Any

# Table 1 excerpt — Llama-3-8B and Qwen2.5-32B rows
TABLE1: list[dict[str, Any]] = [
    {
        "model": "Llama-3-8B",
        "method": "Vanilla",
        "TruthfulQA_MC1": 28.70,
        "TruthfulQA_MC2": 45.03,
        "BBQ": 0.608,
        "Refusal": 0.851,
        "MMLU": 65.9,
    },
    {
        "model": "Llama-3-8B",
        "method": "ReFT",
        "TruthfulQA_MC1": 50.58,
        "TruthfulQA_MC2": 52.51,
        "BBQ": 0.637,
        "Refusal": 0.861,
        "MMLU": 66.0,
    },
    {
        "model": "Llama-3-8B",
        "method": "MARI",
        "TruthfulQA_MC1": 61.81,
        "TruthfulQA_MC2": 67.39,
        "BBQ": 0.792,
        "Refusal": 0.866,
        "MMLU": 66.6,
    },
    {
        "model": "Qwen2.5-32B",
        "method": "Vanilla",
        "TruthfulQA_MC1": 43.98,
        "TruthfulQA_MC2": 59.81,
        "BBQ": 0.736,
        "Refusal": 0.621,
        "MMLU": 83.5,
    },
    {
        "model": "Qwen2.5-32B",
        "method": "MARI",
        "TruthfulQA_MC1": 81.94,
        "TruthfulQA_MC2": 70.59,
        "BBQ": 0.876,
        "Refusal": 0.653,
        "MMLU": 84.2,
    },
]

# Table 2 ablation — Llama-3-8B
TABLE2_ABLATION: list[dict[str, Any]] = [
    {"variant": "Vanilla", "MC1": 28.70, "MMLU": 65.9},
    {"variant": "w/o Energy Gating", "MC1": 65.15, "MMLU": 57.5},
    {"variant": "w/o Multi-Adapter", "MC1": 45.80, "MMLU": 66.2},
    {"variant": "EG-MARI", "MC1": 61.81, "MMLU": 66.6},
]

# Table 3 fixed-threshold transfer
TABLE3_TRANSFER: list[dict[str, Any]] = [
    {"rho": 0.50, "TQA_MC1": 65.81, "GSM8K": 74.5},
    {"rho": 0.70, "TQA_MC1": 65.35, "GSM8K": 76.8},
    {"rho": 0.90, "TQA_MC1": 64.81, "GSM8K": 77.4},
    {"rho": 1.00, "TQA_MC1": 63.90, "GSM8K": 77.2},
]

# Table 4 inference efficiency
TABLE4_LATENCY: list[dict[str, Any]] = [
    {"method": "Base Model", "FTL_ms": 22.12, "TPS": 476.26, "TTLT_s": 0.0223},
    {"method": "ReFT", "FTL_ms": 43.64, "TPS": 445.01, "TTLT_s": 0.0439},
    {"method": "MARI", "FTL_ms": 43.53, "TPS": 446.32, "TTLT_s": 0.0448},
]

INFERENCE_OVERHEAD_NOTE = (
    "Router scores K experts in one batched prompt pass; energy probe is prompt-only "
    "before generation (Appendix B.4)."
)
