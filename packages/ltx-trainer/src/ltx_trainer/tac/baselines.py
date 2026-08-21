"""Table 1 baseline excerpts (arXiv:2605.28713)."""

from __future__ import annotations

from typing import Any

# Qwen3-8B rows from paper Table 1 (F1 / EM averages)
TABLE1_QWEN3_4X: list[dict[str, Any]] = [
    {"method": "Closed-book", "f1": 21.71, "em": 13.14},
    {"method": "Original Prompt", "f1": 52.85, "em": 40.48},
    {"method": "EXIT", "f1": 29.68, "em": 21.18},
    {"method": "Provence", "f1": 46.58, "em": 35.16},
    {"method": "LLMLingua-2-large", "f1": 38.37, "em": 27.42},
    {"method": "LongLLMLingua", "f1": 47.52, "em": 35.52},
    {"method": "ICAE", "f1": 26.40, "em": 17.76},
    {"method": "PCC", "f1": 30.02, "em": 18.64},
    {"method": "TaC-C", "f1": 65.12, "em": 51.17},
]

TABLE1_QWEN3_8X: list[dict[str, Any]] = [
    {"method": "LLMLingua-2-large", "f1": 29.26, "em": 20.42},
    {"method": "LongLLMLingua", "f1": 40.80, "em": 29.40},
    {"method": "ICAE", "f1": 27.91, "em": 18.98},
    {"method": "PCC", "f1": 24.37, "em": 14.06},
    {"method": "TaC-C", "f1": 65.42, "em": 51.79},
]

TABLE5_ABLATION: list[dict[str, Any]] = [
    {"config": "TaC-C", "em": 51.79, "f1": 65.42, "act_pct": 9.90, "hack_pct": 0.88},
    {"config": "w/o R_utility", "em": 30.25, "f1": 41.40, "act_pct": 6.60, "hack_pct": 1.17},
    {"config": "w/o R_budget", "em": 58.25, "f1": 71.78, "act_pct": 36.20, "hack_pct": 1.95},
    {"config": "w/o R_hack", "em": 61.72, "f1": 73.54, "act_pct": 10.30, "hack_pct": 71.67},
]

LOCOMO_4X: dict[str, Any] = {
    "method": "TaC-C",
    "f1": 41.94,
    "act_pct": 1.13,
    "prefill_ms": 77.9,
    "e2e_ms": 480.6,
}

# LLaMA-3.1-8B-Instruct rows (Table 1)
TABLE1_LLAMA_4X: list[dict[str, Any]] = [
    {"method": "Activation Beacon", "f1": 52.97, "em": 41.43},
    {"method": "LongLLMLingua", "f1": 47.55, "em": 36.02},
    {"method": "TaC-C", "f1": 70.09, "em": 57.14},
]

TABLE1_LLAMA_8X: list[dict[str, Any]] = [
    {"method": "Activation Beacon", "f1": 47.87, "em": 36.11},
    {"method": "LongLLMLingua", "f1": 40.56, "em": 29.89},
    {"method": "TaC-C", "f1": 70.02, "em": 57.19},
]

# Table 3: trace transfer across answer models (8×, Qwen3-8B thinker)
TABLE3_TRANSFER_8X: list[dict[str, Any]] = [
    {
        "answer_model": "LLaMA-3.1-8B-Instruct",
        "LongLLMLingua": {"f1": 40.56, "em": 29.89},
        "TaC-C": {"f1": 64.23, "em": 49.56},
    },
    {
        "answer_model": "Gemma-4-26B-A4B",
        "LongLLMLingua": {"f1": 30.28, "em": 19.72},
        "TaC-C": {"f1": 66.35, "em": 51.68},
    },
    {
        "answer_model": "Qwen3-30B-A3B",
        "LongLLMLingua": {"f1": 42.24, "em": 30.82},
        "TaC-C": {"f1": 66.35, "em": 52.95},
    },
]

# Table 6: dataset context length statistics
DATASET_STATS: list[dict[str, Any]] = [
    {"dataset": "NaturalQuestions", "samples": 2655, "avg_tokens": 2933},
    {"dataset": "2WikiMultihopQA", "samples": 12576, "avg_tokens": 1018},
    {"dataset": "HotpotQA", "samples": 7345, "avg_tokens": 1402},
    {"dataset": "MuSiQue", "samples": 2417, "avg_tokens": 2536},
    {"dataset": "Total", "samples": 24993, "avg_tokens": 1481},
]

# Figure 2 pilot study (retention %, EM %)
PILOT_STUDY: list[dict[str, Any]] = [
    {"method": "PCC", "retention_pct": 25.0, "em": 13.4},
    {"method": "LLMLingua-2", "retention_pct": 36.3, "em": 25.5},
    {"method": "LongLLMLingua", "retention_pct": 44.4, "em": 26.2},
    {"method": "TaC-Vanilla", "retention_pct": 53.2, "em": 63.0},
    {"method": "TaC-C", "retention_pct": 11.5, "em": 63.0},
]

# Figure 4: thinker scale vs EM (Qwen3 answerer, 8×)
THINKER_SCALE_EM: list[dict[str, Any]] = [
    {"thinker_b": 1.7, "naturalqa": 40.34, "2wikimqa": 49.99, "hotpotqa": 28.51, "musique": 37.63},
    {"thinker_b": 4, "naturalqa": 41.36, "2wikimqa": 56.75, "hotpotqa": 41.08, "musique": 41.85},
    {"thinker_b": 8, "naturalqa": 57.67, "2wikimqa": 58.08, "hotpotqa": 59.69, "musique": 42.28},
    {"thinker_b": 14, "naturalqa": 69.17, "2wikimqa": 59.69, "hotpotqa": 69.36, "musique": 42.45},
]

# Table 2: trace behavior types (excerpt)
TRACE_EXAMPLES: list[dict[str, Any]] = [
    {
        "type": "Extraction",
        "query": "When did Gina launch an ad campaign for her store?",
        "label": "29 January, 2023",
    },
    {
        "type": "Summarization",
        "query": "Does Calvin wish to become more popular?",
        "label": "Yes.",
    },
    {
        "type": "Association",
        "query": "Who owns the record label of the Shake What God Gave Ya performer?",
        "label": "Warner Music Group",
    },
]
