"""Reference metrics from Fan et al. (arXiv:2603.09573)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panolm.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.panolm.panovqa import dataset_card

# Table 5 — PanoVQA (normalized GPT-score %, selected rows)
TABLE5_PANOVQA: list[dict[str, Any]] = [
    {"method": "Qwen2.5-VL-7B-Ins.", "zero_shot": True, "avg": 29.77},
    {"method": "InternVL3-8B-Ins.", "zero_shot": True, "avg": 34.48},
    {"method": "Qwen2.5-VL-32B-Ins.", "zero_shot": True, "avg": 35.56},
    {"method": "Gemini-2.5-Flash-Lite", "zero_shot": True, "avg": 30.74},
    {"method": "Qwen2.5-VL-7B", "zero_shot": False, "avg": 45.21},
    {"method": "PanoLM-7B (Ours)", "zero_shot": False, "avg": 45.48},
]

TABLE5_OURS_DETAIL: dict[str, Any] = {
    "method": "PanoLM-7B (Ours)",
    "N1": 34.79,
    "N2": 33.54,
    "N3": 28.18,
    "N4": 28.54,
    "O1": 18.62,
    "O2": 50.21,
    "O3": 60.09,
    "D1": 23.65,
    "D2": 42.92,
    "D3": 75.72,
    "D4": 76.74,
    "D5": 72.75,
    "avg_N": None,
    "avg_O": None,
    "avg_D": None,
    "avg": 45.48,
}

# Table 6 — ablation on PanoVQA-mini (3B)
TABLE6_ABLATION: list[dict[str, Any]] = [
    {"LoRA": True, "PSA": False, "LLM": False, "avg": 28.55, "params_M": 259.21},
    {"LoRA": False, "PSA": False, "SSA": True, "LLM": False, "avg": 29.80, "params_M": 85.09},
    {"LoRA": False, "PSA": True, "LLM": False, "avg": 32.14, "params_M": 95.56},
    {"LoRA": False, "PSA": True, "LLM": True, "avg": 41.49, "params_M": 3196.25},
    {"LoRA": False, "PSA": False, "LLM": True, "SFT": True, "avg": 41.42, "params_M": 3754.62},
]

# Table 8 — 1-pano vs 6-cam (PanoVQA-mini, SFT 3B)
TABLE8_INPUT_STRATEGY: list[dict[str, Any]] = [
    {"model": "Qwen2.5-VL-3B", "input": "Multi-view (6 cam)", "avg_N": 16.82, "avg_O": 32.40, "avg_D": 34.92, "avg": 28.26},
    {"model": "Qwen2.5-VL-3B-SFT", "input": "Multi-view (6 cam)", "avg_N": 26.33, "avg_O": 39.88, "avg_D": 54.45, "avg": 40.22},
    {"model": "Qwen2.5-VL-3B-SFT", "input": "Panoramic (1 pano)", "avg_N": 29.68, "avg_O": 40.98, "avg_D": 51.08, "avg": 41.42},
]

# Table 4 — human-GPT quality (1–5)
TABLE4_QUALITY: list[dict[str, Any]] = [
    {"subset": "PanoVQA-N", "question": 4.73, "correctness": 4.50, "actionability": 4.31, "fluency": 4.84},
    {"subset": "PanoVQA-O", "question": 4.21, "correctness": 4.04, "actionability": 3.52, "fluency": 4.66},
    {"subset": "PanoVQA-D", "question": 4.80, "correctness": 4.72, "actionability": 4.64, "fluency": 4.91},
]


def table5_ours() -> dict[str, Any]:
    return next(r for r in TABLE5_PANOVQA if "PanoLM" in r["method"])


def normalized_gpt_score(raw_1_to_5: float, n: int = 1) -> float:
    """SGPT = mean((si-1)/4)*100 (Sec. 4.1)."""
    return (raw_1_to_5 - 1.0) / 4.0 * 100.0 / max(n, 1)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "dataset": dataset_card(),
        "table5_panovqa": TABLE5_PANOVQA,
        "table5_ours_detail": TABLE5_OURS_DETAIL,
        "table6_ablation": TABLE6_ABLATION,
        "table8_input_strategy": TABLE8_INPUT_STRATEGY,
        "table4_quality": TABLE4_QUALITY,
    }
