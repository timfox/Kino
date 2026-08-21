"""Reference metrics (Lin & Zheng, arXiv:2602.21992)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoenv.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.panoenv.panoenv_qa import dataset_card

# Table 2 — baseline VLMs (selected)
TABLE2_BASELINES: list[dict[str, Any]] = [
    {"model": "Qwen2.5-VL-7B", "acc": 49.34, "q_score": 5.60, "p_score": 5.48, "oe": 6.39},
    {"model": "Qwen2.5-VL-32B", "acc": 42.70, "q_score": 5.02, "p_score": 4.92, "oe": 8.36},
    {"model": "InternVL2.5-26B", "acc": 47.07, "q_score": 5.61, "p_score": 5.61, "oe": 3.44},
    {"model": "DeepSeek-VL2-Base", "acc": 38.86, "q_score": 5.24, "p_score": 5.54, "oe": 8.36},
    {"model": "Average", "acc": 36.72, "q_score": 4.82, "p_score": 5.17, "oe": 4.26},
]

# Table 3 — GRPO-trained (Ours)
TABLE3_GRPO: list[dict[str, Any]] = [
    {"model": "Qwen2.5-VL-7B (Base)", "total": 49.34, "tf": 65.19, "mc": 57.24, "oe": 6.39, "params": "7B"},
    {"model": "GRPO-Balanced (Ours)", "total": 52.93, "tf": 68.78, "mc": 58.90, "oe": 14.83, "params": "7B"},
]

# Table 4 — ablation
TABLE4_ABLATION: list[dict[str, Any]] = [
    {"variant": "Baseline (Qwen2.5-VL-7B)", "total": 49.3, "oe": 6.4},
    {"variant": "GRPO-OneStage", "total": 50.8, "oe": 11.8},
    {"variant": "GRPO-Structured", "total": 52.3, "oe": 5.7},
    {"variant": "GRPO-OE", "total": 48.6, "oe": 13.2},
    {"variant": "GRPO-Reverse", "total": 50.9, "oe": 7.0},
    {"variant": "GRPO-Balanced (Struct→Mixed)", "total": 52.9, "oe": 14.8},
]

# Table 7 — OSR-Bench zero-shot transfer
TABLE7_OSR_TRANSFER: list[dict[str, Any]] = [
    {"model": "Qwen2.5-VL-7B (Base)", "obj_count": 0.477, "rel_dist": 0.321, "rel_dir": 0.089},
    {"model": "Qwen2.5-VL-72B", "obj_count": 0.498, "rel_dist": 0.325, "rel_dir": 0.181},
    {"model": "PanoEnv-RL (Ours)", "obj_count": 0.507, "rel_dist": 0.371, "rel_dir": 0.105},
]


def table3_ours() -> dict[str, Any]:
    return next(r for r in TABLE3_GRPO if "GRPO-Balanced" in r["model"])


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "dataset": dataset_card(),
        "table1_categories": dataset_card()["categories"],
        "table2_baselines": TABLE2_BASELINES,
        "table3_grpo": TABLE3_GRPO,
        "table4_ablation": TABLE4_ABLATION,
        "table7_osr_transfer": TABLE7_OSR_TRANSFER,
    }
