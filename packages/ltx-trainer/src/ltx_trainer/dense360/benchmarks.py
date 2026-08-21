"""Paper tables (Zhou et al., arXiv:2506.14471)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dense360.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 3 — Dense360-Bench (caption / grounding %)
TABLE3_BENCH = {
    "Qwen2.5VL-72B": {
        "caption": {"front": 32.15, "right": 33.10, "back": 33.05, "left": 32.06, "omni": 32.44},
        "grounding": {"front": 36.70, "right": 39.43, "back": 26.91, "left": 38.30, "omni": 35.89},
    },
    "InternVL3-78B": {
        "caption": {"front": 28.95, "right": 29.06, "back": 32.00, "left": 29.63, "omni": 29.58},
        "grounding": {"front": 55.36, "right": 51.74, "back": 46.06, "left": 49.13, "omni": 50.50},
    },
    "ChatGPT4o-latest": {
        "caption": {"front": 45.06, "right": 47.03, "back": 48.33, "left": 46.04, "omni": 46.42},
    },
    "SA2VA-4B_dense360": {
        "caption": {"front": 49.89, "right": 49.49, "back": 42.98, "left": 49.68, "omni": 47.80},
        "grounding": {"front": 74.66, "right": 74.12, "back": 74.25, "left": 73.93, "omni": 74.39},
    },
    "Dense360VLM-3B": {
        "caption": {"front": 52.67, "right": 53.21, "back": 50.45, "left": 52.47, "omni": 51.78},
        "grounding": {"front": 76.83, "right": 76.73, "back": 76.56, "left": 76.70, "omni": 76.81},
    },
}

# Table 4 — ablation (omnidirectional scores)
TABLE4_ABLATION = {
    "baseline_only": {"caption_omni": 14.63, "grounding_omni": None},
    "caption_only_rope": {"caption_omni": 49.75, "grounding_omni": None},
    "refseg_only_rope": {"caption_omni": None, "grounding_omni": 70.28},
    "gcg_only_rope": {"caption_omni": None, "grounding_omni": 43.65},
    "full_no_rope": {"caption_omni": 45.86, "grounding_omni": 60.43},
    "full_with_rope": {"caption_omni": 51.78, "grounding_omni": 76.81},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table3_bench": TABLE3_BENCH,
        "table4_ablation": TABLE4_ABLATION,
    }
