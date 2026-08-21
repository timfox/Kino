"""Paper anchors and table stubs for PagedWeight (arXiv:2607.16184)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "PagedWeight",
    "fp16_equiv_mem_savings": 0.72,
    "fp16_equiv_throughput_x": 1.94,
    "quality_gain_vs_quant": 0.393,
    "max_throughput_loss": 0.041,
    "longbench_fp16_avg": 0.17,
    "pagedweight_10gb_avg": 0.17,
    "apl_3bit_avg": 0.122,
    "ablation_full_wikitext2": 7.22,
    "ablation_full_c4": 10.06,
    "qwen_params_b": 14.3,
    "mixtral_params_b": 46.7,
    "gemma4_params_b": 25.2,
}

TABLE_1_MODELS: list[dict[str, Any]] = [
    {"model": "Qwen1.5-MoE-A2.7B", "params_b": 14.3, "experts": "60+4", "topk": 4},
    {"model": "Mixtral-8×7B-v0.1", "params_b": 46.7, "experts": "8", "topk": 2},
    {"model": "Gemma-4-26B-A4B", "params_b": 25.2, "experts": "128+1", "topk": 8},
]

TABLE_2_LONGBENCH: list[dict[str, Any]] = [
    {"method": "FP16", "memory_gb": 35.25, "avg": 0.170},
    {"method": "APL-3bit", "memory_gb": 9.40, "avg": 0.122},
    {"method": "PagedWeight-10GB", "memory_gb": 9.86, "avg": 0.170},
    {"method": "PagedWeight-13GB", "memory_gb": 12.79, "avg": 0.165},
    {"method": "PagedWeight-15GB", "memory_gb": 14.83, "avg": 0.168},
]

TABLE_3_THROUGHPUT: list[dict[str, Any]] = [
    {"method": "FP16", "b1_tps": 67.1, "b4_tps": 258.0},
    {"method": "Uniform-low", "b1_tps": 134.5, "b4_tps": 429.9},
    {"method": "PagedWeight-low", "b1_tps": 130.1, "b4_tps": 419.4, "max_loss_vs_uniform": 0.041},
]

TABLE_4_ABLATION: list[dict[str, Any]] = [
    {"config": "PagedWeight", "wikitext2": 7.22, "c4": 10.06},
    {"config": "w/o routing statistics", "wikitext2": 7.26, "c4": 10.13},
    {"config": "w/o prompt residual", "wikitext2": 7.31, "c4": 10.19},
    {"config": "w/o page movement", "wikitext2": 7.43, "c4": 10.33},
    {"config": "w/o global sensitivity", "wikitext2": 7.46, "c4": 10.40},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_models": list(TABLE_1_MODELS),
        "table_2_longbench": list(TABLE_2_LONGBENCH),
        "table_3_throughput": list(TABLE_3_THROUGHPUT),
        "table_4_ablation": list(TABLE_4_ABLATION),
    }
