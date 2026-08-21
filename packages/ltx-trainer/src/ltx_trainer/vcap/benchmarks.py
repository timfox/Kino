"""Paper benchmark tables (Sec. 3, Tables 1–5)."""

from __future__ import annotations

from typing import Any

TABLE1_IMAGE = [
    {"model": "CapRL", "capmas_avg": 70.98, "decap_avg": 63.01},
    {"model": "Qwen3-VL-8B-Instr", "capmas_avg": 77.91, "decap_avg": 69.17},
    {"model": "Qwen3-VL-235B-Instr", "capmas_avg": 77.60, "decap_avg": 67.88},
    {"model": "Qwen3.5-397B", "capmas_avg": 78.39, "decap_avg": 67.18},
    {"model": "Seed 2.0 Pro", "capmas_avg": 79.71, "decap_avg": 67.35},
    {"model": "Gemini 3.1 Pro", "capmas_avg": 79.85, "decap_avg": 70.20},
    {"model": "GPT-5.4", "capmas_avg": 81.36, "decap_avg": 73.18},
    {"model": "VCap (e1)", "capmas_avg": 81.52, "decap_avg": 73.32},
    {"model": "VCap (e2)", "capmas_avg": 83.53, "decap_avg": 73.67},
]

TABLE1_CAPMAS_DETAIL = [
    {"model": "VCap (e2)", "clair": 89.99, "coverage": 74.18, "factuality": 86.42, "avg": 83.53},
    {"model": "GPT-5.4", "clair": 88.18, "coverage": 73.92, "factuality": 81.99, "avg": 81.36},
    {"model": "VCap (e1)", "clair": 89.14, "coverage": 73.95, "factuality": 81.48, "avg": 81.52},
]

TABLE2_VIDEO = [
    {"model": "Qwen3-VL-8B-Instr", "vcaps_ar": 63.28, "vdc_avg": 33.76},
    {"model": "Seed 2.0 Pro", "vcaps_ar": 76.53, "vdc_avg": 30.39},
    {"model": "VCap (e1)", "vcaps_ar": 71.34, "vdc_avg": 35.21},
    {"model": "VCap (e2)", "vcaps_ar": 72.15, "vdc_avg": 36.01},
]

TABLE3_BON = [
    {"model": "Qwen3-VL-8B-Instr", "capmas_avg": 77.91, "decap_avg": 69.17},
    {"model": "+ Self-distill", "capmas_avg": 77.65, "decap_avg": 67.97},
    {"model": "VCap (e1)", "capmas_avg": 81.52, "decap_avg": 73.32},
    {"model": "+ Distill from VCap (e1)", "capmas_avg": 82.00, "decap_avg": 72.53},
    {"model": "VCap (e2)", "capmas_avg": 83.53, "decap_avg": 73.67},
    {"model": "+ Distill from VCap (e2)", "capmas_avg": 83.02, "decap_avg": 72.99},
]

TABLE5_ABLATION = [
    {"setup": "Qwen3-VL-8B-Instr (no RL)", "capmas_avg": 77.91, "decap_avg": 69.17},
    {"setup": "VCap (e1) – full", "capmas_avg": 81.52, "decap_avg": 73.32},
    {"setup": "− reference caption", "capmas_avg": 77.65, "decap_avg": 69.52},
    {"setup": "− reference image", "capmas_avg": 81.04, "decap_avg": 71.66},
    {"setup": "− Correctness", "capmas_avg": 78.84, "decap_avg": 72.80},
    {"setup": "− Completeness", "capmas_avg": 80.83, "decap_avg": 71.59},
    {"setup": "− Text Quality", "capmas_avg": 81.38, "decap_avg": 73.02},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_image_summary": TABLE1_IMAGE,
        "table1_capmas_vcap_e2": TABLE1_CAPMAS_DETAIL,
        "table2_video": TABLE2_VIDEO,
        "table3_best_of_n": TABLE3_BON,
        "table5_ablation": TABLE5_ABLATION,
        "vcap_e2_capmas_avg": 83.53,
        "vcap_e2_decap_avg": 73.67,
        "gpt54_capmas_avg": 81.36,
        "human_pairwise_agreement_pct": 61.1,
    }
