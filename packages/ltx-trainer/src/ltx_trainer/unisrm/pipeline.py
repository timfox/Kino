"""Framework card and paper benchmark excerpts (arXiv:2605.23261)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.unisrm.config import UnisrmConfig
from ltx_trainer.unisrm.layout import LIMITATIONS
from ltx_trainer.unisrm.mock import evaluation_smoke


def framework_card(cfg: UnisrmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UnisrmConfig()
    return {
        "name": "UniSRM — unified speech reward model",
        "paper": cfg.paper_arxiv,
        "backbone": cfg.backbone,
        "repo": cfg.repo_url,
        "output_format": "<think> dimension scores + rationale </think><answer> decision/scores </answer>",
        "tasks": {
            "T1": "Utterance-level A/B preference (4 dimensions, 0–10)",
            "T2": "QualiSpeech MOS-style quality (7 aspects, 1–5)",
            "T3": "Scenario-aware style consistency EN/ZH (3 dimensions)",
            "T4": "Multi-turn dialogue evaluation (5 dimensions, raw audio history)",
        },
        "training": {
            "stage_1": "SFT on UniSRM-Data (DSFT)",
            "stage_2": "RCR-GRPO on high-quality DRL (G=8 rollouts, β=0.04 KL)",
        },
        "rewards": {
            "combined": "λ_fmt R_fmt + λ_acc R_acc + λ_rc R_rc",
            "rcr_pairwise": "per-dimension sign(ai−bi) vs sign(a*−b*)",
            "rcr_mos": "normalized L1 on seven aspect scores",
        },
        "headlines": {
            "T1_acc": cfg.bench_t1_acc,
            "T2_acc_pcc": f"{cfg.bench_t2_acc}% / {cfg.bench_t2_pcc}",
            "T3_en_acc": cfg.bench_t3_en_acc,
            "T3_zh_acc": cfg.bench_t3_zh_acc,
            "T4_acc": cfg.bench_t4_acc,
            "dataset_samples": cfg.dataset_total_samples,
        },
        "limitations": LIMITATIONS,
    }


def table_1_unisrm_bench() -> list[dict[str, Any]]:
    """Table 1: overall UNISRM-BENCH results (accuracy % / T2 PCC)."""
    return [
        {"model": "WER", "T1": 59.24, "T2_acc": None, "T2_pcc": None, "T3_en": 61.44, "T3_zh": 56.92, "T4": 84.10},
        {"model": "UTMOS", "T1": 50.20, "T2_acc": None, "T2_pcc": 0.449, "T3_en": 33.21, "T3_zh": 48.19, "T4": 40.48},
        {"model": "GPT-4o-Audio", "T1": 61.04, "T2_acc": 24.60, "T2_pcc": 0.060, "T3_en": 64.02, "T3_zh": 64.82, "T4": 71.96},
        {"model": "Gemini-2.5-Pro", "T1": 60.67, "T2_acc": 28.93, "T2_pcc": 0.517, "T3_en": 67.31, "T3_zh": 63.47, "T4": 82.40},
        {"model": "Qwen2.5-Omni-7B", "T1": 51.20, "T2_acc": 24.03, "T2_pcc": 0.289, "T3_en": 49.45, "T3_zh": 52.17, "T4": 56.35},
        {"model": "SpeechJudge", "T1": 57.20, "T2_acc": None, "T2_pcc": None, "T3_en": None, "T3_zh": None, "T4": None},
        {"model": "UniSRM (Ours)", "T1": 65.06, "T2_acc": 39.74, "T2_pcc": 0.551, "T3_en": 85.61, "T3_zh": 91.30, "T4": 88.89},
    ]


def table_2_ablation() -> list[dict[str, Any]]:
    """Table 2: ablation over all tasks (accuracy %)."""
    return [
        {"variant": "UniSRM (Ours)", "T1": 65.06, "T2": 39.74, "T3_en": 85.61, "T3_zh": 91.30, "T4": 88.89},
        {"variant": "w/o RCR-GRPO", "T1": 60.44, "T2": 37.58, "T3_en": 80.81, "T3_zh": 81.42, "T4": 82.54},
        {"variant": "w/o GRPO", "T1": 60.24, "T2": 39.20, "T3_en": 67.16, "T3_zh": 70.95, "T4": 74.60},
    ]


def table_3_task1_dimensions() -> list[dict[str, Any]]:
    """Table 3: T1 per-dimension accuracy (%)."""
    return [
        {"variant": "UniSRM (Ours)", "text": 83.33, "sim": 62.25, "expressiveness": 61.24, "naturalness": 43.98, "avg": 62.70},
        {"variant": "w/o RCR-GRPO", "text": 76.89, "sim": 59.22, "expressiveness": 60.23, "naturalness": 39.76, "avg": 59.03},
        {"variant": "w/o GRPO", "text": 83.53, "sim": 57.83, "expressiveness": 59.84, "naturalness": 42.37, "avg": 60.89},
    ]


def table_4_task2_pcc() -> list[dict[str, Any]]:
    """Table 4: T2 per-aspect PCC on QualiSpeech."""
    return [
        {"model": "QualiSpeech", "noise": 0.686, "distortion": 0.518, "speed": 0.250, "continuity": 0.459, "effort": 0.475, "naturalness": 0.486, "overall": 0.572, "avg": 0.492},
        {"model": "UniSRM (Ours)", "noise": 0.754, "distortion": 0.547, "speed": 0.209, "continuity": 0.526, "effort": 0.478, "naturalness": 0.473, "overall": 0.551, "avg": 0.505},
        {"model": "w/o RCR-GRPO", "noise": 0.688, "distortion": 0.528, "speed": 0.233, "continuity": 0.512, "effort": 0.446, "naturalness": 0.418, "overall": 0.542, "avg": 0.481},
        {"model": "w/o GRPO", "noise": 0.714, "distortion": 0.514, "speed": 0.268, "continuity": 0.471, "effort": 0.481, "naturalness": 0.506, "overall": 0.534, "avg": 0.498},
    ]


def table_7_cross_dataset() -> list[dict[str, Any]]:
    """Table 7: BVCC and SOMOS generalization."""
    return [
        {"model": "DNSMOS", "BVCC_pcc": 0.2990, "BVCC_acc": None, "SOMOS_clean_pcc": 0.0479, "SOMOS_clean_acc": None, "SOMOS_full_pcc": 0.0528, "SOMOS_full_acc": None},
        {"model": "Qwen2.5-Omni-7B", "BVCC_pcc": 0.2563, "BVCC_acc": 25.57, "SOMOS_clean_pcc": 0.1561, "SOMOS_clean_acc": 23.17, "SOMOS_full_pcc": 0.1484, "SOMOS_full_acc": 22.70},
        {"model": "Gemini-2.5-Pro", "BVCC_pcc": 0.3390, "BVCC_acc": 27.42, "SOMOS_clean_pcc": 0.2009, "SOMOS_clean_acc": 30.71, "SOMOS_full_pcc": 0.2218, "SOMOS_full_acc": 33.94},
        {"model": "UniSRM", "BVCC_pcc": 0.4977, "BVCC_acc": 49.16, "SOMOS_clean_pcc": 0.2612, "SOMOS_clean_acc": 41.70, "SOMOS_full_pcc": 0.2347, "SOMOS_full_acc": 52.97},
    ]


def table_9_dataset_stats() -> list[dict[str, Any]]:
    """Table 9: UniSRM-Data split sizes."""
    return [
        {"task": "T1 A/B preference", "sft": 11146, "rl": 2787, "bench": 498, "total": 14431},
        {"task": "T2 QualiSpeech MOS", "sft": 10558, "rl": 2167, "bench": 1852, "total": 14577},
        {"task": "T3 Scenario EN", "sft": 5018, "rl": 1815, "bench": 542, "total": 7375},
        {"task": "T3 Scenario ZH", "sft": 4869, "rl": 1989, "bench": 506, "total": 7364},
        {"task": "T4 Dialogue", "sft": 1470, "rl": 916, "bench": 126, "total": 2512},
        {"task": "Total", "sft": 33061, "rl": 9674, "bench": 3524, "total": 46259},
    ]


def headline_results() -> dict[str, Any]:
    cfg = UnisrmConfig()
    return {
        "T1_acc": cfg.bench_t1_acc,
        "T2_acc": cfg.bench_t2_acc,
        "T2_pcc": cfg.bench_t2_pcc,
        "T3_en_acc": cfg.bench_t3_en_acc,
        "T3_zh_acc": cfg.bench_t3_zh_acc,
        "T4_acc": cfg.bench_t4_acc,
        "dataset_total": cfg.dataset_total_samples,
    }


def evaluation_demo(cfg: UnisrmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UnisrmConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_1_unisrm_bench": table_1_unisrm_bench(),
        "table_2_ablation": table_2_ablation(),
        "table_3_task1_dimensions": table_3_task1_dimensions(),
        "table_4_task2_pcc": table_4_task2_pcc(),
        "table_7_cross_dataset": table_7_cross_dataset(),
        "table_9_dataset_stats": table_9_dataset_stats(),
        "headlines": headline_results(),
    }
