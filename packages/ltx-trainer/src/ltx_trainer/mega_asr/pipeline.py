"""Mega-ASR framework card, tables, and smoke demos (arXiv:2605.19833)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mega_asr.a2s_sft import a2s_sft_phases
from ltx_trainer.mega_asr.config import MegaASRConfig
from ltx_trainer.mega_asr.layout import LIMITATIONS
from ltx_trainer.mega_asr.rewards import ATOMIC_PHENOMENA, dg_wgpo_reward
from ltx_trainer.mega_asr.router import route_decision, router_architecture_summary


def framework_card(cfg: MegaASRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MegaASRConfig()
    return {
        "name": "Mega-ASR",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "hub_dataset": cfg.hub_dataset,
        "bench": cfg.bench_repo,
        "idea": (
            "Unified ASR-in-the-wild: VOICES-IN-THE-WILD-2M compound acoustic simulation, "
            "A2S-SFT progressive training, and DG-WGPO dual-granularity RL rewards."
        ),
        "dataset": {
            "clips": cfg.dataset_clips,
            "hours": cfg.dataset_hours,
            "atomic_phenomena": list(ATOMIC_PHENOMENA),
            "compound_scenarios": cfg.compound_scenarios,
        },
        "training": {
            "backbone": cfg.backbone,
            "a2s_sft": "3-phase WER-graded curriculum (encoder → LLM → joint)",
            "dg_wgpo": f"τ={cfg.wer_gate_tau}, α_s={cfg.soft_error_alpha}, α_dyn={cfg.dynamic_reward_weight}",
            "router": "MFCC binary clean/degraded LoRA routing (plug-and-play)",
        },
        "defaults": cfg.__dict__,
    }


def table_i_robust_benchmarks() -> list[dict[str, Any]]:
    """Table 2 excerpt — adverse-condition ASR (avg WER %, lower better)."""
    return [
        {"method": "Qwen3-ASR", "chime4": 5.39, "voices": 8.94, "noizeus": 9.45, "avg": 7.93},
        {"method": "Whisper-L-v3", "chime4": 7.02, "voices": 11.79, "noizeus": 13.34, "avg": 10.72},
        {"method": "Mega-ASR", "chime4": 5.23, "voices": 7.35, "noizeus": 7.52, "avg": 6.70},
        {"method": "Mega-ASR w/ router", "chime4": 5.00, "voices": 7.37, "noizeus": 7.90, "avg": 6.76},
    ]


def table_iv_vitw_bench_excerpt() -> list[dict[str, Any]]:
    """Table 4 excerpt — Mega-ASR on mixed real/sim splits (WER %)."""
    return [
        {"method": "Qwen3-ASR", "mixed_real": 5.39, "mixed_sim": 5.39},
        {"method": "Whisper-L-v3", "mixed_real": 14.79, "mixed_sim": 14.79},
        {"method": "Mega-ASR", "mixed_real": 2.73, "mixed_sim": 4.57},
    ]


def table_v_ablation() -> list[dict[str, Any]]:
    """Table 5 — A2S-SFT + DG-WGPO ablation (Voices / Noizeus mid+high)."""
    return [
        {"variant": "Qwen3-ASR baseline", "voices": 8.94, "noizeus": 9.45},
        {"variant": "+ SFT w/o A2S", "voices": 8.31, "noizeus": 8.79},
        {"variant": "Mega-ASR-Base", "voices": 7.59, "noizeus": 8.12},
        {"variant": "+ vanilla GRPO (R_wer only)", "voices": 7.73, "noizeus": 8.11},
        {"variant": "+ vanilla DAPO (R_wer only)", "voices": 7.62, "noizeus": 7.98},
        {"variant": "+ DG-WGPO w/o R_rep", "voices": 7.46, "noizeus": 7.73},
        {"variant": "+ DG-WGPO w/o R_fine", "voices": 7.45, "noizeus": 7.71},
        {"variant": "+ DG-WGPO w/o R_struc", "voices": 7.54, "noizeus": 7.85},
        {"variant": "+ DG-WGPO w/o gated fusion", "voices": 7.41, "noizeus": 7.68},
        {"variant": "Mega-ASR (full)", "voices": 7.35, "noizeus": 7.64},
    ]


def table_iii_standard_asr() -> list[dict[str, Any]]:
    """Table 3 excerpt — standard ASR benchmarks (WER %, lower better)."""
    return [
        {"method": "Qwen3-ASR-1.7B", "librispeech_dev_test": "1.62/3.40", "fleurs_zh_en": "3.93/3.19"},
        {"method": "Mega-ASR", "librispeech_dev_test": "1.78/3.57", "fleurs_zh_en": "5.43/3.76"},
        {"method": "Mega-ASR w/ router", "librispeech_dev_test": "1.63/3.37", "fleurs_zh_en": "3.86/3.17"},
    ]


def table_iv_vitw_bench_by_scenario() -> list[dict[str, Any]]:
    """Table 4 excerpt — Voices-in-the-Wild-Bench mixed real/sim (WER %)."""
    return [
        {"method": "Qwen3-ASR", "mixed_real": 3.30, "mixed_sim": 5.39},
        {"method": "Whisper-L-v3", "mixed_real": 8.91, "mixed_sim": 14.79},
        {"method": "Gemini3-Flash", "mixed_real": 7.99, "mixed_sim": 9.62},
        {"method": "Mega-ASR", "mixed_real": 2.73, "mixed_sim": 4.57},
        {"method": "Mega-ASR w/ router", "mixed_real": 2.63, "mixed_sim": 4.53},
    ]


def table_vi_reward_design() -> list[dict[str, Any]]:
    """Table 6 — rule-based vs LLM-judge reward (WER % and training time)."""
    return [
        {"reward": "LLM-judge", "voices": 7.51, "noizeus": 7.71, "avg_time_s": 62.23},
        {"reward": "Rule-based (DG-WGPO)", "voices": 7.53, "noizeus": 7.64, "avg_time_s": 19.57},
    ]


def table_vii_llm_judge_semantic() -> list[dict[str, Any]]:
    """Table 7 — semantic-level judge metrics (avg Voices + Noizeus)."""
    return [
        {"model": "Qwen3-ASR", "hallucination": 18.7, "missed_content": 14.2, "semantic": 71.3},
        {"model": "Mega-ASR-Base", "hallucination": 15.4, "missed_content": 11.6, "semantic": 79.8},
        {"model": "Mega-ASR", "hallucination": 11.8, "missed_content": 5.9, "semantic": 86.4},
    ]


def dataset_coverage_table() -> list[dict[str, Any]]:
    """Table 1 excerpt — VOICES-IN-THE-WILD-2M vs prior robust ASR datasets."""
    return [
        {"dataset": "NOIZEUS", "scale": "1K", "avg_wer_pct": 9.45},
        {"dataset": "VOiCES", "scale": "1M", "avg_wer_pct": 8.94},
        {"dataset": "VOICES-IN-THE-WILD-2M", "scale": "2M", "avg_wer_pct": 18.42},
    ]


def headline_results() -> dict[str, Any]:
    """Paper headline WER comparisons."""
    return {
        "voices_rm4_babb_far_wer_pct": {"baseline_qwen3": 54.01, "mega_asr": 45.69},
        "noizeus_station_0db_wer_pct": {"baseline_qwen3": 29.34, "mega_asr": 21.49},
        "relative_wer_reduction_compositional": ">30% vs strong baselines",
    }


def pipeline_demo(cfg: MegaASRConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MegaASRConfig()
    ref = "among export led electrical and computer makers japan victor company fell fifty"
    hyp_good = "among export led electrical and computer makers japan victor company fell fifty"
    hyp_bad = "among export led computer makers japan victor net sold fifty two thousand"
    rewards_good = dg_wgpo_reward(
        hyp_good,
        ref,
        tau=cfg.wer_gate_tau,
        soft_alpha=cfg.soft_error_alpha,
        alpha_dyn=cfg.dynamic_reward_weight,
    )
    rewards_bad = dg_wgpo_reward(
        hyp_bad,
        ref,
        tau=cfg.wer_gate_tau,
        soft_alpha=cfg.soft_error_alpha,
        alpha_dyn=cfg.dynamic_reward_weight,
    )
    return {
        "a2s_phases": len(a2s_sft_phases()),
        "reward_good": rewards_good,
        "reward_bad": rewards_bad,
        "learnability_filter_80pct_wer": True,
        "router_clean": route_decision(0.2),
        "router_degraded": route_decision(0.8),
    }


def evaluation_demo(cfg: MegaASRConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["mega_asr_avg_robust_wer"] = 6.70
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_dataset_coverage": dataset_coverage_table(),
        "table_ii_robust": table_i_robust_benchmarks(),
        "table_iii_standard_asr": table_iii_standard_asr(),
        "table_iv_bench_excerpt": table_iv_vitw_bench_excerpt(),
        "table_iv_bench_by_scenario": table_iv_vitw_bench_by_scenario(),
        "table_v_ablation": table_v_ablation(),
        "table_vi_reward_design": table_vi_reward_design(),
        "table_vii_llm_judge_semantic": table_vii_llm_judge_semantic(),
        "headline_results": headline_results(),
        "router": router_architecture_summary(),
        "atomic_phenomena": list(ATOMIC_PHENOMENA),
        "a2s_sft_phases": a2s_sft_phases(),
    }
