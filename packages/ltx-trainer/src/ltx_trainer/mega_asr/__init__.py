"""Mega-ASR — robust ASR in-the-wild (arXiv:2605.19833)."""

from ltx_trainer.mega_asr.a2s_sft import a2s_sft_phases, filter_learnability
from ltx_trainer.mega_asr.config import MegaASRConfig
from ltx_trainer.mega_asr.layout import LIMITATIONS
from ltx_trainer.mega_asr.pipeline import (
    benchmarks_bundle,
    dataset_coverage_table,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_robust_benchmarks,
    table_iii_standard_asr,
    table_iv_vitw_bench_by_scenario,
    table_iv_vitw_bench_excerpt,
    table_v_ablation,
    table_vi_reward_design,
    table_vii_llm_judge_semantic,
)
from ltx_trainer.mega_asr.router import route_decision, router_architecture_summary
from ltx_trainer.mega_asr.rewards import (
    ATOMIC_PHENOMENA,
    dg_wgpo_reward,
    dynamic_reward,
    static_reward,
    wer_reward,
    word_wer,
)

__all__ = [
    "ATOMIC_PHENOMENA",
    "LIMITATIONS",
    "MegaASRConfig",
    "a2s_sft_phases",
    "benchmarks_bundle",
    "dataset_coverage_table",
    "dg_wgpo_reward",
    "dynamic_reward",
    "evaluation_demo",
    "filter_learnability",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "route_decision",
    "router_architecture_summary",
    "static_reward",
    "table_i_robust_benchmarks",
    "table_iii_standard_asr",
    "table_iv_vitw_bench_by_scenario",
    "table_iv_vitw_bench_excerpt",
    "table_v_ablation",
    "table_vi_reward_design",
    "table_vii_llm_judge_semantic",
    "wer_reward",
    "word_wer",
]
