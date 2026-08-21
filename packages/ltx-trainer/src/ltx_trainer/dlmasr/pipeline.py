"""DLM-ASR decoding framework card and paper figures (arXiv:2605.29613)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dlmasr.config import DecodingStrategy, DlmAsrConfig
from ltx_trainer.dlmasr.decoding import block_decode_smoke
from ltx_trainer.dlmasr.layout import LIMITATIONS
from ltx_trainer.dlmasr.strategy_compare import compare_strategies, strategy_compare_smoke
from ltx_trainer.dlmasr.uncertainty import uncertainty_smoke


def framework_card(cfg: DlmAsrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DlmAsrConfig()
    return {
        "name": "DLM-ASR Decoding",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Systematic comparison of fixed-number, static-threshold, and dynamic-threshold "
            "decoding for diffusion language model ASR (Whisper-LLaDA). Threshold strategies "
            "exploit skewed ASR confidence to commit reliable tokens early."
        ),
        "baseline": {
            "system": cfg.baseline_system,
            "encoder": cfg.speech_encoder,
            "decoder": cfg.dlm_decoder,
            "train": cfg.train_corpus,
            "eval": cfg.eval_split,
        },
        "strategies": [s.value for s in DecodingStrategy if s != DecodingStrategy.AUTOREGRESSIVE],
        "metrics": ["WER", "RTF", "cumulative NLL", "per-round throughput"],
        "hyperparam_sweeps": {
            "fixed_k": [4, 8, 16, 32, 64, 128],
            "static_C": [0.8, 0.9, 0.95],
            "dynamic_f": [1.0, 0.2, 0.05],
        },
        "limitations": list(LIMITATIONS),
    }


def figure_i_wer_rtf_tradeoff() -> list[dict[str, Any]]:
    """Fig. 1 — WER–RTF trade-off anchors (selected points)."""
    cfg = DlmAsrConfig()
    return [
        {"strategy": "AR baseline", "block_size": 1, "wer_pct": cfg.ar_wer_pct, "rtf": 0.078, "note": "LLM AR reference"},
        {"strategy": "static", "block_size": 4, "C": 0.95, "wer_pct": cfg.static_b4_c095_wer_pct, "rtf": 0.046, "speedup_vs_ar": cfg.static_b4_c095_rtf_speedup},
        {"strategy": "static", "block_size": "L (parallel)", "C": 0.95, "wer_pct": cfg.static_parallel_wer_pct, "speedup_vs_ar": cfg.static_parallel_speedup},
        {"strategy": "dynamic", "block_size": 4, "f": 0.2, "wer_pct": 3.52, "rtf": 0.081},
        {"strategy": "fixed", "block_size": 4, "k": 8, "wer_pct": 4.47, "rtf": 0.298},
    ]


def figure_ii_matched_rtf_wer() -> list[dict[str, Any]]:
    """Fig. 2(d) — WER under matched RTF."""
    cfg = DlmAsrConfig()
    return [
        {"strategy": "static", "wer_pct": cfg.matched_rtf_static_wer_pct},
        {"strategy": "dynamic", "wer_pct": cfg.matched_rtf_dynamic_wer_pct},
        {"strategy": "fixed", "wer_pct": cfg.matched_rtf_fixed_wer_pct},
    ]


def figure_iii_throughput() -> list[dict[str, Any]]:
    """Fig. 3 — mean stopping rounds and RTF (comparable WER 3.05–3.10%)."""
    cfg = DlmAsrConfig()
    return [
        {"strategy": "static", "C": 0.95, "mean_rounds": cfg.static_mean_rounds, "rtf": cfg.static_rtf},
        {"strategy": "dynamic", "f": 0.2, "mean_rounds": cfg.dynamic_mean_rounds, "rtf": cfg.dynamic_rtf},
        {"strategy": "fixed", "k": 1, "mean_rounds": cfg.fixed_k1_mean_rounds, "rtf": cfg.fixed_rtf},
    ]


def figure_iv_confidence_ccdf() -> list[dict[str, Any]]:
    """Fig. 4 — fraction of tokens with confidence >= threshold."""
    cfg = DlmAsrConfig()
    return [
        {"domain": "ASR (LibriSpeech parallel)", "threshold": 0.90, "fraction_ge": cfg.asr_frac_ge_090},
        {"domain": "ASR (LibriSpeech parallel)", "threshold": 0.95, "fraction_ge": cfg.asr_frac_ge_095},
        {"domain": "GSM8K reasoning", "threshold": 0.90, "fraction_ge": cfg.gsm8k_frac_ge_090},
        {"domain": "GSM8K reasoning", "threshold": 0.95, "fraction_ge": cfg.gsm8k_frac_ge_095},
    ]


def headline_results() -> dict[str, Any]:
    cfg = DlmAsrConfig()
    return {
        "best_accuracy_speed": (
            f"Static C=0.95 B=4: {cfg.static_b4_c095_wer_pct}% WER vs AR {cfg.ar_wer_pct}%, "
            f"{cfg.static_b4_c095_rtf_speedup}× faster"
        ),
        "matched_rtf_winner": f"Static {cfg.matched_rtf_static_wer_pct}% WER (vs fixed {cfg.matched_rtf_fixed_wer_pct}%)",
        "confidence_skew": f"ASR {cfg.asr_frac_ge_095:.1%} tokens ≥0.95 vs GSM8K {cfg.gsm8k_frac_ge_095:.1%}",
        "fewest_rounds": f"Static {cfg.static_mean_rounds} mean rounds vs fixed k=1: {cfg.fixed_k1_mean_rounds}",
        "key_insight": "Heavy-start static thresholding harvests high-confidence ASR tokens early",
    }


def pipeline_demo(cfg: DlmAsrConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DlmAsrConfig()
    static = block_decode_smoke(32, strategy=DecodingStrategy.STATIC_THRESHOLD, cfg=cfg, seed=seed)
    fixed = block_decode_smoke(32, strategy=DecodingStrategy.FIXED_NUMBER, cfg=cfg, seed=seed)
    unc = uncertainty_smoke(32, seed=seed)
    strat = strategy_compare_smoke(cfg, seed=seed)
    return {
        "static_rounds": static["rounds"],
        "fixed_rounds": fixed["rounds"],
        "static_fewer_rounds": static["rounds"] < fixed["rounds"],
        "uncertainty_gap": unc["gap_at_full"],
        "ar_wer_anchor_pct": cfg.ar_wer_pct,
        "strategy_compare": strat,
        "best_strategy": strat["best_strategy"],
        "static_wer_pct": strat["static_wer_pct"],
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "figure_i_wer_rtf": figure_i_wer_rtf_tradeoff(),
        "figure_ii_matched_rtf": figure_ii_matched_rtf_wer(),
        "figure_iii_throughput": figure_iii_throughput(),
        "figure_iv_ccdf": figure_iv_confidence_ccdf(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
