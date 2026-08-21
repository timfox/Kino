"""Framework card and benchmark excerpts — Raon-OpenTTS (arXiv:2605.20830)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.raon_opentts.config import RaonOpenTtsConfig
from ltx_trainer.raon_opentts.layout import LIMITATIONS
from ltx_trainer.raon_opentts.mock import evaluation_smoke


def table2_pool_composition_excerpt() -> list[dict[str, Any]]:
    """Table 2 excerpt — largest sources in Raon-OpenTTS-Pool (hours, M segments)."""
    return [
        {"dataset": "Raon-YouTube-Commons", "hours_k": 335, "segments_m": 141.7, "dns": 2.74, "wer": 0.30},
        {"dataset": "Emilia-YODAS", "hours_k": 92, "segments_m": 36.0, "dns": 2.82, "wer": 0.19},
        {"dataset": "Emilia", "hours_k": 47, "segments_m": 18.1, "dns": 3.02, "wer": 0.18},
        {"dataset": "LibriHeavy", "hours_k": 42, "segments_m": 10.8, "dns": 3.22, "wer": 0.11},
        {"dataset": "Total / Avg.", "hours_k": 615, "segments_m": 239.7, "dns": 2.83, "wer": 0.24},
    ]


def table3_filtering_ablation_excerpt() -> list[dict[str, Any]]:
    """Table 3 excerpt — Seed-TTS-Eval and overall rank for filtering strategies (0.3B ablation)."""
    return [
        {"filtering": "Combined (15%)", "seed_wer_pct": 2.00, "seed_sim": 0.672, "overall_rank": 3.40},
        {"filtering": "DNSMOS (15%)", "seed_wer_pct": 1.97, "seed_sim": 0.669, "overall_rank": 3.60},
        {"filtering": "WER (15%)", "seed_wer_pct": 1.99, "seed_sim": 0.668, "overall_rank": 4.50},
        {"filtering": "No filtering", "seed_wer_pct": 2.19, "seed_sim": 0.661, "overall_rank": 4.90},
        {"filtering": "WER (50%)", "seed_wer_pct": 2.20, "seed_sim": 0.655, "overall_rank": 7.60},
    ]


def table4_core_retention_excerpt() -> list[dict[str, Any]]:
    """Table 4 excerpt — segment retention after combined 15th-percentile filter."""
    return [
        {"dataset": "LibriTTS-R", "core_segments_m": 0.3, "retention_pct": 97.7},
        {"dataset": "HiFiTTS2", "core_segments_m": 11.9, "retention_pct": 94.5},
        {"dataset": "Raon-YouTube-Commons", "core_segments_m": 117.5, "retention_pct": 82.9},
        {"dataset": "People's Speech (Dirty)", "core_segments_m": 2.6, "retention_pct": 48.2},
        {"dataset": "Total", "core_segments_m": 193.9, "retention_pct": 84.7},
    ]


def table1_seed_tts_eval_subset() -> list[dict[str, Any]]:
    """Table 1 excerpt — Seed-TTS-Eval WER (%) and SIM for selected open-data baselines and Raon."""
    return [
        {"model": "F5-TTS", "params": "0.3B", "open_data": True, "wer_pct": 2.04, "sim": 0.671},
        {"model": "MaskGCT", "params": "0.6B", "open_data": True, "wer_pct": 2.57, "sim": 0.713},
        {"model": "Raon-OpenTTS-0.3B", "params": "0.3B", "open_data": True, "wer_pct": 1.95, "sim": 0.687},
        {"model": "Raon-OpenTTS-1B", "params": "1.0B", "open_data": True, "wer_pct": 1.78, "sim": 0.749},
        {"model": "Qwen3-TTS", "params": "1.7B", "open_data": False, "wer_pct": 1.46, "sim": 0.715},
    ]


def table5_cv3_excerpt() -> list[dict[str, Any]]:
    """Table 5 excerpt — CV3-EN WER (%) and CV3-Hard-EN WER / SIM / DNSMOS."""
    return [
        {"model": "F5-TTS", "cv3_en_wer_pct": 8.54, "hard_wer_pct": None, "hard_sim": None, "hard_dnsmos": None},
        {"model": "MaskGCT", "cv3_en_wer_pct": 7.73, "hard_wer_pct": 41.09, "hard_sim": 0.624, "hard_dnsmos": 3.48},
        {"model": "Raon-OpenTTS-0.3B", "cv3_en_wer_pct": 4.62, "hard_wer_pct": 7.31, "hard_sim": 0.730, "hard_dnsmos": 3.77},
        {"model": "Raon-OpenTTS-1B", "cv3_en_wer_pct": 3.92, "hard_wer_pct": 6.15, "hard_sim": 0.775, "hard_dnsmos": 3.85},
        {"model": "Qwen3-TTS", "cv3_en_wer_pct": 4.52, "hard_wer_pct": 7.89, "hard_sim": 0.666, "hard_dnsmos": 3.87},
    ]


def table6_raon_eval_overall() -> list[dict[str, Any]]:
    """Table 6 excerpt — Raon-OpenTTS-Eval overall WER (%) and SIM."""
    return [
        {"model": "F5-TTS", "overall_wer_pct": 25.08, "overall_sim": 0.542},
        {"model": "MaskGCT", "overall_wer_pct": 8.61, "overall_sim": 0.635},
        {"model": "CosyVoice 3", "overall_wer_pct": 4.43, "overall_sim": 0.647},
        {"model": "Qwen3-TTS", "overall_wer_pct": 17.59, "overall_sim": 0.626},
        {"model": "Raon-OpenTTS-0.3B", "overall_wer_pct": 2.93, "overall_sim": 0.623},
        {"model": "Raon-OpenTTS-1B", "overall_wer_pct": 2.81, "overall_sim": 0.695},
    ]


def filtering_thresholds_card() -> dict[str, float]:
    """15th-percentile tail cutoffs from Figure 2 (paper text)."""
    cfg = RaonOpenTtsConfig()
    return {
        "dnsmos_min": cfg.filter_dnsmos_min,
        "speech_ratio_min": cfg.filter_speech_ratio_min,
        "wer_max": cfg.filter_wer_max,
    }


def framework_card(cfg: RaonOpenTtsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RaonOpenTtsConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": "Semin Kim, Seungjun Chung, Taehong Moon, Sangheon Lee, Minyoung Ahn, Keon Lee, Nam Soo Kim, Jaewoong Cho, Ludwig Schmidt, Kangwook Lee, Dongmin Park (KRAFTON, SNU, and affiliates)",
        "repo": cfg.repo,
        "problem": "Closed-data SOTA TTS is not reproducible; need open pool, curation, weights, and robust eval.",
        "datasets": {
            "pool": f"~{cfg.pool_hours_k:.0f}k h English, ~{cfg.pool_segments_m:.0f}M segments from 11 public sources plus Raon-YouTube-Commons.",
            "core": f"Filtered ~{cfg.core_hours_k:.0f}k h, ~{cfg.core_segments_m:.0f}M segments via Whisper WER, DNSMOS, Silero speech ratio; combined 15% tail removal ranks best in Table 3 ablation.",
            "eval": f"Raon-OpenTTS-Eval: {cfg.eval_prompt_pairs} prompt–text pairs across {len(cfg.eval_regimes)} regimes (Clean, Noisy, Wild, Expressive) from 12 public sets.",
        },
        "models": "DiT TTS following F5-TTS architecture; Raon-OpenTTS-0.3B and 1B trained on Raon-OpenTTS-Core; 16 kHz, 80-dim log-mel hop 256, char vocab 5512, HiFi-GAN vocoder, 32 NFE ODE steps.",
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: RaonOpenTtsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RaonOpenTtsConfig()
    return {
        "seed_tts_eval_1b_wer_pct": cfg.seed_eval_raon_1b_wer_pct,
        "seed_tts_eval_1b_sim": cfg.seed_eval_raon_1b_sim,
        "cv3_hard_1b_wer_pct": cfg.cv3_hard_raon_1b_wer_pct,
        "cv3_hard_1b_sim": cfg.cv3_hard_raon_1b_sim,
        "raon_eval_1b_overall_wer_pct": cfg.raon_eval_1b_overall_wer_pct,
        "raon_eval_1b_overall_sim": cfg.raon_eval_1b_overall_sim,
    }


def evaluation_demo(cfg: RaonOpenTtsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RaonOpenTtsConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: RaonOpenTtsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RaonOpenTtsConfig()
    return {
        "framework": framework_card(cfg),
        "table1_seed_tts_eval": table1_seed_tts_eval_subset(),
        "table2_pool_composition": table2_pool_composition_excerpt(),
        "table3_filtering_ablation": table3_filtering_ablation_excerpt(),
        "table4_core_retention": table4_core_retention_excerpt(),
        "table5_cv3": table5_cv3_excerpt(),
        "table6_raon_eval_overall": table6_raon_eval_overall(),
        "filtering_thresholds": filtering_thresholds_card(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
