"""Raon-OpenTTS open data and DiT TTS stub (arXiv:2605.20830)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaonOpenTtsConfig:
    paper_arxiv: str = "arXiv:2605.20830"
    title: str = "Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech"
    repo: str = "https://github.com/krafton-ai/RAON-OpenTTS"

    # Data scale (abstract + Sec. 3.1 / 3.2)
    pool_hours_k: float = 615.0
    pool_segments_m: float = 240.0
    core_hours_k: float = 510.0
    core_segments_m: float = 194.0

    # Sec. 3.2 — 15th-percentile removal thresholds (Figure 2)
    filter_dnsmos_min: float = 2.24
    filter_speech_ratio_min: float = 0.79
    filter_wer_max: float = 0.35

    # Raon-OpenTTS-Eval
    eval_prompt_pairs: int = 6000
    eval_regimes: tuple[str, ...] = ("Clean", "Noisy", "Wild", "Expressive")

    # Table 1 — Seed-TTS-Eval (percent WER as in paper table)
    seed_eval_raon_1b_wer_pct: float = 1.78
    seed_eval_raon_1b_sim: float = 0.749
    seed_eval_raon_03b_wer_pct: float = 1.95
    seed_eval_raon_03b_sim: float = 0.687

    # Table 5 — CV3-Hard-EN
    cv3_hard_raon_1b_wer_pct: float = 6.15
    cv3_hard_raon_1b_sim: float = 0.775

    # Table 6 — Raon-OpenTTS-Eval overall
    raon_eval_1b_overall_wer_pct: float = 2.81
    raon_eval_1b_overall_sim: float = 0.695
