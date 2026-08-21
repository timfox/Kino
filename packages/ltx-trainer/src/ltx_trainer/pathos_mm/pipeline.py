"""Framework card and paper benchmark excerpts (arXiv:2605.22732)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pathos_mm.config import PathosMmConfig
from ltx_trainer.pathos_mm.layout import LIMITATIONS
from ltx_trainer.pathos_mm.mock import evaluation_smoke
from ltx_trainer.pathos_mm.russell import E2V_RUSSELL_WEIGHTS


def framework_card(cfg: PathosMmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PathosMmConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "author": "Jürgen Dietrich (Democracy Intelligence gGmbH)",
        "modalities": {
            "emotion2vec": "emotion2vec_plus_large → 8 classes → Russell A/V projection",
            "gemini": "Gemini 2.5 Flash multimodal (audio + transcript), open-ended labels",
            "trust_pathos": "TRUST three-advocate LLM supervisor, median consensus",
        },
        "case_study": {
            "speaker": "Felix Banaszak (Bundestag plenary, 5 Mar 2026)",
            "segments_total": cfg.banaszak_segments_total,
            "segments_analyzed": cfg.banaszak_segments_analyzed,
        },
        "headlines": headline_results(),
        "limitations": LIMITATIONS,
    }


def table1_russell_weights() -> list[dict[str, Any]]:
    """Table 1: emotion2vec class → Russell Arousal/Valence weights."""
    return [
        {"class": cls, "w_arousal": wa, "w_valence": wv}
        for cls, (wa, wv) in E2V_RUSSELL_WEIGHTS.items()
    ]


def table2_emo_db_gemini_match() -> list[dict[str, Any]]:
    """Table 2: Gemini open-ended annotation on EMO-DB."""
    return [
        {"emotion": "Neutral", "n": 79, "match_pct": 65.8, "avg_conf": 0.83},
        {"emotion": "Sadness", "n": 62, "match_pct": 35.5, "avg_conf": 0.80},
        {"emotion": "Happiness", "n": 71, "match_pct": 29.6, "avg_conf": 0.83},
        {"emotion": "Anger", "n": 127, "match_pct": 29.1, "avg_conf": 0.86},
        {"emotion": "Fear", "n": 69, "match_pct": 27.5, "avg_conf": 0.77},
        {"emotion": "Boredom", "n": 81, "match_pct": 12.3, "avg_conf": 0.81},
        {"emotion": "Disgust", "n": 46, "match_pct": 0.0, "avg_conf": 0.81},
        {"emotion": "Total", "n": 535, "match_pct": 30.1, "avg_conf": 0.82},
    ]


def table3_banaszak_descriptive() -> list[dict[str, Any]]:
    """Table 3: descriptive statistics Banaszak speech."""
    return [
        {"measure": "Gemini Arousal", "mean": 0.59, "sd": 0.28, "min": 0.00, "max": 1.00},
        {"measure": "Gemini Valence", "mean": -0.56, "sd": 0.44, "min": -1.00, "max": 0.60},
        {"measure": "emotion2vec Arousal", "mean": 0.36, "sd": 0.21, "min": 0.04, "max": 0.75},
        {"measure": "emotion2vec Valence", "mean": 0.04, "sd": 0.32, "min": -0.74, "max": 0.78},
        {"measure": "TRUST-Pathos", "mean": -0.37, "sd": 0.56, "min": -2.00, "max": 1.00},
    ]


def table4_spearman_correlations() -> list[dict[str, Any]]:
    """Table 4: Spearman ρ between modalities and TRUST-Pathos."""
    return [
        {"comparison": "Gemini Valence ↔ TRUST-Pathos", "rho": 0.664, "p": "<0.001"},
        {"comparison": "Gemini Arousal ↔ TRUST-Pathos", "rho": -0.535, "p": "<0.001"},
        {"comparison": "e2v Valence ↔ TRUST-Pathos", "rho": 0.097, "p": "0.499"},
        {"comparison": "e2v Arousal ↔ TRUST-Pathos", "rho": -0.155, "p": "0.278"},
        {"comparison": "e2v Arousal ↔ Gemini Arousal", "rho": 0.239, "p": "0.091"},
        {"comparison": "e2v Valence ↔ Gemini Valence", "rho": 0.200, "p": "0.159"},
    ]


def banaszak_rhetoric_distribution() -> list[dict[str, Any]]:
    """Section 4.2.4: Gemini rhetorical function counts (51 segments)."""
    return [
        {"function": "Criticism", "n": 16, "pct": 31.0},
        {"function": "Sarcasm", "n": 14, "pct": 27.0},
        {"function": "None", "n": 9, "pct": 18.0},
        {"function": "Appeal", "n": 7, "pct": 14.0},
        {"function": "Metaphor", "n": 2, "pct": 4.0},
        {"function": "Accusation", "n": 1, "pct": 2.0},
        {"function": "Rhetorical Question", "n": 1, "pct": 2.0},
        {"function": "Indignation", "n": 1, "pct": 2.0},
    ]


def headline_results() -> dict[str, Any]:
    cfg = PathosMmConfig()
    return {
        "gemini_valence_trust_rho": cfg.rho_gemini_valence_trust,
        "e2v_valence_trust_rho": cfg.rho_e2v_valence_trust,
        "emo_db_gemini_match_pct": cfg.emo_db_overall_match_pct,
        "llm_beats_acoustic_for_pathos_proxy": True,
        "acoustic_valence_not_significant_with_trust": True,
    }


def evaluation_demo(cfg: PathosMmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PathosMmConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_russell_weights": table1_russell_weights(),
        "table2_emo_db_gemini": table2_emo_db_gemini_match(),
        "table3_banaszak_descriptive": table3_banaszak_descriptive(),
        "table4_spearman": table4_spearman_correlations(),
        "rhetoric_distribution": banaszak_rhetoric_distribution(),
        "headlines": headline_results(),
    }
