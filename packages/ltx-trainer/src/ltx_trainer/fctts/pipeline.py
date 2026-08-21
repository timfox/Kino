"""FC-TTS framework card and paper tables (arXiv:2605.24618)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fctts.config import FcttsConfig
from ltx_trainer.fctts.layout import LIMITATIONS
from ltx_trainer.fctts.mock import evaluation_smoke


def framework_card(cfg: FcttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FcttsConfig()
    return {
        "name": "FC-TTS",
        "paper": cfg.paper_arxiv,
        "authors": "Yoonhyung Lee, Hyunsin Park, Jinhwan Park, Jinkyu Lee (Qualcomm AI Research)",
        "demo": cfg.demo_url,
        "idea": (
            "Zero-shot TTS with disentangled style (prosody cp) and timbre (z_spk) "
            "from two reference utterances via FACodec + two-stage CFM."
        ),
        "stages": [
            "Stage 1: timbre adapter → blurry log-mel h",
            "Stage 2: TCF style + flow-matching decoder → clean spectrogram",
        ],
        "innovations": [
            "Hierarchical two-stage spectrogram generation",
            "VQ-VAE TCF style encoder (phoneme + frame)",
            "Conditional consistency loss (CCL)",
        ],
        "conditioning": "cp + z_spk only (no FACodec cc/cd at inference)",
        "training": {
            "corpus": "LibriHeavy",
            "iterations": cfg.train_iterations,
            "params_m": cfg.num_params_m,
        },
        "inference": {
            "duration_nfe": cfg.duration_nfe,
            "mel_nfe": cfg.mel_nfe,
            "cfg_scale": cfg.cfg_scale,
        },
    }


def table_i_librispeech() -> list[dict[str, Any]]:
    """Table 1 — zero-shot TTS on LibriSpeech test-clean."""
    return [
        {"model": "Ground-truth", "utmos": 4.10, "wer": 2.07, "spk": 0.71, "params_m": None},
        {"model": "NaturalSpeech 3", "utmos": 4.30, "wer": 1.81, "spk": 0.67, "params_m": 500},
        {"model": "F5-TTS", "utmos": None, "wer": 2.42, "spk": 0.66, "params_m": 336},
        {"model": "F5-TTS (LibriHeavy retrain)", "utmos": 4.03, "wer": 3.30, "spk": 0.67, "params_m": 205},
        {"model": "DiTTo-TTS", "utmos": None, "wer": 2.69, "spk": 0.60, "params_m": 508},
        {"model": "CLaM-TTS", "utmos": None, "wer": 5.11, "spk": 0.50, "params_m": 584},
        {"model": "FC-TTS (ours)", "utmos": 4.22, "wer": 1.88, "spk": 0.60, "params_m": 204},
    ]


def table_ii_ravdess_timbre() -> list[dict[str, Any]]:
    """Table 2 — independent timbre control on RAVDESS."""
    return [
        {
            "model": "FACodec-VC",
            "utmos": 3.19,
            "spk": 0.27,
            "wer": 8.40,
            "win_pct": 10.7,
        },
        {
            "model": "FC-TTS (ours)",
            "utmos": 4.03,
            "spk": 0.48,
            "wer": 0.18,
            "win_pct": 66.1,
        },
    ]


def table_iii_ravdess_prosody() -> list[dict[str, Any]]:
    """Table 3 — independent style/prosody control on RAVDESS."""
    return [
        {
            "model": "F5-TTS",
            "utmos": 3.40,
            "spk": 0.57,
            "wer": 4.39,
            "mcd": 3.43,
            "win_pct": 8.9,
        },
        {
            "model": "FC-TTS (ours)",
            "utmos": 3.95,
            "spk": 0.47,
            "wer": 0.30,
            "mcd": 3.21,
            "win_pct": 65.5,
            "dual_reference": True,
        },
    ]


def table_iv_audiollm_judge() -> list[dict[str, Any]]:
    """Table 4 — Gemini 2.5 Pro style control evaluation."""
    return [
        {"model": "F5-TTS", "win_ratio_pct": 8.3, "style_mos": 1.50},
        {"model": "FC-TTS (ours)", "win_ratio_pct": 91.7, "style_mos": 3.92},
    ]


def table_v_ablation() -> list[dict[str, Any]]:
    """Table 5 — ablation on LibriSpeech and RAVDESS."""
    return [
        {
            "variant": "FC-TTS",
            "librispeech": {"utmos": 4.22, "wer": 1.88, "spk": 0.60, "mcd": 5.60},
            "ravdess": {"utmos": 3.91, "wer": 0.30, "spk": 0.37, "mcd": 3.33},
        },
        {
            "variant": "− two-stage generation",
            "librispeech": {"utmos": 4.15, "wer": 1.93, "spk": 0.60, "mcd": 5.83},
            "ravdess": {"utmos": 3.57, "wer": 0.30, "spk": 0.37, "mcd": 3.26},
        },
        {
            "variant": "− VQ-VAE style encoding",
            "librispeech": {"utmos": 4.25, "wer": 2.00, "spk": 0.57, "mcd": 5.62},
            "ravdess": {"utmos": 3.99, "wer": 0.25, "spk": 0.34, "mcd": 3.47},
        },
        {
            "variant": "− conditioning in CCL",
            "librispeech": {"utmos": 4.21, "wer": 1.92, "spk": 0.59, "mcd": 5.67},
            "ravdess": {"utmos": 3.79, "wer": 0.35, "spk": 0.36, "mcd": 3.36},
        },
        {
            "variant": "− entire consistency loss",
            "librispeech": {"utmos": 3.95, "wer": 5.88, "spk": 0.48, "mcd": 6.34},
            "ravdess": {"utmos": 3.70, "wer": 9.36, "spk": 0.21, "mcd": 3.75},
        },
    ]


def headline_results() -> dict[str, Any]:
    ours = next(r for r in table_i_librispeech() if "FC-TTS" in r["model"])
    t2 = next(r for r in table_ii_ravdess_timbre() if "FC-TTS" in r["model"])
    t4 = next(r for r in table_iv_audiollm_judge() if "FC-TTS" in r["model"])
    return {
        "finding": (
            "FC-TTS enables separate timbre and style references with competitive "
            "LibriSpeech zero-shot quality (UTMOS 4.22, WER 1.88) at 204M params."
        ),
        "timbre_win_pct_ravdess": t2["win_pct"],
        "style_win_pct_audiollm": t4["win_ratio_pct"],
        "style_mos": t4["style_mos"],
        "libri_utmos": ours["utmos"],
        "libri_wer": ours["wer"],
    }


def evaluation_demo(cfg: FcttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FcttsConfig()
    smoke = evaluation_smoke()
    return {"paper": cfg.paper_arxiv, "smoke": smoke, "demo_url": cfg.demo_url}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_librispeech": table_i_librispeech(),
        "table_ii_ravdess_timbre": table_ii_ravdess_timbre(),
        "table_iii_ravdess_prosody": table_iii_ravdess_prosody(),
        "table_iv_audiollm_judge": table_iv_audiollm_judge(),
        "table_v_ablation": table_v_ablation(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
