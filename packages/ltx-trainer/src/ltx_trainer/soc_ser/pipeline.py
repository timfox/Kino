"""Framework card, Table 1, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.soc_ser.config import SocSerConfig
from ltx_trainer.soc_ser.layer import soc_demo


def framework_card(cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "repo": c.repo,
        "task": "self_supervised_speech_emotion_recognition",
        "method": "second_order_correlation + log_euclidean_mapping",
        "ssl_backbones": list(c.ssl_backbones),
        "ssl_dim": c.ssl_dim,
        "default_subspace_d": c.default_subspace_d,
        "datasets": list(c.datasets),
        "eval_protocol": c.eval_protocol,
        "baselines": list(c.baselines),
        "headline": headline_results(c),
    }


def headline_results(cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    return {
        "hubert_soc_esd_wa": c.hubert_soc_esd_wa,
        "hubert_soc_ravdess_wa": c.hubert_soc_ravdess_wa,
        "w2v_soc_esd_gain_vs_gap_pct": c.w2v_esd_gain_pct,
        "w2v_soc_ravdess_gain_vs_gap_pct": c.w2v_ravdess_gain_pct,
        "wavlm_ravdess_gain_vs_fa_pct": c.wavlm_ravdess_gain_vs_fa_pct,
    }


def table1_esd_ravdess() -> list[dict[str, Any]]:
    """Table 1 — ESD / RAVDESS WA across SSL backbones (paper anchors)."""
    rows: list[dict[str, Any]] = []

    def add(backbone: str, method: str, esd_wa: float, rav_wa: float, **extra: float) -> None:
        row: dict[str, Any] = {
            "backbone": backbone,
            "method": method,
            "esd_wa": esd_wa,
            "ravdess_wa": rav_wa,
        }
        row.update(extra)
        rows.append(row)

    # Wav2Vec 2.0
    add("Wav2Vec2", "GAP", 67.18, 54.25)
    add("Wav2Vec2", "ASP", 63.83, 52.50)
    add("Wav2Vec2", "FA", 68.94, 56.27)
    add("Wav2Vec2", "SOC (w/o LEM)", 68.30, 55.42)
    add("Wav2Vec2", "SOC", 71.86, 58.67, esd_f1=71.23, ravdess_f1=57.96)

    # HuBERT
    add("HuBERT", "GAP", 71.38, 65.24)
    add("HuBERT", "ASP", 65.32, 62.50)
    add("HuBERT", "FA", 72.48, 66.92)
    add("HuBERT", "SOC (w/o LEM)", 72.05, 68.10)
    add("HuBERT", "SOC", 73.50, 69.75, esd_f1=72.82, ravdess_f1=69.61)

    # WavLM
    add("WavLM", "GAP", 69.49, 60.83)
    add("WavLM", "ASP", 66.71, 63.45)
    add("WavLM", "FA", 71.12, 66.25)
    add("WavLM", "SOC (w/o LEM)", 70.82, 67.30)
    add("WavLM", "SOC", 72.61, 68.74, esd_f1=71.48, ravdess_f1=70.87)

    return rows


def lem_ablation_delta(cfg: SocSerConfig | None = None) -> dict[str, float]:
    """§3.4 LEM ablation on HuBERT (Table 1)."""
    c = cfg or SocSerConfig()
    return {
        "esd_wa_drop_pct": c.hubert_soc_esd_wa - c.hubert_soc_wo_lem_esd_wa,
        "expected_esd_drop_pct": 1.45,
    }


def experimental_protocol(cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    return {
        "metrics": ["weighted_accuracy", "unweighted_accuracy", "macro_f1"],
        "folds": {"ESD": c.esd_folds, "RAVDESS": c.ravdess_folds},
        "optimizer": "AdamW",
        "peak_lr": c.peak_lr,
        "epochs": c.epochs,
        "batch_size": {"ESD": c.batch_size_esd, "RAVDESS": c.batch_size_ravdess},
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {"table1_esd_ravdess": table1_esd_ravdess()}


def evaluation_demo(seed: int = 42, cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    demo = soc_demo(seed=seed, cfg=c)
    hubert_soc = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "HuBERT" and r["method"] == "SOC"
    )
    w2v_gap = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "Wav2Vec2" and r["method"] == "GAP"
    )
    w2v_soc = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "Wav2Vec2" and r["method"] == "SOC"
    )
    gain_esd = w2v_soc["esd_wa"] - w2v_gap["esd_wa"]
    return {
        "soc_layer": demo,
        "lem_ablation": lem_ablation_delta(c),
        "hubert_peak_esd_wa": hubert_soc["esd_wa"],
        "w2v_esd_gain": gain_esd,
        "beats_gap_on_w2v_esd": gain_esd >= c.w2v_esd_gain_pct - 0.01,
        "lem_required": lem_ablation_delta(c)["esd_wa_drop_pct"] > 0,
    }


def pipeline_demo(seed: int = 42, cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    return {
        "framework": framework_card(c),
        "protocol": experimental_protocol(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
        "benchmarks": benchmarks_bundle(),
    }
