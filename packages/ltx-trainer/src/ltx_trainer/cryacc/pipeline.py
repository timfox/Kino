"""CryAcc framework card and paper tables (arXiv:2605.28687)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cryacc.config import CryAccConfig, CryLabel
from ltx_trainer.cryacc.features import features_smoke, measure_registry
from ltx_trainer.cryacc.icc import icc_smoke, table_i_icc, table_ii_bias
from ltx_trainer.cryacc.layout import LIMITATIONS


def annotation_labels() -> list[dict[str, str]]:
    return [
        {"label": CryLabel.CRY_ONLY.value, "description": "Distinct infant cry without background noise"},
        {"label": CryLabel.CRY_NOISE.value, "description": "Cry overlapping speech or ambient noise"},
        {"label": CryLabel.NON_CRY.value, "description": "No distinct cry audio"},
    ]


def framework_card(cfg: CryAccConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CryAccConfig()
    return {
        "name": "CryAcc",
        "paper": cfg.paper_arxiv,
        "venue": cfg.venue,
        "idea": (
            "Cross-modal validation of chest-surface accelerometer (ACC) vs microphone (MIC) "
            "for infant cry vocal function during vaccination visits. Seven measures "
            "(F0, jitter, shimmer, CPP, HNR) compared via ICC(A,1) and ICC(C,1)."
        ),
        "cohort": {
            "n_total": cfg.n_infants_total,
            "n_4_month": cfg.n_4_month,
            "n_12_month": cfg.n_12_month,
            "irb": cfg.irb,
        },
        "recording": {
            "acc_sensor": cfg.acc_sensor,
            "acc_hz": cfg.acc_sample_rate_hz,
            "mic_hz": cfg.mic_sample_rate_hz,
            "window_ms": cfg.window_ms,
            "f0_range_hz": [cfg.f0_floor_hz, cfg.f0_ceiling_hz],
        },
        "measures": measure_registry(cfg),
        "annotation_labels": annotation_labels(),
        "limitations": list(LIMITATIONS),
    }


def headline_results(cfg: CryAccConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CryAccConfig()
    return {
        "f0_icc": f"F0 ICC(A,1)={cfg.icc_f0:.3f} (excellent, >0.94 both age groups)",
        "jitter_icc": f"JCV ICC={cfg.icc_jcv:.3f}, Jlocal ICC={cfg.icc_jlocal:.3f} (good–excellent)",
        "shimmer_icc": f"SCV/Slocal poor absolute agreement (ICC {cfg.icc_scv:.2f}/{cfg.icc_slocal:.2f}), ACC lower bias",
        "hnr_bias": f"HNR ACC +{cfg.bias_hnr_4m_db:.1f} dB vs MIC (moderate consistency, systematic bias)",
        "clinical_takeaway": (
            "Chest-surface ACC reliably captures timing-based F0 and jitter for scalable, "
            "privacy-preserving infant cry biomarker research"
        ),
    }


def pipeline_demo(cfg: CryAccConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or CryAccConfig()
    icc_out = icc_smoke(cfg, seed=seed)
    return {
        "n_measures": len(measure_registry(cfg)),
        "n_infants": cfg.n_infants_total,
        "features": features_smoke(seed=seed),
        "icc": icc_out,
        "f0_icc_computed": icc_out.get("f0_icc_a1_computed"),
        "f0_icc_anchor": cfg.icc_f0,
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "measures": measure_registry(),
        "annotation_labels": annotation_labels(),
        "table_i_icc": table_i_icc(),
        "table_ii_bias": table_ii_bias(),
        "headline": headline_results(),
    }
