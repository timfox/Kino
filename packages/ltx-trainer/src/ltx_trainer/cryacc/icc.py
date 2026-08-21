"""ICC agreement classification and Table I anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cryacc.config import CryAccConfig, VocalMeasure


def icc_rating(icc: float) -> str:
    """Koo & Li guidelines (paper §II-D)."""
    if icc > 0.90:
        return "excellent"
    if icc >= 0.75:
        return "good"
    if icc >= 0.50:
        return "moderate"
    return "poor"


def table_i_icc(cfg: CryAccConfig | None = None) -> list[dict[str, Any]]:
    """Table I — ICC(A,1) and ICC(C,1) between MIC and ACC."""
    cfg = cfg or CryAccConfig()
    rows = [
        ("F0 (Hz)", cfg.icc_f0, 0.950, 0.942, 0.950, 0.954, 0.954),
        ("JCV (%)", cfg.icc_jcv, 0.958, 0.959, 0.965, 0.919, 0.935),
        ("Jlocal (%)", cfg.icc_jlocal, 0.872, 0.903, 0.901, 0.819, 0.817),
        ("SCV (%)", cfg.icc_scv, 0.647, 0.208, 0.700, 0.154, 0.577),
        ("Slocal (%)", cfg.icc_slocal, 0.601, 0.320, 0.594, 0.309, 0.589),
        ("CPP (dB)", cfg.icc_cpp, 0.586, 0.598, 0.593, 0.573, 0.584),
        ("HNR (dB)", cfg.icc_hnr, 0.610, 0.437, 0.638, 0.378, 0.572),
    ]
    out: list[dict[str, Any]] = []
    for name, a_all, c_all, a4, c4, a12, c12 in rows:
        out.append(
            {
                "measure": name,
                "ICC_A1_overall": a_all,
                "ICC_C1_overall": c_all,
                "ICC_A1_4mo": a4,
                "ICC_C1_4mo": c4,
                "ICC_A1_12mo": a12,
                "ICC_C1_12mo": c12,
                "rating_overall": icc_rating(a_all),
            }
        )
    return out


def table_ii_bias(cfg: CryAccConfig | None = None) -> list[dict[str, Any]]:
    """Table II — ACC minus MIC bias."""
    cfg = cfg or CryAccConfig()
    return [
        {"measure": "SCV (pp)", "bias_4m": cfg.bias_scv_4m_pp, "p_4m": 7.87e-21, "bias_12m": -6.603, "p_12m": 1.32e-20},
        {"measure": "Slocal (pp)", "bias_4m": cfg.bias_slocal_4m_pp, "p_4m": 1.27e-11, "bias_12m": -3.250, "p_12m": 1.17e-12},
        {"measure": "CPP (dB)", "bias_4m": cfg.bias_cpp_4m_db, "p_4m": 0.751, "bias_12m": -0.430, "p_12m": 0.090},
        {"measure": "HNR (dB)", "bias_4m": cfg.bias_hnr_4m_db, "p_4m": 7.20e-9, "bias_12m": 5.110, "p_12m": 4.03e-9},
    ]


def icc_smoke(cfg: CryAccConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    from ltx_trainer.cryacc.icc_compute import icc_compute_smoke

    cfg = cfg or CryAccConfig()
    computed = icc_compute_smoke(seed=seed, cfg=cfg)
    table = computed["computed_table"]
    excellent = [r for r in table if r["rating_overall"] == "excellent"]
    f0 = next(r for r in table if r["measure"].startswith("F0"))
    scv = next(r for r in table if r["measure"].startswith("SCV"))
    jcv = next(r for r in table if "JCV" in r["measure"])
    return {
        "n_measures": len(table),
        "f0_excellent": f0["rating_overall"] == "excellent",
        "f0_icc_a1_computed": f0["ICC_A1_overall"],
        "n_excellent_absolute": len(excellent),
        "scv_poor": scv["rating_overall"] == "poor",
        "jcv_good_or_better": icc_rating(jcv["ICC_A1_overall"]) in {"excellent", "good"},
        "computed_from_cohort": True,
    }
