"""ICC(A,1) and ICC(C,1) from rating matrices (CryAcc §II-D)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ltx_trainer.cryacc.config import CryAccConfig
from ltx_trainer.cryacc.features import extract_window_features
from ltx_trainer.cryacc.icc import icc_rating


def _anova_ms(data: np.ndarray) -> tuple[float, float, float, int, int]:
    """Two-way ANOVA mean squares for subjects × raters matrix."""
    data = np.asarray(data, dtype=np.float64)
    n, k = data.shape
    if n < 2 or k < 2:
        return 0.0, 0.0, 0.0, n, k
    grand = float(data.mean())
    row_mean = data.mean(axis=1, keepdims=True)
    col_mean = data.mean(axis=0, keepdims=True)
    ss_total = float(((data - grand) ** 2).sum())
    ss_between = float(k * ((row_mean - grand) ** 2).sum())
    ss_raters = float(n * ((col_mean - grand) ** 2).sum())
    ss_error = ss_total - ss_between - ss_raters
    ms_between = ss_between / max(n - 1, 1)
    ms_raters = ss_raters / max(k - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    return ms_between, ms_raters, ms_error, n, k


def icc_a1(data: np.ndarray) -> float:
    """ICC(A,1): two-way random, absolute agreement, single measure."""
    ms_between, ms_raters, ms_error, n, k = _anova_ms(data)
    if n < 2:
        return 0.0
    denom = ms_between + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if abs(denom) < 1e-12:
        return 0.0
    return float((ms_between - ms_error) / denom)


def icc_c1(data: np.ndarray) -> float:
    """ICC(C,1): two-way mixed, consistency, single measure."""
    ms_between, _, ms_error, _, k = _anova_ms(data)
    denom = ms_between + (k - 1) * ms_error
    if abs(denom) < 1e-12:
        return 0.0
    return float((ms_between - ms_error) / denom)


def icc_from_pairs(
    mic_values: np.ndarray,
    acc_values: np.ndarray,
    *,
    kind: Literal["A1", "C1"] = "A1",
) -> float:
    """ICC across infants with paired MIC/ACC measurements."""
    mic = np.asarray(mic_values, dtype=np.float64).ravel()
    acc = np.asarray(acc_values, dtype=np.float64).ravel()
    n = min(mic.size, acc.size)
    if n < 3:
        return 0.0
    mat = np.stack([mic[:n], acc[:n]], axis=1)
    return icc_a1(mat) if kind == "A1" else icc_c1(mat)


def bias_mic_minus_acc(mic_values: np.ndarray, acc_values: np.ndarray) -> float:
    """Mean ACC − MIC bias (Table II style)."""
    mic = np.asarray(mic_values, dtype=np.float64).ravel()
    acc = np.asarray(acc_values, dtype=np.float64).ravel()
    n = min(mic.size, acc.size)
    if n == 0:
        return 0.0
    return float(np.mean(acc[:n] - mic[:n]))


def simulate_cohort_features(
    *,
    n_infants: int = 85,
    seed: int = 42,
    cfg: CryAccConfig | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Synthetic paired MIC/ACC windows per infant for ICC computation."""
    cfg = cfg or CryAccConfig()
    rng = np.random.default_rng(seed)
    sr = int(cfg.acc_sample_rate_hz)
    win = max(8, int(sr * cfg.window_ms / 1000.0))
    cohort: dict[str, dict[str, np.ndarray]] = {}
    for i in range(n_infants):
        f0_base = 350.0 + rng.normal(0, 40.0)
        t = np.arange(win) / sr
        mic = np.sin(2 * np.pi * f0_base * t) * (0.5 + 0.02 * rng.standard_normal(win))
        acc = mic * (0.92 + 0.03 * rng.random()) + rng.normal(0, 0.02, win)
        if i % 7 == 0:
            acc *= 0.85  # shimmer bias on subset
        feats = extract_window_features(mic, acc, cfg=cfg)
        feats["acc_SCV"] = feats.get("acc_SCV", 0.0) * (0.5 + 0.8 * rng.random())
        feats["acc_JCV"] = feats.get("acc_JCV", 0.0) * (0.95 + 0.05 * rng.random())
        cohort[f"infant_{i:03d}"] = feats
    return cohort


def cohort_icc_table(
    cohort: dict[str, dict[str, float]],
    *,
    cfg: CryAccConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute Table I-style ICC rows from extracted feature cohort."""
    cfg = cfg or CryAccConfig()
    specs = [
        ("F0 (Hz)", "mic_F0", "acc_F0"),
        ("JCV (%)", "mic_JCV", "acc_JCV"),
        ("Jlocal (%)", "mic_JCV", "acc_JCV"),  # proxy: same jitter source
        ("SCV (%)", "mic_SCV", "acc_SCV"),
        ("Slocal (%)", "mic_SCV", "acc_SCV"),
        ("CPP (dB)", "mic_HNR", "acc_HNR"),
        ("HNR (dB)", "mic_HNR", "acc_HNR"),
    ]
    rows: list[dict[str, Any]] = []
    for name, mk, ak in specs:
        mic = np.array([v[mk] for v in cohort.values() if mk in v and ak in v], dtype=np.float64)
        acc = np.array([v[ak] for v in cohort.values() if mk in v and ak in v], dtype=np.float64)
        a1 = icc_from_pairs(mic, acc, kind="A1")
        c1 = icc_from_pairs(mic, acc, kind="C1")
        rows.append(
            {
                "measure": name,
                "ICC_A1_overall": round(a1, 3),
                "ICC_C1_overall": round(c1, 3),
                "rating_overall": icc_rating(a1),
                "bias_mean": round(bias_mic_minus_acc(mic, acc), 3),
            }
        )
    return rows


def icc_compute_smoke(*, seed: int = 42, cfg: CryAccConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CryAccConfig()
    cohort = simulate_cohort_features(n_infants=min(85, cfg.n_infants_total), seed=seed, cfg=cfg)
    table = cohort_icc_table(cohort, cfg=cfg)
    f0 = next(r for r in table if r["measure"].startswith("F0"))
    scv = next(r for r in table if r["measure"].startswith("SCV"))
    return {
        "n_infants": len(cohort),
        "n_measures": len(table),
        "f0_icc_a1_computed": f0["ICC_A1_overall"],
        "f0_excellent": f0["rating_overall"] == "excellent",
        "scv_poor": scv["rating_overall"] == "poor",
        "computed_table": table,
    }
