"""Framework card, MV2LRS3 tables, evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config
from ltx_trainer.mv2lrs3.matching import matching_demo
from ltx_trainer.mv2lrs3.metrics import predict_mv2lrs3_wer
from ltx_trainer.mv2lrs3.models import model_registry


def framework_card(cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "benchmark": cfg.benchmark,
        "reference": cfg.reference_set,
        "candidate_pool": cfg.candidate_pool,
        "release_url": cfg.release_url,
        "method": "weighted kNN distribution matching (8-D covariates)",
        "factors": list(cfg.matching_factors),
        "models_evaluated": list(cfg.evaluated_models),
        "linear_fit": {
            "slope": cfg.linear_fit_slope,
            "intercept": cfg.linear_fit_intercept,
            "equation": "WER_MV2LRS3 = 10.4 × WER_LRS3 + 8.1",
        },
        "headline": {
            "best_lrs3_wer_pct": 0.77,
            "best_mv2lrs3_wer_pct": 14.0,
            "collapse": "universal performance drop under matched distribution",
        },
    }


def table1_lrs3_vs_mv2lrs3() -> list[dict[str, Any]]:
    """Table 1 — WER % and rank on LRS3 Test vs MV2LRS3 (5-run mean ± std)."""
    return [
        {
            "model": "Llama-AVSR",
            "lrs3_wer": 0.77,
            "lrs3_rank": 1,
            "mv2lrs3_wer": "16.5 ± 0.4",
            "mv2lrs3_wer_mean": 16.5,
            "mv2lrs3_rank": 2,
        },
        {
            "model": "Whisper-Flamingo",
            "lrs3_wer": 0.86,
            "lrs3_rank": 2,
            "mv2lrs3_wer": "18.6 ± 1.0",
            "mv2lrs3_wer_mean": 18.6,
            "mv2lrs3_rank": 3,
        },
        {
            "model": "Auto-AVSR",
            "lrs3_wer": 0.95,
            "lrs3_rank": 3,
            "mv2lrs3_wer": "14.0 ± 0.3",
            "mv2lrs3_wer_mean": 14.0,
            "mv2lrs3_rank": 1,
        },
        {
            "model": "USR",
            "lrs3_wer": 1.10,
            "lrs3_rank": 4,
            "mv2lrs3_wer": "21.5 ± 0.3",
            "mv2lrs3_wer_mean": 21.5,
            "mv2lrs3_rank": 4,
        },
        {
            "model": "AV-HuBERT",
            "lrs3_wer": 1.50,
            "lrs3_rank": 5,
            "mv2lrs3_wer": "23.5 ± 0.6",
            "mv2lrs3_wer_mean": 23.5,
            "mv2lrs3_rank": 5,
        },
    ]


def table2_ten_x_subset() -> list[dict[str, Any]]:
    """Table 2 — MV2LRS3 vs scaled 10x set."""
    return [
        {"model": "Auto-AVSR", "mv2lrs3": 14.0, "mv2lrs3_rank": 1, "ten_x": 12.9, "ten_x_rank": 1},
        {"model": "Llama-AVSR", "mv2lrs3": 16.5, "mv2lrs3_rank": 2, "ten_x": 13.5, "ten_x_rank": 2},
        {"model": "Whisper-Flamingo", "mv2lrs3": 18.6, "mv2lrs3_rank": 3, "ten_x": 16.0, "ten_x_rank": 3},
        {"model": "USR", "mv2lrs3": 21.5, "mv2lrs3_rank": 4, "ten_x": 19.4, "ten_x_rank": 4},
        {"model": "AV-HuBERT", "mv2lrs3": 23.5, "mv2lrs3_rank": 5, "ten_x": 21.2, "ten_x_rank": 5},
    ]


def table3_leave_one_out() -> list[dict[str, Any]]:
    """Table 3 — leave-one-out attribute analysis (MV2LRS3 baseline + exclude factor)."""
    return [
        {
            "model": "Whisper-Flamingo",
            "mv2lrs3": 18.6,
            "exclude_duration": 9.9,
            "rel_change_duration_pct": -47,
        },
        {
            "model": "Auto-AVSR",
            "mv2lrs3": 14.0,
            "exclude_duration": 10.6,
            "rel_change_duration_pct": -24,
        },
        {
            "model": "Llama-AVSR",
            "mv2lrs3": 16.5,
            "exclude_yaw": 14.2,
            "rel_change_yaw_pct": -14,
        },
    ]


def table4_binned_subsets() -> list[dict[str, Any]]:
    """Table 4 — duration and yaw bins on MV2LRS3."""
    return [
        {"model": "Whisper-Flamingo", "duration_0_3s": 22.4, "duration_3_7s": 8.3, "yaw_0_30": 16.8, "yaw_60_90": 20.4},
        {"model": "Auto-AVSR", "duration_0_3s": 15.0, "duration_3_7s": 10.4, "yaw_0_30": 13.5, "yaw_60_90": 15.1},
        {"model": "Llama-AVSR", "duration_0_3s": 19.9, "duration_3_7s": 7.7, "yaw_0_30": 14.2, "yaw_60_90": 14.1},
    ]


def table5_vocabulary_iwer() -> list[dict[str, Any]]:
    """Table 5 — IWER on V_share vs V_diff."""
    return [
        {"model": "Auto-AVSR", "vlrs3_test": 0.8, "v_share": 8.5, "v_diff": 19.9, "delta": 11.4},
        {"model": "Llama-AVSR", "vlrs3_test": 0.6, "v_share": 9.1, "v_diff": 15.2, "delta": 6.1},
        {"model": "Whisper-Flamingo", "vlrs3_test": 0.6, "v_share": 14.8, "v_diff": 17.8, "delta": 3.0},
        {"model": "AV-HuBERT", "vlrs3_test": 1.3, "v_share": 19.0, "v_diff": 28.7, "delta": 9.7},
    ]


def table6_modalities() -> list[dict[str, Any]]:
    """Table 6 — AV / AO / VO WER on MV2LRS3."""
    return [
        {"model": "Auto-AVSR", "av": 14.0, "ao": 16.4, "vo": 45.5, "visual_helps": True},
        {"model": "Llama-AVSR", "av": 16.5, "ao": 15.3, "vo": 81.5, "visual_helps": False},
        {"model": "Whisper-Flamingo", "av": 18.6, "ao": 16.6, "vo": None, "visual_helps": False},
        {"model": "USR", "av": 21.5, "ao": 21.0, "vo": 57.9, "visual_helps": False},
        {"model": "AV-HuBERT", "av": 23.5, "ao": 23.6, "vo": 65.1, "visual_helps": False},
    ]


def table7_error_rates() -> list[dict[str, Any]]:
    """Table 7 — Sub / Del / Ins on MV2LRS3."""
    return [
        {"model": "Auto-AVSR", "sub": 5.5, "del": 2.6, "ins": 5.1},
        {"model": "Llama-AVSR", "sub": 4.0, "del": 3.9, "ins": 6.4},
        {"model": "Whisper-Flamingo", "sub": 4.1, "del": 1.8, "ins": 11.2},
        {"model": "AV-HuBERT", "sub": 7.1, "del": 1.6, "ins": 13.1},
    ]


def linear_fit_points() -> list[dict[str, float]]:
    rows = table1_lrs3_vs_mv2lrs3()
    return [
        {
            "model": r["model"],
            "lrs3_wer": r["lrs3_wer"],
            "mv2lrs3_wer": r["mv2lrs3_wer_mean"],
            "predicted": predict_mv2lrs3_wer(r["lrs3_wer"]),
        }
        for r in rows
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_lrs3_vs_mv2lrs3": table1_lrs3_vs_mv2lrs3(),
        "table2_ten_x_subset": table2_ten_x_subset(),
        "table3_leave_one_out": table3_leave_one_out(),
        "table4_binned_subsets": table4_binned_subsets(),
        "table5_vocabulary_iwer": table5_vocabulary_iwer(),
        "table6_modalities": table6_modalities(),
        "table7_error_rates": table7_error_rates(),
        "linear_fit_points": linear_fit_points(),
        "model_registry": model_registry(),
    }


def headline_results(cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    best_mv2 = min(table1_lrs3_vs_mv2lrs3(), key=lambda r: r["mv2lrs3_wer_mean"])
    best_lrs3 = min(table1_lrs3_vs_mv2lrs3(), key=lambda r: r["lrs3_wer"])
    return {
        "best_mv2lrs3_model": best_mv2["model"],
        "best_mv2lrs3_wer_pct": best_mv2["mv2lrs3_wer_mean"],
        "best_lrs3_model": best_lrs3["model"],
        "best_lrs3_wer_pct": best_lrs3["lrs3_wer"],
        "ranking_inverts": best_lrs3["model"] != best_mv2["model"],
        "linear_slope": cfg.linear_fit_slope,
    }


def evaluation_demo(*, seed: int = 0, cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    match = matching_demo(seed=seed)
    llama = next(r for r in table1_lrs3_vs_mv2lrs3() if r["model"] == "Llama-AVSR")
    predicted = predict_mv2lrs3_wer(
        llama["lrs3_wer"], slope=cfg.linear_fit_slope, intercept=cfg.linear_fit_intercept
    )
    return {
        "framework": framework_card(cfg),
        "matching": match,
        "linear_fit": {
            "lrs3_wer": llama["lrs3_wer"],
            "observed_mv2lrs3": llama["mv2lrs3_wer_mean"],
            "predicted_mv2lrs3": round(predicted, 1),
        },
        "headline": headline_results(cfg),
    }
