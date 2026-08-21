"""Reference results (Arai et al., arXiv:2512.20093)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvc_erp_qpa.config import (
    BD_RATE_WS_PSNR_AVG_PCT,
    ENC_FPS_BASELINE,
    ENC_FPS_PROPOSED,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROC_TIME_INCREASE_PCT,
)

# Table I — JVET class S1, WS-PSNR BD-Rate (%), lower is better
TABLE1_BD_RATE: dict[str, float] = {
    "SkateboardInLot": -4.27,
    "ChairLift": -11.43,
    "KiteFlite": -6.51,
    "Harbor": -2.83,
    "Trolley": -2.49,
    "GasLamp": -3.65,
    "Average": BD_RATE_WS_PSNR_AVG_PCT,
}

TABLE1_WO_INTERP_AVG = -5.08

TABLE1_FPS: dict[str, float] = {
    "encode_baseline": ENC_FPS_BASELINE,
    "decode_baseline": 1.789,
    "encode_proposed": ENC_FPS_PROPOSED,
    "decode_proposed": 1.784,
    "encode_wo_interp": 2.010,
}

# Table II — BD-Rate vs each reference (average %)
TABLE2_AVERAGE: dict[str, float] = {
    "HEVC_RA_HM": -4.09,
    "VVC_RA_VVenC": -4.11,
    "VVC_LDP_VVenC": 1.11,
    "DCVC_RT_LDP_proposed": BD_RATE_WS_PSNR_AVG_PCT,
}


def table1_row(method: str = "proposed") -> dict[str, Any]:
    if method == "baseline":
        return {k: 0.0 for k in TABLE1_BD_RATE}
    if method == "wo_interp":
        out = dict(TABLE1_BD_RATE)
        out["Average"] = TABLE1_WO_INTERP_AVG
        return out
    return dict(TABLE1_BD_RATE)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_bd_rate_proposed": TABLE1_BD_RATE,
        "table1_fps": TABLE1_FPS,
        "table2_average_bd_rate": TABLE2_AVERAGE,
        "proc_time_increase_pct": PROC_TIME_INCREASE_PCT,
    }
