"""Reference metrics from Lu et al. (arXiv:2603.23179)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gimbal360.config import (
    HORIZON360_SIZE,
    LAMBDA_FLOW,
    LAMBDA_SHIFT,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)

# Table 1 — Structured3D (indoor) and CVRG-Pano (outdoor)
TABLE1_METRICS: list[dict[str, Any]] = [
    {
        "method": "HY-World",
        "indoor": {"FID": 71.40, "KID_x100": 1.8219, "FAED": 15.5, "CS": 27.99},
        "outdoor": {"FID": 82.11, "KID_x100": 1.9048, "FAED": 15.7, "CS": 33.16},
    },
    {
        "method": "WorldGen",
        "indoor": {"FID": 65.66, "KID_x100": 1.1640, "FAED": 13.6, "CS": 30.37},
        "outdoor": {"FID": 71.35, "KID_x100": 1.1688, "FAED": 13.5, "CS": 33.17},
    },
    {
        "method": "DiT360",
        "indoor": {"FID": 80.50, "KID_x100": 3.1790, "FAED": 25.1, "CS": 31.62},
        "outdoor": {"FID": 111.28, "KID_x100": 3.2554, "FAED": 25.4, "CS": 32.92},
    },
    {
        "method": "w/o DAL",
        "indoor": {"FID": 64.48, "KID_x100": 1.2726, "FAED": 13.8, "CS": 31.50},
        "outdoor": {"FID": 69.82, "KID_x100": 1.9039, "FAED": 15.8, "CS": 32.43},
    },
    {
        "method": "w/o TEG",
        "indoor": {"FID": 55.05, "KID_x100": 0.8012, "FAED": 8.8, "CS": 32.83},
        "outdoor": {"FID": 62.74, "KID_x100": 0.8194, "FAED": 8.6, "CS": 34.38},
    },
    {
        "method": "Ours",
        "indoor": {"FID": 54.18, "KID_x100": 0.8106, "FAED": 8.6, "CS": 32.91},
        "outdoor": {"FID": 63.28, "KID_x100": 0.8236, "FAED": 8.5, "CS": 34.50},
    },
]

TRAINING = {
    "backbone": "Flux.1-fill-dev + LoRA r=64",
    "dataset": f"Horizon360 {HORIZON360_SIZE} ERP",
    "erp_resolution": "960×1920",
    "optimizer": "AdamW",
    "lr": 1e-4,
    "steps": 10_000,
    "loss": f"L_LDM + {LAMBDA_SHIFT} L_shift + {LAMBDA_FLOW} L_flow",
}

EVAL_TEST = {
    "structured3d": 500,
    "cvrg_pano": 500,
    "total": 1000,
}


def table1_ours() -> dict[str, Any]:
    return next(r for r in TABLE1_METRICS if r["method"] == "Ours")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL, "project": PROJECT_URL},
        "table1": TABLE1_METRICS,
        "training": TRAINING,
        "eval_test": EVAL_TEST,
    }
