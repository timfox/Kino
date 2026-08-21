"""Reference metrics from Yan et al. (arXiv:2604.23953)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vuga.config import (  # noqa: F401 — re-export for paper card
    CODE_URL,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)

# Table II — VUGA on OIQA databases (SRCC / PLCC)
TABLE2_OIQA_VUGA: dict[str, dict[str, float]] = {
    "CVIQ": {"SRCC": 0.970, "PLCC": 0.984},
    "OIQA": {"SRCC": 0.973, "PLCC": 0.974},
    "MVAQD": {"SRCC": 0.966, "PLCC": 0.973},
    "IQA-ODI": {"SRCC": 0.967, "PLCC": 0.978},
    "OSIQA": {"SRCC": 0.928, "PLCC": 0.950},
    "AIGCOIQA": {"SRCC": 0.913, "PLCC": 0.868},
    "JUFE-10K": {"SRCC": 0.846, "PLCC": 0.842},
    "OIQ-10K": {"SRCC": 0.830, "PLCC": 0.834},
}

# Table II — second-best MTAOIQA on key sets
TABLE2_BASELINE_MTAOIQA: dict[str, dict[str, float]] = {
    "JUFE-10K": {"SRCC": 0.821, "PLCC": 0.822},
    "OIQ-10K": {"SRCC": 0.824, "PLCC": 0.829},
}

# Table III — VUGA on IQA databases
TABLE3_IQA_VUGA: dict[str, dict[str, float]] = {
    "KADID-10k": {"SRCC": 0.973, "PLCC": 0.975},
    "KonIQ-10k": {"SRCC": 0.934, "PLCC": 0.945},
    "LIVE": {"SRCC": 0.982, "PLCC": 0.985},
    "SPAQ": {"SRCC": 0.924, "PLCC": 0.919},
    "TID2013": {"SRCC": 0.969, "PLCC": 0.973},
}

# Table IV — ablation
TABLE4_ABLATION: list[dict[str, Any]] = [
    {"variant": "w/o CMP", "JUFE_SRCC": 0.835, "OIQ_SRCC": 0.819},
    {"variant": "w/o SDA", "JUFE_SRCC": 0.836, "OIQ_SRCC": 0.827},
    {"variant": "w/o CAE", "JUFE_SRCC": 0.842, "OIQ_SRCC": 0.828},
    {"variant": "VUGA", "JUFE_SRCC": 0.846, "OIQ_SRCC": 0.830},
]

# Table V — cross-database
TABLE5_CROSS_DB: dict[str, dict[str, float]] = {
    "VUGA_train_JUFE_test_OIQ": {"SRCC": 0.603, "PLCC": 0.594},
    "VUGA_train_OIQ_test_JUFE": {"SRCC": 0.713, "PLCC": 0.718},
    "OIQAND_train_JUFE_test_OIQ": {"SRCC": 0.536, "PLCC": 0.529},
}

# Table VI — resolution
TABLE6_RESOLUTION: list[dict[str, Any]] = [
    {"res": "224", "JUFE_SRCC": 0.810, "OIQ_SRCC": 0.783},
    {"res": "512", "JUFE_SRCC": 0.828, "OIQ_SRCC": 0.817},
    {"res": "768", "JUFE_SRCC": 0.835, "OIQ_SRCC": 0.829},
    {"res": "1024", "JUFE_SRCC": 0.846, "OIQ_SRCC": 0.830},
]

TRAINING = {
    "backbone": "SwinV2-T (frozen, ImageNet)",
    "optimizer": "Adam",
    "lr": 1e-4,
    "scheduler": "cosine",
    "epochs": 10,
    "batch_size": 8,
    "input_sizes": [224, 1024],
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL, "code": CODE_URL},
        "table2_oiqa_vuga": TABLE2_OIQA_VUGA,
        "table2_baseline_mtao": TABLE2_BASELINE_MTAOIQA,
        "table3_iqa_vuga": TABLE3_IQA_VUGA,
        "table4_ablation": TABLE4_ABLATION,
        "table5_cross_db": TABLE5_CROSS_DB,
        "table6_resolution": TABLE6_RESOLUTION,
        "training": TRAINING,
    }
