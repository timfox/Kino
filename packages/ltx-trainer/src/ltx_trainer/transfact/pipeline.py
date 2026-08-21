"""Framework card and benchmarks for TransFACT."""

from __future__ import annotations

from typing import Any

from ltx_trainer.transfact.config import TransfactConfig
from ltx_trainer.transfact.layout import ARCHITECTURE_NOTES, LIMITATIONS, STAGE_CLASSES
from ltx_trainer.transfact.mock import evaluation_smoke, mhi_demo
from ltx_trainer.transfact.tables import (
    progressive_accuracy_curve,
    table1_input_modalities,
    table2_significance,
    table2_vs_sfr,
)


def framework_card(cfg: TransfactConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TransfactConfig()
    return {
        "name": cfg.model_name,
        "paper": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "institutions": list(cfg.institutions),
        "problem": (
            "Predict bovine embryo transferability (T vs NT) at 4 DPI from 2D time-lapse "
            "videomicroscopy (1–4 DPI), using developmental stage segmentation as auxiliary supervision."
        ),
        "dataset": {
            "name": cfg.dataset_name,
            "videos": cfg.num_videos,
            "splits": {
                "train": cfg.split_train,
                "val": cfg.split_val,
                "test": cfg.split_test,
            },
            "nt_fraction": cfg.nt_class_fraction,
            "frames": cfg.num_frames,
            "resolution": list(cfg.frame_size),
            "interval_minutes": cfg.frame_interval_minutes,
        },
        "architecture": {
            "inspired_by": "FACT (frame–action cross-attention, CVPR 2024)",
            "stage_classes": list(STAGE_CLASSES),
            "num_stage_tokens": cfg.num_stage_tokens,
            "update_blocks": cfg.num_update_blocks,
            "notes": list(ARCHITECTURE_NOTES),
            "optional_mhi": {"tau": cfg.mhi_tau, "theta": cfg.mhi_theta},
        },
        "training": {
            "epochs": cfg.training_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.learning_rate,
            "optimizer": cfg.optimizer,
            "loss_terms": list((cfg.loss_weights or {}).keys()),
        },
        "findings": [
            "Frame features alone reach 82.7% accuracy (best overall).",
            "Beats SFR 3D-CNN baseline (~69.2% accuracy) on INRAE-Gertrude-DT.",
            "Transferability accuracy grows with observed cleavage / LAG events (54% → 82%).",
            "~79% accuracy possible with only 120 frames (~2 DPI) but higher variance.",
        ],
        "limitations": list(LIMITATIONS),
        "baseline_competitor": cfg.baseline_competitor,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_modalities": table1_input_modalities(),
        "table2_vs_sfr": table2_vs_sfr(),
        "table2_significance": table2_significance(),
        "progressive_accuracy": progressive_accuracy_curve(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "mhi_demo": mhi_demo()}
