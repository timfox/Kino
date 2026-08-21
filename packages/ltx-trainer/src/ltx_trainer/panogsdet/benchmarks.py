"""Reference metrics from Ning et al. (arXiv:2605.14601)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panogsdet.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, STRUCTURED3D_CATEGORIES

# Table I — AP@25 per class + mAP
TABLE1_AP25: dict[str, float] = {
    "bed": 0.9023,
    "chair": 0.2314,
    "sofa": 0.2142,
    "table": 0.3033,
    "desk": 0.4215,
    "dresser": 0.5234,
    "cabinet": 0.4071,
    "fridge": 0.5854,
    "sink": 0.2562,
    "lamp": 0.2344,
    "bathtub": 0.6745,
    "mAP@25": 0.4321,
    "TR3D-PanoFormer_mAP@25": 0.3682,
    "PanoContext-Former_mAP@25": 0.3680,
    "DeepPanoContext_mAP@25": 0.2989,
}

TABLE2_AP50: dict[str, float] = {
    "bed": 0.7040,
    "chair": 0.0539,
    "sofa": 0.0753,
    "table": 0.1276,
    "desk": 0.1429,
    "dresser": 0.2270,
    "cabinet": 0.2012,
    "fridge": 0.4238,
    "sink": 0.1071,
    "lamp": 0.0624,
    "bathtub": 0.5829,
    "mAP@50": 0.2461,
    "TR3D-PanoFormer_mAP@50": 0.1944,
}

TABLE3_RESOURCES: dict[str, Any] = {
    "DeepPanoContext_train_mb": 22145,
    "DeepPanoContext_test_mb": 8653,
    "DeepPanoContext_fps": 1,
    "PanoContextFormer_train_mb": 21122,
    "PanoContextFormer_test_mb": 8342,
    "PanoContextFormer_fps": 2,
    "Ours_train_mb": 3893,
    "Ours_test_mb": 3197,
    "Ours_fps": 11,
    "depth_branch_mb": 2876,
    "gaussian_train_mb": 1000,
    "gaussian_infer_mb": 321,
}

TABLE4_ABLATION: list[dict[str, Any]] = [
    {"lifting": False, "optimize": False, "mAP@25": 0.3682, "mAP@50": 0.1944},
    {"lifting": True, "optimize": False, "mAP@25": 0.3852, "mAP@50": 0.2223},
    {"lifting": False, "optimize": True, "mAP@25": 0.3922, "mAP@50": 0.2112},
    {"lifting": True, "optimize": True, "mAP@25": 0.4321, "mAP@50": 0.2461},
]

DATASET_STRUCTURED3D = {
    "images": 196_000,
    "scenes": 3500,
    "rooms": 12_835,
    "categories": list(STRUCTURED3D_CATEGORIES),
    "train_scenes": 3000,
    "val_scenes": "3000-3250",
    "test_scenes": 250,
    "modes": ("Normal", "Full", "Empty"),
    "train_mode": "Full",
}

TRAINING = {
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.001,
    "grad_clip": 35,
    "epochs": 12,
    "lr_decay_epochs": (8, 11),
    "depth_backbone": "Panoformer (frozen)",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "table1_ap25": TABLE1_AP25,
        "table2_ap50": TABLE2_AP50,
        "table3_resources": TABLE3_RESOURCES,
        "table4_ablation": TABLE4_ABLATION,
        "dataset_structured3d": DATASET_STRUCTURED3D,
        "training": TRAINING,
    }
