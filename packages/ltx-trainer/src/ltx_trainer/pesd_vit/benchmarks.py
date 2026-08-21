"""Paper table excerpts for PESD-ViT (arXiv:2605.29852)."""

from __future__ import annotations

from typing import Any

# Table II — backbone comparison (Ours NAFLD dataset)
TABLE2_OURS_NAFLD = {
    "Swin-Tiny": {"steatosis": 0.9050, "ballooning": 0.9625, "inflammation": 0.8262},
    "Swin-Small": {"steatosis": 0.8460, "ballooning": 0.9250, "inflammation": 0.7420},
    "DINOv2": {"steatosis": 0.7650, "ballooning": 0.9526, "inflammation": 0.7250},
}

TABLE2_FARZI = {
    "Swin-Tiny": {"steatosis": 0.8080, "ballooning": 0.9108, "inflammation": 0.7226},
    "Swin-Small": {"steatosis": 0.7087, "ballooning": 0.9139, "inflammation": 0.6783},
    "DINOv2": {"steatosis": 0.7907, "ballooning": 0.9146, "inflammation": 0.7501},
}

# Table III — vs InceptionV3
TABLE3_INCEPTION = {
    "InceptionV3": {"steatosis": 0.8722, "ballooning": 0.9061, "inflammation": 0.8027},
    "Ours": {"steatosis": 0.9050, "ballooning": 0.9625, "inflammation": 0.8262},
}

# Table IV — ablations
TABLE4_LAMBDA = {
    0.0: {"steatosis": 0.8327, "ballooning": 0.9220, "inflammation": 0.6720},
    0.01: {"steatosis": 0.8632, "ballooning": 0.9423, "inflammation": 0.7460},
    0.1: {"steatosis": 0.9050, "ballooning": 0.9625, "inflammation": 0.8262},
    1.0: {"steatosis": 0.8525, "ballooning": 0.9324, "inflammation": 0.4500},
}

TABLE4_STRUCTURE = {
    "Adapter-only": {"steatosis": 0.9050, "ballooning": 0.9625, "inflammation": 0.8262},
    "LoRA-only": {"steatosis": 0.7875, "ballooning": 0.9625, "inflammation": 0.7624},
}


def benchmarks_bundle() -> dict[str, Any]:
    ours = TABLE3_INCEPTION["Ours"]
    inc = TABLE3_INCEPTION["InceptionV3"]
    best_lambda = TABLE4_LAMBDA[0.1]
    no_ortho = TABLE4_LAMBDA[0.0]
    return {
        "table2_ours_nafld": TABLE2_OURS_NAFLD,
        "table2_farzi": TABLE2_FARZI,
        "table3_vs_inception": TABLE3_INCEPTION,
        "table4_lambda": TABLE4_LAMBDA,
        "table4_structure": TABLE4_STRUCTURE,
        "beats_inception_all_tasks": all(ours[k] >= inc[k] for k in ours),
        "ortho_lambda_01_best": best_lambda["inflammation"] > no_ortho["inflammation"],
        "default_backbone": "Swin-Tiny",
        "dataset_patches": 3192,
    }
