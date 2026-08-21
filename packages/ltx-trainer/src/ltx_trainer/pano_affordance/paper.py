"""Paper integration card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_affordance.benchmarks import benchmarks_bundle, table1_ours_hard
from ltx_trainer.pano_affordance.config import CODE_URL, NUM_AFFORDANCE_CLASSES, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    hard = table1_ours_hard()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "task": "Holistic affordance grounding in 360° indoor ERP panoramas",
        "method": {
            "DASM": "Dual-frequency spectral modulation for ERP distortion",
            "OSDH": "Spherical affinity + seed propagation for dense maps",
            "losses": "LBCE + LKL + region-text contrastive (LRTC)",
            "backbone": "DINOv2 + CLIP with LoRA (r=16)",
        },
        "dataset": {"360-AGD": {"classes": NUM_AFFORDANCE_CLASSES, "splits": ["Easy", "Hard"]}},
        "results": {
            "360_AGD_hard_KLD": hard["KLD"],
            "360_AGD_hard_SIM": hard["SIM"],
            "360_AGD_hard_NSS": hard["NSS"],
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX implements DASM/OSDH stubs and Table I–III anchors. "
            "Production: github.com/GL-ZHU925/PanoAffordanceNet + 360-AGD."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.pano_affordance.mock import evaluation_smoke

    return evaluation_smoke()
