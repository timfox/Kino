"""Agent-facing framework card for World Tracing."""

from __future__ import annotations

from typing import Any

from ltx_trainer.world_tracing.benchmarks import benchmarks_bundle
from ltx_trainer.world_tracing.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL


def framework_card() -> dict[str, Any]:
    return {
        "paper": {
            "title": PAPER_TITLE,
            "arxiv": f"arXiv:{PAPER_ARXIV}",
            "url": PAPER_URL,
            "project": PROJECT_URL,
            "authors": "Hao Zhang et al. (World Labs / UIUC)",
        },
        "problem": (
            "Single-image 3D must be both pixel-aligned (faithful visible surface) and complete "
            "(occluded geometry). Depth predictors stop at L0; image-to-3D generators lose correspondence."
        ),
        "method": {
            "representation": "L×H×W camera-space XYZ stack; L0=visible, deeper layers=occluded surfaces",
            "model": "WT-DiT — flow-matching DiT with layer/ray/global attention + frozen MoGe encoder",
            "training": "Depth-filling dense targets, XYZ-only endpoint loss, layer-aware timestep mixture",
            "variants": ["WT-O objects", "WT-S scenes", "WT-D dynamic clips"],
        },
        "downstream": [
            "Text-driven 3D scene editing (camera-frame compose)",
            "Geometry-guided novel-view video (multilayer depth memory)",
            "Training-free TRELLIS hybrid (voxelized L0–L5 prior)",
        ],
        "results": {
            "object_visible_mae": benchmarks_bundle()["table1_object_visible"]["WT-O"]["mae"],
            "object_full_l1": benchmarks_bundle()["table1_object_full"]["WT-O"]["l1"],
            "scene_mae_3dfront": benchmarks_bundle()["table2_scene_3dfront"]["WT-S"]["mae"],
            "dynamic_mean_cd_l2": benchmarks_bundle()["table3_dynamic"]["WT-D"]["mean"],
        },
        "gopex": {
            "package": "ltx_trainer.world_tracing",
            "cli": "./scripts/kino-world-tracing.sh knowledge",
            "tools": "world_tracing_framework_card, world_tracing_eval_demo",
        },
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.world_tracing.mock import evaluation_smoke

    return evaluation_smoke()
