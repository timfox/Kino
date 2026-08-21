"""Paper integration card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mtpano.benchmarks import benchmarks_bundle, table1_ours
from ltx_trainer.mtpano.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL, TRAIN_PANORAMAS


def framework_card() -> dict[str, Any]:
    ref = table1_ours()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "paradigm": "Label-free multi-task panoramic dense prediction",
        "method": {
            "PD-BridgeNet": "dual invariant/variant streams + truncated gradient bridge",
            "ERP_Token_Mixer": "latitude-adaptive 3×3 / 3×9 kernels (Eq. 2)",
            "training": f"~{TRAIN_PANORAMAS} unlabeled panoramas, 32 perspective patches, MoGe-2 + InternImage-H pseudo-labels",
            "tasks": "semseg, depth, normals + aux gradient/EDF/point map",
            "backbone": "DINOv3-Large + DPT head",
        },
        "results": {
            "Structured3D_mIoU": ref["mIoU"],
            "Structured3D_AbsRel": ref["AbsRel"],
            "Stanford2D3D_mIoU": 69.47,
            "Matterport3D_mIoU": 39.11,
            "PD_BridgeNet_delta_mtl_pct": 7.21,
            "DINOv3_init_mIoU": 28.58,
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": "GOPEX stub for MTPano / PD-BridgeNet; production weights at github.com/Evergreen0929/MTPano",
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.mtpano.mock import evaluation_smoke

    return evaluation_smoke()
