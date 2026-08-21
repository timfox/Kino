"""SJTU 8K 360° dataset card (Liu et al. [30])."""

from __future__ import annotations

from typing import Any

from ltx_trainer.oma_fme.config import PAPER_ARXIV, SJTU_ERP_SIZE, SJTU_SEQUENCES

SJTU_SEQUENCE_NAMES: tuple[str, ...] = (
    "AcademicBuilding",
    "AdministrationBuilding",
    "EastGate",
    "kaixuanmenye_pano",
    "lanqiuchang_pano",
    "Library",
    "runner_pano",
    "shuangziqiao_pano",
    "SiyuanGate",
    "SouthGate",
    "StudyRoom",
    "Sward",
    # paper uses 15; list may be partial in stub
)

# Full 15 from paper Fig. 4 / dataset
SJTU_ALL_15: tuple[str, ...] = (
    "AcademicBuilding",
    "AdministrationBuilding",
    "EastGate",
    "kaixuanmenye_pano",
    "lanqiuchang_pano",
    "Library",
    "runner_pano",
    "shuangziqiao_pano",
    "SiyuanGate",
    "SouthGate",
    "StudyRoom",
    "Sward",
    "Fountain",
    "ScienceBuilding",
    "Stadium",
)


def sjtu_dataset_card() -> dict[str, Any]:
    return {
        "name": "SJTU UHD 360° Immersive Video",
        "hub_note": "See Liu et al. ICVRV 2017; sequences at 8192×4096 ERP",
        "sequence_count": SJTU_SEQUENCES,
        "erp_resolution_wh": [8192, 4096],
        "erp_resolution_hw": list(SJTU_ERP_SIZE),
        "duration_s_per_clip": 30,
        "sequences_sample": list(SJTU_ALL_15[:12]),
        "vca_features": "EY, h from Video Complexity Analyzer (Fig. 4)",
        "paper": f"arXiv:{PAPER_ARXIV}",
    }
