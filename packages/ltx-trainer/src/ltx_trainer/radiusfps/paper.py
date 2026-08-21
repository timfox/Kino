"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.radiusfps.benchmarks import (
    fig1_fps_share,
    fig15_peak_speedups,
    summary_anchors,
    table_2_pointmetabase,
    table_4_ablation,
)
from ltx_trainer.radiusfps.constants import (
    BACKBONES,
    BASELINE_SAMPLERS,
    DATASETS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.radiusfps.fps import fps_algorithm_card
from ltx_trainer.radiusfps.gpu import fusion_kernel_card, gpu_pipeline_stages
from ltx_trainer.radiusfps.references import reference_anchors
from ltx_trainer.radiusfps.voxel import voxel_method_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "authors": "Ziyang Yu, Xiang Li, Qiong Chang, Jun Miyazaki",
        "affiliation": "Institute of Science Tokyo",
        "findings": [
            "Spherical voxel radius pruning + coordinate point-skip for exact FPS",
            "RadiusFPS-G warp fusion kernels with coalesced SoA memory access",
            "Up to 186× CPU and 52× GPU speedup vs vanilla FPS",
            "2.5× E2E segmentation; FastPoint+RadiusFPS-G up to 3.3× / 11.7× sampling",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "datasets": list(DATASETS),
        "backbones": list(BACKBONES),
        "baselines": list(BASELINE_SAMPLERS),
        "fps": fps_algorithm_card(),
        "radiusfps": voxel_method_card(),
        "radiusfps_g": fusion_kernel_card(),
        "gpu_stages": gpu_pipeline_stages(),
        "fig_1": fig1_fps_share(),
        "table_2": table_2_pointmetabase(),
        "table_4": table_4_ablation(),
        "fig_15": fig15_peak_speedups(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
