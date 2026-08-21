"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_flight.benchmarks import TABLE1_DATASET_GAP, benchmarks_bundle
from ltx_trainer.pano_flight.config import PAPER_ARXIV, PanoFlightConfig
from ltx_trainer.pano_flight.future import future_card
from ltx_trainer.pano_flight.gaps import DomainGap
from ltx_trainer.pano_flight.paper import framework_card
from ltx_trainer.pano_flight.pipeline import (
    cross_method_demo,
    cross_task_demo,
    evaluation_demo_run,
    stitching_pipeline_card,
)
from ltx_trainer.pano_flight.strategies import MitigationStrategy, classify_method


def evaluation_smoke() -> dict[str, Any]:
    bundle = benchmarks_bundle()
    seg_pano = TABLE1_DATASET_GAP["segmentation"]["panoramic"]["size"]
    seg_persp = TABLE1_DATASET_GAP["segmentation"]["perspective"]["size"]

    return {
        "package": "pano_flight",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "papers_reviewed": bundle["papers_reviewed"],
        "tasks_covered": bundle["tasks_covered"],
        "three_gaps": len(DomainGap) == 3,
        "four_strategies": len(MitigationStrategy) == 4,
        "panoramic_data_scarcer_than_perspective": seg_pano < 10_000 and seg_persp != seg_pano,
        "classify_densepass": classify_method("DensePASS").value,
        "demo": evaluation_demo_run(PanoFlightConfig(height=64, width=128)),
        "cross_method": cross_method_demo(),
        "cross_task": cross_task_demo(),
        "stitching": stitching_pipeline_card(),
        "future": future_card(),
        "framework": framework_card()["axes"],
    }
