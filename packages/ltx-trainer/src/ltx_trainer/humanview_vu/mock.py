"""CPU smoke for Human-View video understanding survey."""

from __future__ import annotations

from typing import Any

from ltx_trainer.humanview_vu.benchmarks import benchmarks_bundle, table1_ours_covers_all_axes
from ltx_trainer.humanview_vu.config import PAPER_ARXIV
from ltx_trainer.humanview_vu.datasets import TrainDataCategory
from ltx_trainer.humanview_vu.paper import evaluation_demo, framework_card, knowledge_blob
from ltx_trainer.humanview_vu.pipeline import evaluation_demo_run
from ltx_trainer.humanview_vu.subfields import VideoSubfield
from ltx_trainer.humanview_vu.taxonomy import CoreAbility, classify_method


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo_run()
    return {
        "package": "humanview_vu",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "three_abilities": len(CoreAbility) == 3,
        "table1_full_coverage": table1_ours_covers_all_axes(),
        "train_categories": len(TrainDataCategory) == 4,
        "subfields": len(VideoSubfield) == 5,
        "classify_timechat": classify_method("TimeChat"),
        "classify_video_o3": classify_method("Video-o3"),
        "benchmark_axes": len(benchmarks_bundle()["table1_axes"]),
        "demo": demo,
        "framework": framework_card(),
        "knowledge": knowledge_blob(),
        "evaluation_demo": evaluation_demo(),
    }
