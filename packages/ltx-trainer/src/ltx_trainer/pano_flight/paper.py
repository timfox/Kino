"""Framework card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_flight.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL, PAPERS_REVIEWED, TASKS_COVERED
from ltx_trainer.pano_flight.future import future_card
from ltx_trainer.pano_flight.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "papers_reviewed": PAPERS_REVIEWED,
        "tasks_covered": TASKS_COVERED,
        "axes": [
            "perspective_to_panorama_domain_gap",
            "cross_method_mitigation_strategies",
            "cross_task_taxonomy_four_pillars",
            "data_model_application_future",
        ],
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "pano_flight", **evaluation_demo_run()}
