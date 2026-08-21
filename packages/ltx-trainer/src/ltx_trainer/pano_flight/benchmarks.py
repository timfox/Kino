"""Table 1 perspective vs panoramic dataset scale (supplementary)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pano_flight.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PAPERS_REVIEWED, PROJECT_URL, TASKS_COVERED

# Selected rows from survey Table 1 (perspective vs panoramic size)
TABLE1_DATASET_GAP = {
    "generation": {
        "perspective": {"name": "SPARF", "size": "17M", "source": "synthetic"},
        "panoramic": {"name": "HDR360-UHD", "size": 4392, "source": "real"},
    },
    "super_resolution": {
        "perspective": {"name": "GameIR-SR", "size": 19200, "source": "synthetic"},
        "panoramic": {"name": "Flickr360", "size": 3150, "source": "real"},
    },
    "object_detection": {
        "perspective": {"name": "Open Images V7", "size": "1.9M", "source": "real"},
        "panoramic": {"name": "PANDORA", "size": 3000, "source": "real"},
    },
    "segmentation": {
        "perspective": {"name": "Open Images V4", "size": "9.2M", "source": "real"},
        "panoramic": {"name": "WildPASS", "size": 2500, "source": "real"},
    },
    "depth_estimation": {
        "perspective": {"name": "ScanNet", "size": "5M", "source": "synthetic"},
        "panoramic": {"name": "Deep360", "size": 2100, "source": "real"},
    },
    "saliency": {
        "perspective": {"name": "SALICON", "size": 10000, "source": "real"},
        "panoramic": {"name": "Salient360!", "size": 85, "source": "real"},
    },
}

IMAGING_SYSTEMS = (
    "fisheye",
    "catadioptric",
    "hyper_hemispheric",
    "panoramic_annular",
    "panomorph",
    "multi_camera_stitching",
    "monocentric",
)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "papers_reviewed": PAPERS_REVIEWED,
        "tasks_covered": TASKS_COVERED,
        "table1_dataset_gap": TABLE1_DATASET_GAP,
        "imaging_systems": list(IMAGING_SYSTEMS),
    }
