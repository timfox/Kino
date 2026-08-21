"""Training dataset catalogue (Sec. 5.1, Table 5 excerpts)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ltx_trainer.humanview_vu.config import PAPER_ARXIV, PAPER_TITLE


class TrainDataCategory(str, Enum):
    VIDEO_QA = "video_qa"
    VIDEO_CAPTIONING = "video_captioning"
    VIDEO_TEMPORAL_GROUNDING = "video_temporal_grounding"
    LONG_VIDEO_MEMORY = "long_video_memory"


TABLE5_TRAIN_DATASETS: dict[TrainDataCategory, list[dict[str, str | int]]] = {
    TrainDataCategory.VIDEO_QA: [
        {"name": "VideoChat2-IT", "year": 2024, "scale": "1.9M", "focus": "mixed instruction tuning"},
        {"name": "LLaVA-Video-178K", "year": 2024, "scale": "1.3M", "focus": "caption + QA + MCQ"},
        {"name": "Video-R1-CoT-165K", "year": 2025, "scale": "165K", "focus": "CoT + RL reasoning"},
        {"name": "STGR", "year": 2025, "scale": "30K", "focus": "grounded timestamps + boxes"},
        {"name": "Seeker-173K", "year": 2026, "scale": "173K", "focus": "multi-turn tool interaction"},
    ],
    TrainDataCategory.VIDEO_CAPTIONING: [
        {"name": "Panda-70M", "year": 2024, "scale": "70M", "focus": "auto-selected captions"},
        {"name": "ShareGPT4Video", "year": 2024, "scale": "4.8M", "focus": "dense captions"},
        {"name": "Video ReCap", "year": 2024, "scale": "5.3M", "focus": "hierarchical hour-long"},
        {"name": "Tarsier2-Recap-585K", "year": 2025, "scale": "585K", "focus": "fine-grained recaption"},
        {"name": "TimeChatCap-42K", "year": 2026, "scale": "42K", "focus": "time-aware AV scripts"},
    ],
    TrainDataCategory.VIDEO_TEMPORAL_GROUNDING: [
        {"name": "TimeIT", "year": 2023, "scale": "125K", "focus": "unified VTG instruction"},
        {"name": "VTG-IT-120K", "year": 2025, "scale": "120K", "focus": "standardized time tokens"},
        {"name": "Moment-10M", "year": 2024, "scale": "10.4M", "focus": "moment-level instructions"},
        {"name": "TimeLens-100K", "year": 2025, "scale": "100K", "focus": "high-precision VTG"},
        {"name": "MTVR", "year": 2025, "scale": "72K+110K", "focus": "tool-augmented CoT VTG"},
    ],
    TrainDataCategory.LONG_VIDEO_MEMORY: [
        {"name": "VideoMarathon", "year": 2025, "scale": "3.3M QA", "focus": "hour-scale instruction"},
        {"name": "M3-Agent train", "year": 2026, "scale": "10.9K videos", "focus": "entity-centric LTM"},
    ],
}


def datasets_card() -> dict[str, Any]:
    return {
        "paper_arxiv": PAPER_ARXIV,
        "paper_title": PAPER_TITLE,
        "categories": [c.value for c in TrainDataCategory],
        "datasets_by_category": {
            c.value: TABLE5_TRAIN_DATASETS[c] for c in TrainDataCategory
        },
    }
