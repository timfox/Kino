"""Cross-task taxonomy (Sec. 4, Fig. 5)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class TaskPillar(str, Enum):
    QUALITY = "visual_quality_enhancement_assessment"
    UNDERSTANDING = "visual_understanding"
    MULTIMODAL = "multimodal_understanding"
    GENERATION = "visual_generation"


PILLAR_TASKS: dict[TaskPillar, list[str]] = {
    TaskPillar.QUALITY: [
        "super_resolution",
        "reflection_removal",
        "image_restoration",
        "visual_quality_assessment",
    ],
    TaskPillar.UNDERSTANDING: [
        "segmentation",
        "semantic_mapping",
        "detection",
        "tracking",
        "pose_estimation",
        "saliency_prediction",
        "layout_detection",
        "optical_flow",
        "keypoint_matching",
        "decomposition",
        "lighting_estimation",
        "depth_estimation",
    ],
    TaskPillar.MULTIMODAL: [
        "audio_visual_fusion",
        "lidar_fusion",
        "text_fusion_vqa",
    ],
    TaskPillar.GENERATION: [
        "text_guided_generation",
        "image_completion",
        "novel_view_synthesis",
        "world_model_applications",
    ],
}

FIG5_HIGHLIGHTS: dict[str, list[str]] = {
    "super_resolution": ["LAUNet", "OSRT", "SphereSR", "OmniSSR", "DiffOSR"],
    "depth_estimation": ["OmniDepth", "BiFuse", "PanoFormer", "PanSplat", "Depth Anywhere"],
    "segmentation": ["DensePASS", "Trans4PASS+", "GoodSAM", "OmniSAM"],
    "text_guided_generation": ["Text2Light", "SphereDiffusion", "PanoDiT", "VideoPanda", "Diffusion360"],
    "novel_view_synthesis": ["PanoGRF", "360-GS", "PanSplat", "HunyuanWorld 1.0", "Matrix-3D"],
}


def taxonomy_card() -> dict[str, Any]:
    return {
        "pillars": [p.value for p in TaskPillar],
        "tasks_by_pillar": {p.value: PILLAR_TASKS[p] for p in TaskPillar},
        "sample_methods": FIG5_HIGHLIGHTS,
    }
