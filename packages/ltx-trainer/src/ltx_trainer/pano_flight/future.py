"""Future directions (Sec. 5, Figs. 10–17)."""

from __future__ import annotations

from typing import Any

FUTURE_DATA = [
    "large_scale_standardized_360_datasets",
    "diversity_beyond_indoor_urban",
    "high_quality_annotations",
    "multimodal_image_text_video_audio_lidar",
]

FUTURE_MODELS = [
    "foundation_models_on_360_data",
    "task_specific_expert_modules",
    "panorama_language_audio_alignment",
    "world_models_open_panoramic_generation",
    "specialized_erp_evaluation_metrics",
]

FUTURE_APPLICATIONS = [
    "spatial_intelligence_embodied_autonomous",
    "xr_immersive_interaction",
    "3d_reconstruction_digital_twins",
    "security_education_entertainment_healthcare",
]


def future_card() -> dict[str, Any]:
    return {
        "data_bottlenecks": FUTURE_DATA,
        "model_paradigms": FUTURE_MODELS,
        "applications": FUTURE_APPLICATIONS,
    }
