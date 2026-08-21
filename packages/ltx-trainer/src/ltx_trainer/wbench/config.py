"""Hyperparameters and dataset scale for WBENCH (Ying et al., arXiv:2605.25874)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WBENCHConfig:
    """Paper defaults; evaluation uses external VLM / MegaSaM / specialist models."""

    paper_arxiv: str = "arXiv:2605.25874"
    repo_url: str = "https://github.com/meituan-longcat/WBench"
    num_cases: int = 289
    num_turns: int = 1058
    nav_cases: int = 158
    """Shared navigation subset for cross-paradigm comparison (Sec. 5.1)."""

    num_evaluated_models: int = 20
    num_sub_metrics: int = 22
    num_dimensions: int = 5

    arc_length_samples: int = 20
    """K for GT/pred trajectory resampling (Appendix C.4.1)."""

    min_pred_displacement: float = 0.1
    fallback_translation_length: float = 1.0
    min_pred_rotation_deg: float = 3.0
    fallback_rotation_deg: float = 30.0
    fallback_orbit_radius_tpp: float = 1.0

    spatial_gate_tau: float = 0.15
    """Gated spatial consistency threshold (Eq. 15)."""

    hpsv3_p1: float = 5.21
    hpsv3_p99: float = 8.66

    vlm_model: str = "doubao-seed-2-0-lite-260215"
    visual_plausibility_model: str = "Qwen3-VL-30B-A3B (fine-tuned)"

    interaction_types: tuple[str, ...] = (
        "navigation",
        "subject_action",
        "event_editing",
        "perspective_switching",
    )

    dimensions: tuple[str, ...] = (
        "video_quality",
        "setting_adherence",
        "interaction_adherence",
        "consistency",
        "physical",
    )

    navigation_keys: tuple[str, ...] = ("W", "S", "A", "D", "left", "right", "up", "down")

    text_driven_models: tuple[str, ...] = field(
        default_factory=lambda: (
            "Seedance 1.5",
            "Wan 2.7",
            "Kling 3.0",
            "YUME 1.5",
            "HY-Video 1.5",
            "LTX 2.3",
            "LongCat-Video",
            "Kairos 3.0",
            "Cosmos 2.5",
        )
    )
