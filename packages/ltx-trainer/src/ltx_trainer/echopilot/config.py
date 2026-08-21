"""EchoPilot training-free ultrasound VOS (Xiao et al., arXiv:2605.25944)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EchoPilotConfig:
    """Defaults from paper Sec. 2–3."""

    category_default: str = "placenta"
    vlm_backbone: str = "BioMedCLIP-ViT-B"
    vfm_backbone: str = "DINOv3-ViT-Plus"
    segmentor_backends: tuple[str, ...] = ("SAM2", "MedSAM2")

    scale_factors: tuple[float, ...] = (1.0, 0.8, 0.5, 0.4)
    attribution_layers: int = 3
    max_aux_prompts: int = 3
    nms_radius_px: int = 6
    memory_gate_tau: float = 0.5
    stability_eps: float = 1e-8

    eval_seeds: tuple[int, ...] = (2025, 2026, 42)
    placenta_frames: int = 671

    project_page: str = "https://keeplearning-again.github.io/EchoPilot/"
