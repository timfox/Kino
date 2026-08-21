"""CLEAR concept-layer erasure for T2V diffusion (Xie et al., arXiv:2605.25941)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CLEARConfig:
    """Defaults from paper Sec. 3–4 and Appendix Table 10."""

    backbone_models: tuple[str, ...] = ("Wan2.2-5B", "CogVideoX-2B")
    text_encoder: str = "T5"
    num_layers: int = 24

    d_sae: int = 131_072
    sparsity_lambda: float = 1e-4
    lr_alpha: float = 3e-2
    lr_sae: float = 1e-3
    batch_size: int = 16

    gumbel_tau_max: float = 1.0
    gumbel_tau_min: float = 0.1
    iterations_objects: int = 2500
    iterations_nudity: int = 3750

    intervention_gamma: float = 10.0
    stability_eps: float = 1e-8

    # Smoke/demo dimensions (full training uses d_sae above)
    demo_d_model: int = 64
    demo_d_sae: int = 128

    selected_blocks: dict[str, int] = field(
        default_factory=lambda: {
            "springer_dog": 2,
            "parachute": 6,
            "nudity": 18,
        }
    )
