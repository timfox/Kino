"""SafeDIG configuration (arXiv:2605.30049)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SafeDIGConfig:
    paper_arxiv: str = "arXiv:2605.30049"
    backbones: tuple[str, ...] = ("FLUX.1 Dev", "Stable Diffusion 3.5 Large")
    intervention_groups: tuple[str, ...] = ("Ltext", "Lbind", "Lrender")
    operators: tuple[str, ...] = ("Blend", "Repel")
    routing_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15)  # α, ρ, η, ξ
    sparsity_lambda: float = 0.03125
    sae_expansion: int = 16
    max_staging_steps: int = 3
    max_composition_steps: int = 7
    max_lighting_steps: int = 7
    default_blend_beta: float = 0.2
    default_repel_gamma: float = 0.2
    images_per_prompt: int = 10
    source_categories: tuple[str, ...] = (
        "Self-Harm",
        "Hate",
        "Illegal Activity",
        "Shock",
        "Violence",
        "Harassment",
    )
    target_category: str = "Sexual"
    benchmark_name: str = "i2p"


@dataclass
class InterventionHook:
    """Candidate DiT intervention position ℓ."""

    name: str
    group: str  # Ltext | Lbind | Lrender
    dim: int
    path: str = ""
