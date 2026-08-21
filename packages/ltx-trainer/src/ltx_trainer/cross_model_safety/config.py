"""Cross-model safety steering config (arXiv:2606.05290)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SteeringParams:
    default_alpha: float = 5.0
    z_image_alpha: float = 3.0
    ridge_lambda: float = 1e-2
    mlp_hidden: int = 128
    anchor_count: int = 512


@dataclass(frozen=True)
class CrossModelSafetyConfig:
    paper_arxiv: str = "arXiv:2606.05290"
    source_llms: tuple[str, ...] = ("Llama3.1-8B", "Mistral-7B", "Qwen3.5-9B")
    t2i_targets: tuple[str, ...] = (
        "Flux1-Schnell",
        "Flux1-Dev",
        "Qwen-Image",
        "Z-Image-Turbo",
    )
    t2v_targets: tuple[str, ...] = ("Wan2.2",)
    alignment_methods: tuple[str, ...] = ("svd", "ridge", "mlp")
    safety_categories: tuple[str, ...] = (
        "Hate",
        "Sexual",
        "Violence",
        "Humiliation",
        "Illegal Activities",
        "Disturbing",
    )
    params: SteeringParams = field(default_factory=SteeringParams)
    packages: tuple[str, ...] = (
        "alignment",
        "steering",
        "metrics",
        "simulation",
    )
