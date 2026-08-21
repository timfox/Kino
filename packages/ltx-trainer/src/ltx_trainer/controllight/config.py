"""ControlLight / Light100K (Yang et al., arXiv:2605.25569)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControlLightConfig:
    """Paper defaults (Appendix B/C, Sec. 3.3)."""

    paper_arxiv: str = "arXiv:2605.25569"
    website: str = "https://yfyang007.github.io/ControlLight/"
    base_model: str = "FLUX.2-klein-9B"
    text_encoder: str = "Qwen3-VL"
    code_url: str = "https://github.com/yfyang007/ControlLight"

    enhancement_strengths: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    reflectance_beta_scale: float = 0.5
    """β_s = reflectance_beta_scale * s in Eq. (2)."""

    # Misalignment-aware FM (Eq. 4–5, Appendix B)
    dist_threshold_px: float = 3.0
    mask_alpha: float = 0.8
    weight_min: float = 0.2

    # LoRA (Appendix C, Table 5)
    lora_rank: int = 64
    lora_alpha: int = 64
    trainable_params_m: float = 317.0
    learning_rate: float = 1e-4
    training_steps: int = 3000
    global_batch_size: int = 16
    train_resolution: int = 1024
    precision: str = "bf16"
    optimizer: str = "AdamW-8bit"

    # Evaluation strengths (Sec. 4.2)
    eval_strengths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    restoration_prompt: str = (
        "Please restore this low-quality image, recover its normal brightness and clarity, "
        "and keep the structure the same."
    )

    # Light100K scale (Appendix A)
    light100k_n_images: int = 27529
    light100k_n_pairs: int = 17809
    edge_filter_max_diff: float = 0.55

    @property
    def light100k_strengths(self) -> tuple[float, ...]:
        """Alias used by legacy mocks."""
        return self.enhancement_strengths


@dataclass
class Light100KGroup:
    """Training group G = {I_s} (Sec. 3.1)."""

    i0: object  # Tensor
    targets: dict[float, object] = field(default_factory=dict)
