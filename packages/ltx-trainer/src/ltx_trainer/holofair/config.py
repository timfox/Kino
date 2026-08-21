"""Configuration for HoloFair + Fair-GRPO (Chen et al., arXiv:2605.24687)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HoloFairConfig:
    """Defaults from paper Sec. 3–4 and Appendix B/E."""

    attributes: tuple[str, ...] = ("gender", "age", "race")
    gender_categories: tuple[str, ...] = ("female", "male")
    age_categories: tuple[str, ...] = ("young", "middle", "elderly")
    race_categories: tuple[str, ...] = ("Asian", "Black", "Indian", "Others", "White")
    semantic_triggers: tuple[str, ...] = (
        "aggressive",
        "compassionate",
        "gentle",
        "intelligent",
        "poor",
        "professional",
        "successful",
        "trustworthy",
        "unprofessional",
    )
    ca_quantile: float = 0.10
    entropy_epsilon: float = 1e-6
  # Fair-GRPO (Sec. 3.4)
    reward_min: float = -5.0
    reward_max: float = 5.0
    edit_reward_epsilon: float = 1e-6
    attribute_reward_weights: dict[str, float] = field(
        default_factory=lambda: {"gender": 1.0, "age": 1.0, "race": 1.0}
    )
    kl_beta: float = 0.05
    lora_rank: int = 32
    learning_rate: float = 5e-5
    images_per_prompt: int = 20

    def categories_for(self, attribute: str) -> tuple[str, ...]:
        if attribute == "gender":
            return self.gender_categories
        if attribute == "age":
            return self.age_categories
        if attribute == "race":
            return self.race_categories
        raise KeyError(f"unknown attribute: {attribute}")
