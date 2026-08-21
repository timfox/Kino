"""Stance simulation audit config (arXiv:2606.06443)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StanceSimConfig:
    paper_arxiv: str = "arXiv:2606.06443"
    dataset_conversations: int = 1821
    dataset_subreddits: int = 97
    dataset_posts: int = 416
    dataset_target_users: int = 851
    stance_targets: tuple[str, ...] = ("DeepSeek", "Claude", "Llama")
    stance_labels: tuple[str, ...] = ("negative", "neutral", "positive")
    text_revision_strategies: tuple[str, ...] = ("paraphrase", "explain", "add")
    multimodal_strategies: tuple[str, ...] = ("meme",)
    meme_ablations: tuple[str, ...] = (
        "r_meme",
        "r_white_meme",
        "r_humor",
        "r_caption_cut",
        "r_caption",
    )
    meme_templates: int = 5
    stance_models: tuple[str, ...] = (
        "gpt-5.2-2025-12-11",
        "claude-sonnet-4-6",
        "qwen3.5-plus-2026-02-15",
    )
    revision_models: tuple[str, ...] = (
        "gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
    )
    meme_image_model: str = "gpt-image-2"
    primary_stance_model: str = "gpt-5.2-2025-12-11"
    primary_revision_model: str = "gemini-3-flash-preview"
    liwc_tone_midpoint: float = 50.0
    counts_by_target: dict[str, int] = field(
        default_factory=lambda: {"DeepSeek": 787, "Claude": 538, "Llama": 496}
    )
