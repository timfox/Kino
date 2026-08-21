"""UNIVID video moderation VLM config (arXiv:2606.05748)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnividConfig:
    paper_arxiv: str = "arXiv:2606.05748"
    backbone_7b: str = "UNIVID-7B (Mistral-v0.3-7B + LLaVA-OV)"
    backbone_1b: str = "UNIVID-1B (proprietary LLM decoder)"
    vision_encoder: str = "SigLIP / LLaVA-OV default"
    max_frames: int = 32
    caption_max_words: int = 150
    training_stages: tuple[str, ...] = ("pretrain", "instruction_tune", "continue_ft")
    training_samples_m: dict[str, float] = field(
        default_factory=lambda: {
            "pretrain": 1.6,
            "ft_caption": 3.2,
            "ft_vqa": 2.0,
            "cft_caption": 0.1,
        }
    )
    training_gpus: str = "32× H100, ~120h"
    lite_train_videos: int = 1_000_000
    lite_pos_neg_ratio: str = "1:5"
    vkb_events: int = 100_000
    rag_top_k: int = 3
    capbench_total: int = 17210
    capbench_violative: int = 11476
    capbench_domains: tuple[str, ...] = (
        "Violence",
        "Sex Abuse",
        "Mental Health",
        "Regulated Act",
        "Integrity",
    )
    policy_heads: tuple[str, ...] = (
        "Violence",
        "Sexual Abuse",
        "Integrity",
        "Dangerous Driving",
        "Bank Account Fraud",
    )
    deployment_qps_per_h100: float = 5.7
    deployment_cost_usd_per_1m: float = 180.0
    leakage_reduction_pct: float = 42.7
    overkill_reduction_pct: float = 37.0
    brand_ads_match_pct: float = 81.0
    trend_few_shot_max: int = 50
    risk_filter_precision_threshold: float = 0.65
