"""HoliTok: continuous holistic speech tokenization (arXiv:2605.29948)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HoliTokConfig:
    paper_arxiv: str = "arXiv:2605.29948"
    github_repo: str = "https://github.com/bovod-sjtu/HoliTok"
    sample_rate_hz: int = 48000
    latent_frame_rate_hz: float = 25.0
    latent_dim: int = 128
    encoder_hop: int = 1920
    bfloat_bits: int = 32
    compression_ratio: float = 7.5
    tokens_per_second: float = 25.0
    # Stage training
    stage_i_steps: int = 500_000
    stage_ii_steps: int = 50_000
    stage_iii_steps: int = 200_000
    beta_low: float = 0.1
    beta_high: float = 7.0
    lambda_wavlm_distill: float = 1.0
    lambda_xvector_distill: float = 1.0
    lambda_sup: float = 1.0
    lambda_spec: float = 45.0
    lambda_adv: float = 1.0
    lambda_fm: float = 2.0
    # Downstream AR+DiT
    patch_size: int = 4
    llm_params_m: float = 494.0
    dit_params_m: float = 345.0
    dit_layers: int = 18
    flow_matching_lambda_eos: float = 1.0
    tokenizer_base_params_m: float = 181.0
    tokenizer_unite_params_m: float = 861.0
