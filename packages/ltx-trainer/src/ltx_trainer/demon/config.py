"""DEMON: Diffusion Engine for Musical Orchestrated Noise (arXiv:2605.28657)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PropagationClass(str, Enum):
    PER_REQUEST = "per_request"
    SCHEDULE_MIGRATED = "schedule_migrated"
    PER_STEP_SHARED = "per_step_shared_mutable"
    MODEL_WEIGHT = "model_weight"


@dataclass
class DemonConfig:
    paper_arxiv: str = "arXiv:2605.28657"
    base_model: str = "ACE-Step 1.5 turbo"
    denoising_steps: int = 8
    latent_hz: float = 25.0
    latent_frames_60s: int = 1500
    latent_dim: int = 64
    # Ring buffer
    production_depth: int = 4
    max_depth: int = 8
    # Throughput (RTX 5090, 60s)
    gens_per_sec_depth8: float = 12.3
    gens_per_sec_depth4: float = 11.3
    tick_ms_depth8: float = 81.1
    tick_ms_depth4: float = 42.8
    e2e_ms_depth8_3s_window: float = 88.0
    # VAE windowed decode
    vae_full_decode_ms: float = 56.0
    vae_windowed_3s_ms: float = 7.0
    vae_speedup_3s: float = 8.0
    vae_overlap_s: float = 0.5
    vae_rf_convergence_ms: float = 333.0
    # Table 13 ablation
    per_slot_completion_rate_sweep: float = 1.0
    global_reset_completion_rate_sweep: float = 0.017
    # Table 12 first-effect (depth 8)
    denoise_first_effect_ms: float = 648.0
    sde_curve_first_effect_ms: float = 81.0
    lora_refit_s: float = 1.2
    # Table 14 depth tradeoff
    depth4_first_effect_ms: float = 471.0
    depth8_first_effect_ms: float = 649.0
