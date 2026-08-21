"""GLASS GRPO LoRA style steering — Kang et al., arXiv:2606.05889."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GlassConfig:
    paper_arxiv: str = "arXiv:2606.05889"
    title: str = (
        "GLASS: GRPO-Trained LoRA for Acoustic Style Steering "
        "in Zero-Shot Text-to-Speech"
    )
    framework: str = "GLASS"

    # Backbone (Appendix A)
    backbone: str = "CosyVoice2-0.5B"
    ar_params_m: float = 495.0
    lora_params_m: float = 1.08
    lora_rank: int = 16
    lora_alpha: int = 32
    grpo_group_size: int = 8

    # GRPO hyperparams (Appendix A)
    ppo_epsilon: float = 0.2
    kl_beta: float = 0.01
    reward_eta: float = 0.5  # η in Eq. 5
    wer_gamma: float = 1.0

    # Table 1 — baseline Seed-TTS-eval test_en
    baseline_sps: float = 3.65
    baseline_f0_m: float = 120.4
    baseline_f0_f: float = 192.2
    baseline_wer: float = 2.81
    baseline_spksim: float = 0.655
    baseline_utmos: float = 3.28

    # Table 1 — GLASS LoRA adapters
    fast_sps: float = 5.59
    fast_wer: float = 3.49
    fast_spksim: float = 0.617
    slow_sps: float = 2.30
    slow_wer: float = 3.18
    slow_spksim: float = 0.650
    high_f0_m: float = 156.1
    high_f0_f: float = 241.0
    high_wer: float = 3.01
    low_f0_m: float = 108.9
    low_f0_f: float = 164.6
    low_wer: float = 3.11

    # Table 2 — interpolation WER min at α=0.5
    interp_speed_wer_min: float = 2.16
    interp_pitch_wer_min: float = 2.14
    compose_weight_default: float = 0.5

    @property
    def lora_fraction_pct(self) -> float:
        return round(100.0 * self.lora_params_m / self.ar_params_m, 2)
