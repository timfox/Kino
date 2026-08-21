"""SpectCount — spectrotemporal counting for LALMs (arXiv:2606.06907)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpectCountConfig:
    paper_arxiv: str = "arXiv:2606.06907"
    title: str = (
        "SpectCount: Spectrotemporal Counting via Synthetic Signals "
        "Improves Large Audio Language Models"
    )
    framework: str = "SpectCount"
    backbone_af3: str = "Audio Flamingo 3 (8.3B)"
    backbone_qwen2: str = "Qwen2-Audio-Instruct (8.4B)"

    # LoRA (§3.1)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 8

    # Signal generation (Table 2)
    sample_rate_hz: int = 16000
    n_max: int = 10
    cmel: int = 128
    t_min_ms: float = 40.0
    t_max_ms: float = 160.0
    attack_ms: float = 3.0
    release_ms: float = 10.0
    t_gap_ms: float = 40.0
    t_total_s: float = 30.0
    amp_min: float = 0.1
    amp_max: float = 0.9
    noise_min: float = 1e-4
    noise_max: float = 1e-3

    # Table 1 — Audio Flamingo 3 SpectCount (reproduced base → ours)
    mmau_mini_sound: float = 83.18
    mmau_mini_music: float = 77.54
    mmau_mini_speech: float = 74.47
    mmau_mini_total: float = 78.40
    mmau_mini_base: float = 73.90
    mmau_test_total: float = 73.79
    mmau_test_base: float = 72.36
    mmar: float = 56.30
    mmar_base: float = 52.90
    mmsu: float = 63.18
    mmsu_base: float = 61.92
    air_bench: float = 64.85
    air_bench_base: float = 64.16

    # Table 3 ablation (MMAU-test-mini total %)
    ablation_freq_only: float = 74.7
    ablation_time_only: float = 77.2
    ablation_encoder_only: float = 75.2
    ablation_llm_only: float = 77.0
