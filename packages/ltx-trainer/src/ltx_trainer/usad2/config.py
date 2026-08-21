"""USAD 2.0 — universal audio distillation (Chang et al., arXiv:2606.06444)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usad2Config:
    paper_arxiv: str = "arXiv:2606.06444"
    title: str = "USAD 2.0: Scaling Representation Distillation for Universal Audio Understanding"
    framework: str = "USAD 2.0"
    collection_url: str = "https://hf.co/collections/MIT-SLS/usad2"

    # §2.2 domain-aware distillation
    num_teachers_ssl: int = 3
    domain_aware_alpha: float = 10.0
    ssl_teachers: tuple[str, ...] = ("WavLM", "ATST-Frame", "MuQ")
    supervised_teachers: tuple[str, ...] = ("Whisper Large-v3", "Audio Flamingo 3")

    # Training (§3.1, Tab. 6)
    stage1_updates: int = 600_000
    stage2_updates: int = 50_000
    train_hours_total: float = 208_607.0  # multi-domain SSL stage

    # Model sizes
    sizes: tuple[str, ...] = ("Small", "Base", "Large", "XLarge", "XXLarge+")
    xxlarge_params_m: float = 1036.0
    xxlarge_layers: int = 48
    xxlarge_framerate_hz: int = 25

    # Table 1 — benchmark averages (encoder-only)
    xxlarge_plus_hear_avg: float = 84.4
    xxlarge_plus_marble_avg: float = 75.6
    xxlarge_plus_xares_track_a: float = 0.783
    xxlarge_plus_xares_track_b: float = 0.624
    xlarge_plus_hear_avg: float = 84.4
    xlarge_plus_xares_track_a: float = 0.772
    xlarge_plus_xares_track_b: float = 0.611
    spear_xlarge_hear_avg: float = 82.6
    spear_xlarge_xares_track_a: float = 0.782

    # Table 2 ablation (Small 25M, NSynth pitch Acc)
    usad2_nsynth_acc: float = 70.3
    usad_v1_nsynth_acc: float = 55.1
    wo_domain_aware_nsynth_acc: float = 69.1
    wo_music_teacher_nsynth_acc: float = 49.1

    # Tab. 4 efficiency — XXLarge 25Hz
    xxlarge_rtf_25hz: float = 0.0026
    xxlarge_gpu_mem_gb: float = 2.4

    domains: tuple[str, ...] = ("speech", "general_audio", "music")
    benchmarks: tuple[str, ...] = ("HEAR", "MARBLE", "XARES-LLM")
