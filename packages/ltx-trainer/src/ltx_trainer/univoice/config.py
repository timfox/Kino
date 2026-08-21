"""UniVoice unified speech/singing CFM — Zheng et al., arXiv:2606.05852."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UniVoiceConfig:
    paper_arxiv: str = "arXiv:2606.05852"
    title: str = "UniVoice: A Unified Model for Speech and Singing Voice Generation"
    framework: str = "UniVoice"

    # Architecture (Table 3 / Sec 3.2)
    dit_layers: int = 24
    hidden_dim: int = 1024
    attention_heads: int = 16
    params_m: float = 329.0
    vae: str = "Song Bloom VAE"
    vae_latent_dim: int = 48
    vae_frame_rate_hz: int = 25
    sample_rate_hz: int = 48000
    ode_steps: int = 32
    text_guidance: float = 5.0
    audio_melody_guidance: float = 1.0

    # Training (Sec 3.5)
    speech_hours_k: float = 30.0
    singing_hours_k: float = 35.0
    cfg_dropout: float = 0.1

    # Table 1 — UniVoice main results
    speech_per: float = 5.26
    speech_sim: float = 67.42
    speech_s_mos: float = 3.76
    speech_n_mos: float = 3.07
    singing_per: float = 16.22
    singing_sim: float = 35.70
    singing_s_mos: float = 3.19
    singing_n_mos: float = 3.25

    # Table 1 — baselines (speech)
    f5_tts_speech_per: float = 5.21
    cosyvoice3_speech_per: float = 5.30
    vevo15_speech_per: float = 14.10
    vevo15_singing_per: float = 45.07  # paper Table 1; text cites 24.72% elsewhere
    soul_x_singer_per: float = 26.22

    # Table 2 — ablation (full model)
    ablation_full_speech_per: float = 5.26
    ablation_full_singing_per: float = 16.22
    ablation_no_factorized_speech: float = 12.31
    ablation_no_factorized_singing: float = 23.45
    ablation_no_task_speech: float = 8.34
    ablation_no_task_singing: float = 19.92
    ablation_no_null_speech: float = 7.86
    ablation_no_null_singing: float = 18.64
    ablation_no_melody_enc_speech: float = 6.27
    ablation_no_melody_enc_singing: float = 23.21

    # UNISINGING-EVAL (Sec 4)
    eval_styles: tuple[str, ...] = (
        "Pop",
        "Hip-Hop/Rap",
        "Pop Electronic",
        "Rock",
        "Country",
        "City Pop",
        "R&B",
        "Afrobeats",
        "Bossa Nova",
        "Jazz",
        "Funk",
        "Neo-Soul",
    )
    eval_songs: int = 60
    eval_samples: int = 900
    eval_duration_hours: float = 2.0

    @property
    def total_training_hours_k(self) -> float:
        return self.speech_hours_k + self.singing_hours_k

    @property
    def params_b(self) -> float:
        return round(self.params_m / 1000.0, 2)
