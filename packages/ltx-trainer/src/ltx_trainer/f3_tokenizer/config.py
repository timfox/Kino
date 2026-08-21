"""F3-Tokenizer — Zhou et al., arXiv:2606.06357."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class F3TokenizerConfig:
    paper_arxiv: str = "arXiv:2606.06357"
    title: str = "F3-Tokenizer: Taming Audio Autoencoder Latents for Understanding and Generation"
    framework: str = "F3-Tokenizer"
    affiliations: tuple[str, ...] = ("Nanjing University", "WeNet Open Source Community")

    # Architecture (§3)
    backbone: str = "SpectroStream-style STFT autoencoder"
    latent_dim: int = 64
    ae_token_rate_hz: float = 25.0
    generator_token_rate_hz: float = 12.5
    noise_gamma: float = 1.0
    patch_frames: int = 2

    # Stage 0 — reconstruction (Table 1, autoencoder w/ norm + noise)
    recon_mcd_aishell3: float = 2.33
    recon_mcd_librispeech: float = 3.27
    recon_pesq_aishell3: float = 3.07
    recon_pesq_librispeech: float = 3.21
    recon_visqol_aishell3: float = 4.41
    recon_visqol_librispeech: float = 4.68
    recon_fd_openl3_audiocaps: float = 15.24
    recon_kl_passt_audiocaps: float = 0.2164

    # Table 1 — VibeVoice σ-VAE baseline (AudioCaps FD)
    vibevoice_fd_openl3_audiocaps: float = 51.10
    vibevoice_token_rate_hz: float = 7.5

    # Table 2 — probing highlights (F3-Tokenizer, %)
    probe_asv2015: float = 99.65
    probe_fsc: float = 94.86
    probe_librispeech_100h: float = 96.00
    probe_fsd18_kaggle: float = 81.38
    probe_gtzan: float = 85.00

    # Table 2 — ablations
    probe_no_rq_fsc: float = 88.60
    probe_no_llm_fsc: float = 84.20
    probe_no_repr_fsc: float = 1.10

    # Table 3 — TTS (Seed-zh / Seed-en)
    tts_seed_zh_cer: float = 0.90
    tts_seed_zh_sim: float = 0.76
    tts_seed_en_wer: float = 1.88
    tts_seed_en_sim: float = 0.68

    # Table 3 — TTA (AudioCaps)
    tta_fd_openl3: float = 62.700
    tta_kl_passt: float = 1.520
    tta_clap: float = 0.438

    # Table 3 — Ming-omni-tts-16.8B TTA baseline
    ming_tta_fd_openl3: float = 65.918
    ming_tta_clap: float = 0.424

    # Downstream LLM size
    downstream_llm_params_b: float = 4.0
