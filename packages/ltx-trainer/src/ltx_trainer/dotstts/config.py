"""dots.tts — continuous AR TTS foundation model (arXiv:2606.07080)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DotsttsConfig:
    paper_arxiv: str = "arXiv:2606.07080"
    title: str = "dots.tts: Continuous Autoregressive TTS Foundation Model"
    framework: str = "dots.tts"
    license: str = "Apache-2.0"
    params_b: float = 2.0
    github: str = "https://github.com/rednote-hilab/dots.tts"
    hf_collection: str = "https://huggingface.co/collections/rednote-hilab/dotstts"

    # AudioVAE (§2.2)
    sample_rate_hz: int = 48000
    latent_dim: int = 128
    latent_fps: float = 25.0
    semantic_fps: float = 6.25
    patch_frames: int = 4
    llm_init: str = "Qwen2.5-1.5B-Base"

    # Training corpus (§3.1)
    training_hours_m: float = 1.5

    # Inference (§3.4, MF NFE=4)
    mf_nfe: int = 4
    cfg_gamma: float = 1.2
    ttfp_plain_ms: float = 85.4
    ttfp_interleaved_ms: float = 54.4
    rtf_plain: float = 0.231
    rtf_interleaved: float = 0.245

    # Table 1 — LibriSpeech test-other reconstruction (dots.tts VAE)
    vae_pesq_nb: float = 4.09
    vae_pesq_wb: float = 3.95
    vae_stoi: float = 0.973
    vae_utmos: float = 3.75
    vae_sim: float = 0.969
    vae_wer_pct: float = 4.14

    # Table 2 — Seed-TTS-Eval SOAR row
    seed_en_wer: float = 1.30
    seed_en_sim: float = 77.1
    seed_zh_wer: float = 0.94
    seed_zh_sim: float = 81.0
    seed_zh_hard_wer: float = 6.60
    seed_zh_hard_sim: float = 79.5
    seed_avg_wer: float = 2.95
    seed_avg_sim: float = 79.2

    # Table 2 — Pretrain average (best WER)
    seed_pretrain_avg_wer: float = 2.92
    seed_pretrain_avg_sim: float = 78.8

    # Table 2 — MF NFE=4
    seed_mf4_avg_wer: float = 2.94
    seed_mf4_avg_sim: float = 78.2

    # Table 3 — MiniMax multilingual average
    minimax_avg_wer: float = 6.8
    minimax_avg_sim_soar: float = 83.9

    # Table 5 — EmergentTTS-Eval SOAR
    emergent_overall: float = 47.6
    emergent_syntax: float = 65.7
    emergent_wer: float = 10.45

    # SOAR post-training (§3.2.3)
    soar_lambda_aux: float = 1.0
    soar_gamma: float = 1.2
    soar_aux_samples: int = 6

    checkpoints: list[str] = field(
        default_factory=lambda: ["Pretrain", "SOAR", "MF"]
    )

    # Upstream Hugging Face releases (rednote-hilab/dotstts collection)
    hf_checkpoints: dict[str, str] = field(
        default_factory=lambda: {
            "base": "rednote-hilab/dots.tts-base",
            "soar": "rednote-hilab/dots.tts-soar",
            "mf": "rednote-hilab/dots.tts-mf",
        }
    )
    default_checkpoint: str = "soar"
    upstream_clone_url: str = "https://github.com/rednote-hilab/dots.tts.git"
