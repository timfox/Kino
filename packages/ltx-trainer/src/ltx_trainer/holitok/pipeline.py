"""HoliTok framework card and paper tables (arXiv:2605.29948)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.ar_dit import generation_objective, patchify_latents, understanding_ce_loss
from ltx_trainer.holitok.encoder import encode_decode_smoke
from ltx_trainer.holitok.sampler import sampler_smoke
from ltx_trainer.holitok.torch_vae import torch_vae_smoke
from ltx_trainer.holitok.train_loop import train_loop_smoke
from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.layout import LIMITATIONS
from ltx_trainer.holitok.vae import (
    compression_ratio,
    implicit_fidelity_bound,
    latent_sequence_shape,
    stage_ii_vae_loss,
    stage_iii_loss,
)


def framework_card(cfg: HoliTokConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    cr = compression_ratio(cfg.sample_rate_hz, cfg.latent_frame_rate_hz, cfg.latent_dim, bfloat_bits=cfg.bfloat_bits)
    return {
        "name": "HoliTok",
        "paper": cfg.paper_arxiv,
        "github": cfg.github_repo,
        "idea": (
            "Continuous holistic speech tokenizer: 48 kHz → 25 Hz × 128-D latents "
            "with progressive VAE training (AE → weak KL → distill + LM supervision), "
            "then unified AR+DiT generation and understanding."
        ),
        "tokenizer": {
            "sample_rate_hz": cfg.sample_rate_hz,
            "latent_hz": cfg.latent_frame_rate_hz,
            "latent_dim": cfg.latent_dim,
            "encoder_hop": cfg.encoder_hop,
            "compression_ratio": round(cr, 2),
            "tokens_per_second": cfg.tokens_per_second,
        },
        "training_stages": [
            "Stage I: deterministic AE (500K steps, ℓ_gen spectral+adv+fm)",
            "Stage II: frozen E/G, temporal VAE bottleneck (β_low=0.1, 50K steps)",
            "Stage III: distill WavLM + x-vector + task LM supervision (β_high=7, 200K steps)",
        ],
        "downstream": {
            "architecture": "AR+DiT (Qwen2.5-0.5B + 18-layer DiT flow-matching)",
            "variants": ["HoliTok-Base (VAE latent patches)", "HoliTok-Unite (causal semantic encoder)"],
            "patch_size": cfg.patch_size,
        },
        "limitations": list(LIMITATIONS),
    }


def table_i_reconstruction() -> list[dict[str, Any]]:
    """Table 1 — LibriSpeech test-other reconstruction."""
    return [
        {"model": "Ground Truth", "cr": 1.0, "tps": None, "nb_pesq": None, "wb_pesq": None, "stoi": 1.000, "wer": 1.0, "spksim": 1.000, "emosim": 1.000, "utmos": 3.75},
        {"model": "Mel Spectrogram", "cr": 2.0, "tps": 86, "nb_pesq": 4.15, "wb_pesq": 4.05, "stoi": 0.988, "wer": 3.96, "spksim": 0.957, "emosim": 0.988, "utmos": 3.75},
        {"model": "SemanticVAE", "cr": 2.73, "tps": 40, "nb_pesq": 3.99, "wb_pesq": 3.80, "stoi": 0.969, "wer": 4.15, "spksim": 0.963, "emosim": 0.993, "utmos": 3.76},
        {"model": "MingTok-Audio", "cr": 2.19, "tps": 50, "nb_pesq": 4.23, "wb_pesq": 4.12, "stoi": 0.981, "wer": 4.27, "spksim": 0.950, "emosim": 0.992, "utmos": 3.75},
        {"model": "Vanilla VAE", "cr": 7.5, "tps": 25, "nb_pesq": 3.18, "wb_pesq": 2.65, "stoi": 0.925, "wer": 5.41, "spksim": 0.859, "emosim": 0.988, "utmos": 3.75},
        {"model": "HoliTok", "cr": 7.5, "tps": 25, "nb_pesq": 4.10, "wb_pesq": 4.01, "stoi": 0.974, "wer": 4.22, "spksim": 0.968, "emosim": 0.995, "utmos": 3.75},
    ]


def table_ii_zero_shot_tts() -> list[dict[str, Any]]:
    """Table 2 — Seed-TTS-Eval + Emergent-TTS zero-shot TTS."""
    return [
        {"model": "Semantic-VAE", "seed_en_wer": 1.42, "seed_en_sim": 0.63, "seed_zh_wer": 0.91, "seed_hard_wer": 7.53, "emotion_win": 14.3, "para_win": 44.2},
        {"model": "MingTok-Audio", "seed_en_wer": 1.84, "seed_en_sim": 0.61, "seed_zh_wer": 1.03, "seed_hard_wer": 14.75, "emotion_win": 8.4, "para_win": 39.8},
        {"model": "HoliTok", "seed_en_wer": 1.33, "seed_en_sim": 0.62, "seed_zh_wer": 0.98, "seed_hard_wer": 7.59, "emotion_win": 25.5, "para_win": 53.6},
    ]


def table_iii_unified_asr_tts() -> list[dict[str, Any]]:
    """Table 3 — unified spoken language modeling (ASR + TTS)."""
    return [
        {"model": "Semantic-VAE", "tts_en_wer": 102.32, "tts_zh_wer": 99.30, "tts_hard_wer": 97.31, "ls_clean": 9.69, "ls_other": 21.32, "aishell1": 15.81},
        {"model": "MingTok-Audio", "tts_en_wer": 51.06, "tts_zh_wer": 18.17, "tts_hard_wer": 50.35, "ls_clean": 4.62, "ls_other": 9.06, "aishell1": 5.01},
        {"model": "HoliTok-Base", "tts_en_wer": 27.85, "tts_zh_wer": 4.40, "tts_hard_wer": 30.44, "ls_clean": 6.45, "ls_other": 16.51, "aishell1": 14.92},
        {"model": "HoliTok-Unite", "tts_en_wer": 7.20, "tts_zh_wer": 1.78, "tts_hard_wer": 16.79, "ls_clean": 5.48, "ls_other": 12.65, "aishell1": 5.93},
    ]


def table_iv_downstream_config() -> dict[str, Any]:
    """Table 4 — AR+DiT downstream parameter counts."""
    return {
        "HoliTok-Base": {"audio_repr": "VAE latent", "input_dim": 128, "patch_encoder_m": 102, "llm_m": 494, "dit_m": 345, "total_m": 942},
        "HoliTok-Unite": {"audio_repr": "causal semantic", "input_dim": 1536, "patch_encoder_m": 1, "llm_m": 494, "dit_m": 345, "total_m": 842, "semantic_encoder_m": 680},
    }


def table_vi_tokenizer_params() -> dict[str, Any]:
    """Table 6 — tokenizer-side module parameters."""
    return {
        "HoliTok-Base": {"encoder_m": 36, "decoder_m": 128, "vae_bottleneck_m": 17, "semantic_encoder_m": 0, "total_m": 181},
        "HoliTok-Unite": {"encoder_m": 36, "decoder_m": 128, "vae_bottleneck_m": 17, "semantic_encoder_m": 680, "total_m": 861},
    }


def table_vii_optimizer() -> dict[str, Any]:
    """Table 7 — tokenizer optimization settings."""
    return {
        "optimizer": "AdamW",
        "lr_init": 1e-4,
        "lr_floor": 1e-6,
        "betas": (0.8, 0.99),
        "beta_low_stage_ii": 0.1,
        "beta_high_stage_iii": 7.0,
        "lambda_spec": 45.0,
        "lambda_wavlm_distill": 1.0,
        "lambda_xvector_distill": 1.0,
        "lambda_sup": 1.0,
    }


def headline_results() -> dict[str, Any]:
    return {
        "compression_ratio": "7.5× @ 48 kHz (25 TPS, 128-D float latents)",
        "reconstruction": "Best SPKSIM/EMOSIM among compact continuous reps (Table 1)",
        "zero_shot_tts": "Highest Emergent-TTS win rates (Table 2)",
        "unified_modeling": "Only representation robust in AR+DiT without extra tricks (Table 3)",
        "holitok_unite_tts_avg_wer": "8.59% vs 20.90% HoliTok-Base",
    }


def pipeline_demo(cfg: HoliTokConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    rng = np.random.default_rng(seed)
    num_samples = cfg.encoder_hop * 40
    wave = rng.standard_normal(num_samples) * 0.1
    recon = wave + 0.02 * rng.standard_normal(num_samples)
    t, d = latent_sequence_shape(num_samples, cfg=cfg)
    latents = rng.standard_normal((t, d))
    mean = 0.1 * rng.standard_normal((t, d))
    log_var = -1.0 + 0.05 * rng.standard_normal((t, d))
    s2 = stage_ii_vae_loss(wave, recon, mean, log_var, beta_low=cfg.beta_low, cfg=cfg)
    s3 = stage_iii_loss(
        wave,
        recon,
        mean,
        log_var,
        teacher_frame=rng.standard_normal(d * 2),
        teacher_utt=rng.standard_normal(d),
        latent_pred=latents,
        beta_high=cfg.beta_high,
        cfg=cfg,
    )
    patches = patchify_latents(latents, patch_size=cfg.patch_size)
    v_tgt = rng.standard_normal(patches.shape)
    v_pred = v_tgt + 0.05 * rng.standard_normal(v_tgt.shape)
    gen = generation_objective(v_pred, v_tgt, eos_logits=rng.standard_normal(4), eos_targets=np.array([0, 0, 0, 1]))
    und = understanding_ce_loss(rng.standard_normal((6, 32)), np.array([1, 5, 12, 3, 8, 2]))
    cr = compression_ratio(cfg.sample_rate_hz, cfg.latent_frame_rate_hz, cfg.latent_dim, bfloat_bits=cfg.bfloat_bits)
    bound = implicit_fidelity_bound(epsilon_ae=0.01, delta_shift=0.002)
    enc = encode_decode_smoke(cfg)
    samp = sampler_smoke(cfg, seed=seed)
    torch_vae = torch_vae_smoke(cfg, seed=seed)
    stage_ii = train_loop_smoke(cfg, seed=seed)
    return {
        "compression_ratio": round(cr, 2),
        "latent_shape": [t, d],
        "num_patches": int(patches.shape[0]),
        "stage_ii_total": round(s2["total"], 4),
        "stage_iii_total": round(s3["total"], 4),
        "generation_total": round(gen["total"], 4),
        "understanding_ce": round(und, 4),
        "fidelity_bound": round(bound, 6),
        "vae_encode_shape_ok": enc["latent_shape"] == enc["expected_shape"],
        "sampler_generation_finite": samp["generation_finite"],
        "torch_vae_backend": torch_vae.get("backend", "numpy"),
        "torch_available": torch_vae.get("torch_available", False),
        "stage_ii_loss_decreased": stage_ii.get("stage_ii_loss_decreased", stage_ii.get("loss_decreased")),
        "stage_iii_ran": stage_ii.get("stage_iii_ran", False),
        "stage_iii_distill_improved": stage_ii.get("stage_iii_distill_improved", False),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    demo = pipeline_demo(seed=seed)
    return {
        "headline": headline_results(),
        "demo": demo,
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_reconstruction": table_i_reconstruction(),
        "table_ii_zero_shot_tts": table_ii_zero_shot_tts(),
        "table_iii_unified_asr_tts": table_iii_unified_asr_tts(),
        "table_iv_downstream_config": table_iv_downstream_config(),
        "table_vi_tokenizer_params": table_vi_tokenizer_params(),
        "table_vii_optimizer": table_vii_optimizer(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
