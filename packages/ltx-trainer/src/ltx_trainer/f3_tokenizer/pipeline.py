"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.f3_tokenizer.config import F3TokenizerConfig
from ltx_trainer.f3_tokenizer.tokenizer import (
    channel_normalize,
    flow_matching_mse,
    llm_ce_loss,
    noise_perturb,
    patch_targets,
    rq_mtp_loss,
)


def framework_card(cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    c = cfg or F3TokenizerConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "unified_audio_tokenizer",
        "f3": ("fidelity", "high_dimensional_representations", "flow_generation"),
        "components": [
            "Normalized noise-regularized SpectroStream AE bottleneck (z, D=64)",
            "Latent-side representation encoder (u) with RQ-MTP",
            "Frozen-LLM supervision on downsampled audio tokens",
            "Patch DiT flow head predicting continuous z patches",
        ],
        "token_rates_hz": {
            "ae_latent": c.ae_token_rate_hz,
            "generator": c.generator_token_rate_hz,
        },
        "headline": headline_results(c),
    }


def headline_results(cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    c = cfg or F3TokenizerConfig()
    return {
        "recon_fd_openl3_audiocaps": c.recon_fd_openl3_audiocaps,
        "beats_vibevoice_fd": c.recon_fd_openl3_audiocaps < c.vibevoice_fd_openl3_audiocaps,
        "probe_librispeech_100h": c.probe_librispeech_100h,
        "tts_seed_zh_cer": c.tts_seed_zh_cer,
        "tts_seed_en_wer": c.tts_seed_en_wer,
        "tta_clap": c.tta_clap,
        "beats_ming_tta_clap": c.tta_clap > c.ming_tta_clap,
    }


def table1_reconstruction(cfg: F3TokenizerConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — reconstruction metrics (selected rows)."""
    c = cfg or F3TokenizerConfig()
    return [
        {
            "model": "VibeVoice (σ-VAE)",
            "token_rate_hz": c.vibevoice_token_rate_hz,
            "mcd_aishell3": 5.19,
            "mcd_librispeech": 3.90,
            "pesq_aishell3": 2.93,
            "pesq_librispeech": 3.01,
            "fd_openl3_audiocaps": c.vibevoice_fd_openl3_audiocaps,
        },
        {
            "model": "Autoencoder",
            "token_rate_hz": c.ae_token_rate_hz,
            "mcd_aishell3": 2.58,
            "mcd_librispeech": 3.41,
            "pesq_aishell3": 2.96,
            "pesq_librispeech": 3.22,
            "fd_openl3_audiocaps": 31.40,
        },
        {
            "model": "Autoencoder w/ norm + noise",
            "token_rate_hz": c.ae_token_rate_hz,
            "mcd_aishell3": c.recon_mcd_aishell3,
            "mcd_librispeech": c.recon_mcd_librispeech,
            "pesq_aishell3": c.recon_pesq_aishell3,
            "pesq_librispeech": c.recon_pesq_librispeech,
            "visqol_aishell3": c.recon_visqol_aishell3,
            "visqol_librispeech": c.recon_visqol_librispeech,
            "fd_openl3_audiocaps": c.recon_fd_openl3_audiocaps,
            "kl_passt_audiocaps": c.recon_kl_passt_audiocaps,
            "best_recon": True,
        },
    ]


def table2_probing(cfg: F3TokenizerConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — frozen-representation probing (selected tasks)."""
    c = cfg or F3TokenizerConfig()
    return [
        {"dataset": "ASV2015", "whisper": 96.60, "ming_u": 98.70, "no_repr": 49.80, "w/o_rq": 94.70, "w/o_llm": 96.10, "f3": c.probe_asv2015},
        {"dataset": "FSC", "whisper": 77.60, "ming_u": 98.58, "no_repr": c.probe_no_repr_fsc, "w/o_rq": c.probe_no_rq_fsc, "w/o_llm": c.probe_no_llm_fsc, "f3": c.probe_fsc},
        {"dataset": "LibriSpeech-100h", "whisper": 81.50, "ming_u": 93.45, "no_repr": 0.00, "w/o_rq": 86.40, "w/o_llm": 80.80, "f3": c.probe_librispeech_100h},
        {"dataset": "FSD18-Kaggle", "whisper": 24.10, "ming_u": 40.81, "no_repr": 6.10, "w/o_rq": 63.50, "w/o_llm": 75.40, "f3": c.probe_fsd18_kaggle},
        {"dataset": "GTZAN", "whisper": 62.20, "ming_u": 71.17, "no_repr": 14.20, "w/o_rq": 72.60, "w/o_llm": 80.40, "f3": c.probe_gtzan},
    ]


def table3_generation(cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    """Table 3 — TTS and TTA generation."""
    c = cfg or F3TokenizerConfig()
    return {
        "tts": [
            {"model": "CosyVoice 3-1.5B", "token_rate": "25 Hz", "seed_zh_cer": 1.12, "seed_en_wer": 2.21},
            {"model": "VibeVoice-1.5B", "token_rate": "7.5 Hz", "seed_zh_cer": 1.16, "seed_en_wer": 3.04},
            {
                "model": f"F3-Tokenizer-LLM ({c.downstream_llm_params_b:.0f}B)",
                "token_rate": f"{c.ae_token_rate_hz:g}→{c.generator_token_rate_hz:g} Hz",
                "seed_zh_cer": c.tts_seed_zh_cer,
                "seed_zh_sim": c.tts_seed_zh_sim,
                "seed_en_wer": c.tts_seed_en_wer,
                "seed_en_sim": c.tts_seed_en_sim,
                "best_f3": True,
            },
        ],
        "tta": [
            {"model": "Ming-omni-tts-16.8B-A3B", "fd_openl3": c.ming_tta_fd_openl3, "clap": c.ming_tta_clap},
            {
                "model": f"F3-Tokenizer-LLM ({c.downstream_llm_params_b:.0f}B)",
                "token_rate": f"{c.ae_token_rate_hz:g}→{c.generator_token_rate_hz:g} Hz",
                "fd_openl3": c.tta_fd_openl3,
                "kl_passt": c.tta_kl_passt,
                "clap": c.tta_clap,
                "best_f3": True,
            },
        ],
    }


def benchmarks_bundle(cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    c = cfg or F3TokenizerConfig()
    return {
        "table1_reconstruction": table1_reconstruction(c),
        "table2_probing": table2_probing(c),
        "table3_generation": table3_generation(c),
        "training_stages": ("stage0_ae", "stage1_repr_flow", "stage2_task"),
    }


def evaluation_demo(seed: int = 42, cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or F3TokenizerConfig()
    t, d = 32, c.latent_dim

    z0 = rng.normal(0, 1, (t, d))
    zn = channel_normalize(z0)
    z_tilde, alpha = noise_perturb(zn, gamma=c.noise_gamma, rng=rng)

    logits = [rng.random((8, 64)) for _ in range(3)]
    targets = [np.eye(64)[rng.integers(0, 64, 8)] for _ in range(3)]
    rq_loss = rq_mtp_loss(logits, targets)

    llm_probs = rng.random((6, 50))
    llm_probs /= llm_probs.sum(axis=-1, keepdims=True)
    llm_targets = np.zeros_like(llm_probs)
    llm_targets[np.arange(6), rng.integers(0, 50, 6)] = 1.0
    llm_loss = llm_ce_loss(llm_probs, llm_targets)

    patches = patch_targets(zn, c.patch_frames)
    flow_loss = flow_matching_mse(rng.normal(0, 0.1, patches[0].shape), patches[0])

    recon = next(r for r in table1_reconstruction(c) if r.get("best_recon"))
    fsc = next(r for r in table2_probing(c) if r["dataset"] == "FSC")
    tta = next(r for r in table3_generation(c)["tta"] if r.get("best_f3"))

    return {
        "normalized_latent_shape": list(zn.shape),
        "noise_alpha": alpha,
        "rq_mtp_loss": rq_loss,
        "llm_ce_loss": llm_loss,
        "num_patches": len(patches),
        "flow_mse": flow_loss,
        "norm_noise_beats_vibevoice_fd": recon["fd_openl3_audiocaps"] < c.vibevoice_fd_openl3_audiocaps,
        "full_objectives_beat_ablations": fsc["f3"] > fsc["w/o_rq"] and fsc["f3"] > fsc["w/o_llm"],
        "tta_clap": tta["clap"],
        "beats_ming_tta_clap": tta["clap"] > c.ming_tta_clap,
    }


def pipeline_demo(seed: int = 42, cfg: F3TokenizerConfig | None = None) -> dict[str, Any]:
    c = cfg or F3TokenizerConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
