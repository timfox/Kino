"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.univoice.cfm import cfm_loss, euler_sample_step, ot_interpolant, target_velocity
from ltx_trainer.univoice.conditioning import TaskModality, axis_guidance_delta, build_condition
from ltx_trainer.univoice.config import UniVoiceConfig


def framework_card(cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    c = cfg or UniVoiceConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "unified_speech_singing_tts",
        "backbone": f"Wan-DiT ({c.dit_layers}L, {c.params_b}B)",
        "conditioning": ["content (IPA phonemes)", "melody (MIDI / null token)", "timbre (ICL prompt)", "task token"],
        "training": f"CFM on {c.total_training_hours_k:.0f}k h mixed speech+singing",
        "benchmark": "UNISINGING-EVAL (12 styles, 900 samples)",
        "headline": headline_results(c),
    }


def headline_results(cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    c = cfg or UniVoiceConfig()
    return {
        "speech_per": c.speech_per,
        "singing_per": c.singing_per,
        "f5_tts_speech_per": c.f5_tts_speech_per,
        "vevo15_singing_per": c.vevo15_singing_per,
        "params_b": c.params_b,
        "eval_styles": len(c.eval_styles),
        "eval_samples": c.eval_samples,
    }


def table1_main_results(cfg: UniVoiceConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — speech and singing comparison."""
    c = cfg or UniVoiceConfig()
    return [
        {
            "model": "F5-TTS",
            "size": "0.3B",
            "speech_per": c.f5_tts_speech_per,
            "speech_sim": 72.73,
            "singing_per": None,
        },
        {
            "model": "CosyVoice3",
            "size": "0.5B",
            "speech_per": c.cosyvoice3_speech_per,
            "speech_sim": 74.94,
            "singing_per": None,
        },
        {
            "model": "Vevo1.5",
            "size": "1B",
            "speech_per": c.vevo15_speech_per,
            "speech_sim": 59.07,
            "singing_per": c.vevo15_singing_per,
        },
        {
            "model": "Soul-X-Singer",
            "size": "0.7B",
            "speech_per": None,
            "speech_sim": None,
            "singing_per": c.soul_x_singer_per,
        },
        {
            "model": "UniVoice (ours)",
            "size": f"{c.params_b}B",
            "speech_per": c.speech_per,
            "speech_sim": c.speech_sim,
            "singing_per": c.singing_per,
            "singing_sim": c.singing_sim,
            "speech_s_mos": c.speech_s_mos,
            "singing_n_mos": c.singing_n_mos,
        },
    ]


def table2_ablation(cfg: UniVoiceConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — conditioning ablations."""
    c = cfg or UniVoiceConfig()
    return [
        {
            "config": "UniVoice (Full)",
            "speech_per": c.ablation_full_speech_per,
            "singing_per": c.ablation_full_singing_per,
        },
        {
            "config": "w/o factorized cond.",
            "speech_per": c.ablation_no_factorized_speech,
            "singing_per": c.ablation_no_factorized_singing,
        },
        {
            "config": "w/o task token",
            "speech_per": c.ablation_no_task_speech,
            "singing_per": c.ablation_no_task_singing,
        },
        {
            "config": "w/o null melody token",
            "speech_per": c.ablation_no_null_speech,
            "singing_per": c.ablation_no_null_singing,
        },
        {
            "config": "w/o melody encoder",
            "speech_per": c.ablation_no_melody_enc_speech,
            "singing_per": c.ablation_no_melody_enc_singing,
        },
    ]


def unisinging_eval_summary(cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    c = cfg or UniVoiceConfig()
    return {
        "name": "UNISINGING-EVAL",
        "styles": list(c.eval_styles),
        "n_styles": len(c.eval_styles),
        "n_songs": c.eval_songs,
        "n_samples": c.eval_samples,
        "duration_hours": c.eval_duration_hours,
        "difficulty_levels": ["Minor (L1)", "Major (L2)", "Elastic (L3)"],
    }


def benchmarks_bundle(cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    c = cfg or UniVoiceConfig()
    return {
        "table1_main": table1_main_results(c),
        "table2_ablation": table2_ablation(c),
        "unisinging_eval": unisinging_eval_summary(c),
    }


def pipeline_demo(seed: int = 42, cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    """CPU stub: factorized CFM + null melody vs MIDI melody."""
    c = cfg or UniVoiceConfig()
    rng = np.random.default_rng(seed)
    dim = c.vae_latent_dim
    seq_len = 32
    x0 = rng.standard_normal((seq_len, dim))
    x1 = rng.standard_normal((seq_len, dim))
    t = 0.5
    xt = ot_interpolant(x0, x1, t)
    u = target_velocity(x0, x1)
    loss = cfm_loss(u, x0, x1)

    null_tok = rng.standard_normal(dim) * 0.1
    sp_cond = build_condition(dim=dim, seq_len=seq_len, task=TaskModality.SPEECH, null_token=null_tok, seed=seed)
    sg_cond = build_condition(dim=dim, seq_len=seq_len, task=TaskModality.SINGING, seed=seed + 1)

    v_full = u
    v_drop = u * 0.9
    v_guided = axis_guidance_delta(v_full, v_drop, c.text_guidance)
    x_out = euler_sample_step(xt, v_guided, dt=1.0 / c.ode_steps)

    return {
        "cfm_loss": round(loss, 6),
        "ode_steps": c.ode_steps,
        "speech_melody_is_null": sp_cond.melody.ndim == 1,
        "singing_melody_frames": sg_cond.melody.shape[0],
        "guided_velocity_norm": float(np.linalg.norm(v_guided)),
        "sample_latent_shape": list(x_out.shape),
        "speech_per_anchor": c.speech_per,
        "singing_per_anchor": c.singing_per,
    }


def evaluation_demo(seed: int = 42, cfg: UniVoiceConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
