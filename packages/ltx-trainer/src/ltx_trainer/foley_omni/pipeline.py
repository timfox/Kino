"""Foley-Omni framework card, benchmarks, and smoke demos (arXiv:2606.03672)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.foley_omni.conditioning import format_structured_prompt, toy_project_conditions
from ltx_trainer.foley_omni.config import FoleyOmniConfig
from ltx_trainer.foley_omni.curation import (
    ClipFilterThresholds,
    passes_visual_audio_filters,
    verify_component_labels,
)
from ltx_trainer.foley_omni.layout import LIMITATIONS
from ltx_trainer.foley_omni.training import curriculum_schedule, training_step_smoke


def framework_card(cfg: FoleyOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FoleyOmniConfig()
    return {
        "name": "Foley-Omni",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_url,
        "task": "Unified audio: TTA/TTS/TTM/V2A/VisualTTS → complete video soundtrack (V2ST)",
        "idea": (
            "Single DiT generates mixed speech + SFX + music in one latent flow; structured "
            "[WORDS]/[AUDIO]/[MUSIC] tags + CLIP semantics + Synchformer sync (additive + cross-attn); "
            "curriculum from text-only tasks to V2ST finetune."
        ),
        "conditioning": {
            "text": "UM-T5 on structured fields",
            "video_semantic": cfg.visual_semantic,
            "video_sync": cfg.visual_sync,
            "injection": "C_uni cross-attn + Z_sync additive to latents",
        },
        "backbone": cfg.backbone,
        "data": {
            "curation": "~2.0M video-audio-text tuples after pipeline",
            "training_total_m": cfg.training_pairs_m,
            "benchmark": f"V2ST-Bench ({cfg.v2st_bench_size} clips)",
        },
        "curriculum": curriculum_schedule(cfg),
        "defaults": cfg.__dict__,
    }


def table_v2st_main() -> list[dict[str, float | str]]:
    """Table 1 — V2ST-Bench complete soundtrack generation."""
    return [
        {
            "method": "GT",
            "clap": 0.30,
            "ib": 0.36,
            "wer": 8.03,
            "desync": 0.14,
            "a_mos": 4.33,
            "s_mos": 4.37,
            "t_mos": 4.42,
        },
        {
            "method": "MMAudio + CosyVoice 3 + AudioX",
            "clap": 0.26,
            "ib": 0.25,
            "wer": 10.57,
            "desync": 0.85,
            "a_mos": 2.99,
            "s_mos": 3.01,
            "t_mos": 2.37,
        },
        {
            "method": "MMAudio + LipVoicer + AudioX",
            "clap": 0.22,
            "ib": 0.16,
            "wer": 37.84,
            "desync": 0.26,
            "a_mos": 2.09,
            "s_mos": 2.31,
            "t_mos": 2.51,
        },
        {
            "method": "Foley-Omni",
            "clap": 0.27,
            "ib": 0.26,
            "wer": 7.59,
            "desync": 0.16,
            "a_mos": 3.92,
            "s_mos": 4.13,
            "t_mos": 4.14,
        },
    ]


def table_text_tasks() -> list[dict[str, float | str]]:
    """Table 2 — TTA / TTS / TTM."""
    return [
        {"model": "AudioLDM 2", "clap_tta": 0.43, "wer_tts": None, "clap_ttm": None},
        {"model": "MMAudio", "clap_tta": 0.49, "wer_tts": None, "clap_ttm": None},
        {"model": "CosyVoice 3", "clap_tta": None, "wer_tts": 1.96, "clap_ttm": None},
        {"model": "AudioX", "clap_tta": 0.44, "wer_tts": None, "clap_ttm": 0.386},
        {"model": "UniFlow-Audio", "clap_tta": 0.46, "wer_tts": 2.19, "clap_ttm": 0.241},
        {"model": "Foley-Omni", "clap_tta": 0.46, "wer_tts": 2.31, "clap_ttm": 0.374},
    ]


def table_vgg_sound() -> list[dict[str, float | str]]:
    """Table 3 — V2A on VGGSound."""
    return [
        {
            "model": "VTA-LDM",
            "fd": 18.77,
            "fd_passt": 827.57,
            "kl": 3.45,
            "clap": -0.04,
            "is": 2.01,
            "ib": 0.06,
            "desync": 1.17,
        },
        {
            "model": "FoleyCrafter",
            "fd": 2.54,
            "fd_passt": 137.52,
            "kl": 2.32,
            "clap": 0.19,
            "is": 15.02,
            "ib": 0.25,
            "desync": 1.23,
        },
        {
            "model": "MMAudio",
            "fd": 1.27,
            "fd_passt": 102.93,
            "kl": 1.99,
            "clap": 0.21,
            "is": 15.76,
            "ib": 0.31,
            "desync": 0.57,
        },
        {
            "model": "HunyuanVideo-Foley",
            "fd": 2.18,
            "fd_passt": 79.07,
            "kl": 2.02,
            "clap": 0.23,
            "is": 15.49,
            "ib": 0.32,
            "desync": 0.55,
        },
        {
            "model": "Foley-Omni",
            "fd": 1.57,
            "fd_passt": 101.40,
            "kl": 1.92,
            "clap": 0.21,
            "is": 14.00,
            "ib": 0.28,
            "desync": 0.50,
        },
    ]


def table_ablation() -> list[dict[str, float | str]]:
    """Table 6 — curriculum and Z_sync ablations."""
    return [
        {
            "variant": "Single-stage training",
            "fd_vgg": 1.73,
            "wer_grid": 27.4,
            "ib_v2st": 0.24,
            "wer_v2st": 29.29,
        },
        {
            "variant": "w/o Z_sync",
            "fd_vgg": 2.21,
            "wer_grid": 18.9,
            "ib_v2st": 0.22,
            "wer_v2st": 12.40,
        },
        {
            "variant": "Full model",
            "fd_vgg": 1.57,
            "wer_grid": 15.3,
            "ib_v2st": 0.26,
            "wer_v2st": 7.59,
        },
    ]


def v2st_bench_composition() -> list[dict[str, int | str]]:
    """Table 8 — V2ST-Bench audio combination counts."""
    return [
        {"combination": "Speech + Sound Effects", "count": 150},
        {"combination": "Speech + Music", "count": 120},
        {"combination": "Speech + Sound Effects + Music", "count": 30},
        {"combination": "Total", "count": 300},
    ]


def training_data_groups() -> list[dict[str, float | str]]:
    """Table 9 — training sources by task group."""
    return [
        {"group": "TTS", "datasets": "LJSpeech, LibriTTS, internal", "hours": 1253},
        {"group": "TTA", "datasets": "AudioCaps, Freesound", "hours": 912},
        {"group": "TTM", "datasets": "MusicCaps, MusicBench, AudioSet music", "hours": 139},
        {"group": "VisualTTS", "datasets": "Chem, GRID, LRS2, SpeakerVid, Talkvid", "hours": 1980},
        {"group": "V2A", "datasets": "VGGSound, Kling-Foley, internal", "hours": 403},
        {"group": "V2ST", "datasets": "SpeakerVid, internal (curated)", "hours": 216},
    ]


def pipeline_demo(cfg: FoleyOmniConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FoleyOmniConfig()
    prompt = format_structured_prompt(
        words="Usually, my health advice gets cut off by the meeting bell.",
        audio="gentle rhythmic typing on keyboards, soft office hum",
        music="",
    )
    cond = toy_project_conditions(seed=7)
    train = training_step_smoke(seed=8)
    import numpy as np

    stems = {
        "words": np.ones(8000) * 0.1,
        "audio": np.ones(8000) * 0.02,
        "music": np.ones(8000) * 1e-5,
    }
    verified = verify_component_labels(
        {"words": True, "audio": True, "music": True},
        stems,
        cfg,
    )
    filter_ok = passes_visual_audio_filters(
        resolution_p=720,
        bitrate_mbps=2.5,
        motion_score=1.2,
        audio_quality=0.75,
        ib_score=0.35,
        sync_score=0.25,
        thresholds=ClipFilterThresholds(
            min_resolution_p=cfg.filter_min_resolution_p,
            ib_score_min=cfg.filter_ib_min,
            sync_score_min=cfg.filter_sync_min,
            audio_quality_min=cfg.filter_audio_quality_min,
        ),
    )
    return {
        "structured_prompt_chars": len(prompt),
        "c_uni_dim": int(cond["c_uni"].shape[-1]),
        "x_tilde_norm": float(np.linalg.norm(cond["x_tilde"])),
        "flow_matching_loss": train["loss"],
        "bandit_verified": verified,
        "filter_pass": filter_ok,
        "curriculum_stages": len(curriculum_schedule(cfg)),
    }


def evaluation_demo(cfg: FoleyOmniConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    rows = {r["method"]: r for r in table_v2st_main()}
    demo["foley_omni_v2st_wer"] = rows["Foley-Omni"]["wer"]
    demo["baseline_cosy_wer"] = rows["MMAudio + CosyVoice 3 + AudioX"]["wer"]
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_v2st_main": table_v2st_main(),
        "table_text_tasks": table_text_tasks(),
        "table_vgg_sound": table_vgg_sound(),
        "table_ablation": table_ablation(),
        "v2st_bench_composition": v2st_bench_composition(),
        "training_data_groups": training_data_groups(),
        "curriculum": curriculum_schedule(),
    }


def knowledge_summary() -> dict[str, Any]:
    """Agent-facing facts bundle."""
    return {
        "arxiv": "2606.03672",
        "benchmarks": ["V2ST-Bench", "VGGSound", "AudioCaps", "LibriSpeech-PC", "MusicCaps", "GRID", "LRS2"],
        "baselines_compositional": [
            "MMAudio + CosyVoice 3 + AudioX",
            "MMAudio + LipVoicer + AudioX",
        ],
        "key_metrics": ["CLAP", "IB", "WER", "DeSync", "A-MOS", "S-MOS", "T-MOS"],
        "tags": ["WORDS", "AUDIO", "MUSIC"],
    }
