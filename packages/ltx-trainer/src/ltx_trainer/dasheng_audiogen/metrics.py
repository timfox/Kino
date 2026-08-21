"""Paper table anchors (Dasheng AudioGen arXiv:2605.27838)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig


def table_1_capability_comparison() -> list[dict[str, Any]]:
    return [
        {"model": "MusicGen", "sound_effect": False, "music": True, "speech": False, "audio_scene": False},
        {"model": "TangoFlux", "sound_effect": True, "music": True, "speech": False, "audio_scene": False},
        {"model": "Qwen3-TTS", "sound_effect": False, "music": False, "speech": True, "audio_scene": False},
        {"model": "UniFlow-Audio", "sound_effect": True, "music": True, "speech": True, "audio_scene": False},
        {"model": "Dasheng AudioGen", "sound_effect": True, "music": True, "speech": True, "audio_scene": True},
    ]


def table_2_standard_benchmarks(cfg: DashengAudioGenConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or DashengAudioGenConfig()
    return [
        {
            "benchmark": "AudioCaps",
            "ours_fad": cfg.audiocaps_fad,
            "tangoflux_fad": 2.26,
            "audioldm2_fad": 2.29,
        },
        {
            "benchmark": "MusicCaps",
            "ours_fad": cfg.musiccaps_fad,
            "musicgen_fad": 3.80,
            "audiox_fad": 1.42,
        },
        {
            "benchmark": "LibriTTS",
            "ours_wer_pct": cfg.librispeech_wer_pct,
            "ours_utmos": cfg.librispeech_utmos,
            "qwen3_tts_wer_pct": 2.15,
            "qwen3_tts_utmos": 3.40,
        },
    ]


def table_3_mecat_mixed(cfg: DashengAudioGenConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or DashengAudioGenConfig()
    return [
        {"category": "0MA", "ours_fad": 3.25, "expert_fad": 5.55},
        {"category": "S0A", "ours_fad": 1.75, "expert_fad": 7.10, "ours_wer": 22.98},
        {"category": "SM0", "ours_fad": 1.70, "expert_fad": 9.55, "ours_wer": 21.96},
        {
            "category": "SMA",
            "ours_fad": cfg.mecat_sma_fad,
            "expert_fad": cfg.expert_pipeline_sma_fad,
            "ours_wer": cfg.mecat_sma_wer_pct,
            "expert_wer": cfg.expert_pipeline_sma_wer_pct,
        },
    ]


def table_4_structured_ablation(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    return {
        "structured_wer_librispeech": cfg.librispeech_wer_pct,
        "unstructured_wer_librispeech": cfg.unstructured_wer_pct,
        "structured_utmos": cfg.librispeech_utmos,
        "unstructured_utmos": 2.70,
        "0ma_structured_fad": 3.25,
        "0ma_unstructured_fad": 5.04,
    }


def unified_vs_acoustic_gains() -> list[dict[str, Any]]:
    """Figure 3 selected relative gains (%)."""
    return [
        {"train_eval": "ACAVCaps | MECAT SMA", "avg_gain_pct": 21.1},
        {"train_eval": "ACAVCaps | AudioCaps", "avg_gain_pct": 33.3},
        {"train_eval": "ACAVCaps | MusicCaps", "avg_gain_pct": 27.0},
        {"train_eval": "ACAVCaps | LibriTTS WER", "avg_gain_pct": 67.3},
    ]


def mecat_category_legend() -> dict[str, str]:
    return {
        "S00": "speech only",
        "0M0": "music only",
        "00A": "sound effects only",
        "0MA": "music + audio (SFX)",
        "S0A": "speech + audio",
        "SM0": "speech + music",
        "SMA": "speech + music + audio",
    }
