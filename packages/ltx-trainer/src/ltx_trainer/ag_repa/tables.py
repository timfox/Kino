"""Paper tables (arXiv:2603.01006)."""

from __future__ import annotations

from typing import Any


def table1_scd_dissociation() -> list[dict[str, Any]]:
    """Store-Contribute Dissociation top-3 layers (Table 1)."""
    return [
        {
            "config": "A: S3 + AudioSet",
            "cos_sem_top3": ["L24(0.323)", "L18(0.293)", "L17(0.293)"],
            "cos_evt_top3": ["L20(0.180)", "L21(0.174)", "L19(0.168)"],
            "fog_speech_top3": ["L1(0.167)", "L9(0.0695)", "L5(0.0588)"],
            "fog_audio_top3": ["L1(0.140)", "L21(0.060)", "L9(0.0561)"],
        },
        {
            "config": "B: A + interleaved BEATs",
            "cos_sem_top3": ["L23(0.318)", "L22(0.314)", "L24(0.309)"],
            "cos_evt_top3": ["L14(0.218)", "L13(0.212)", "L15(0.207)"],
            "fog_speech_top3": ["L1(0.192)", "L2(0.109)", "L7(0.085)"],
            "fog_audio_top3": ["L1(0.164)", "L7(0.094)", "L2(0.083)"],
        },
    ]


def table2_alignment_strategies() -> list[dict[str, Any]]:
    """Config B alignment comparison (Table 2)."""
    return [
        {"method": "Base (no layer align.)", "speech_wer": 5.82, "speech_fad": 1.84, "audio_fad": 3.45, "speech_mos": "3.62±.08", "audio_mos": "3.45±.09"},
        {"method": "REPA @ Layer 4", "speech_wer": 5.15, "speech_fad": 1.65, "audio_fad": 3.12, "speech_mos": "3.75±.07", "audio_mos": "3.58±.08"},
        {"method": "REPA @ Layer 8", "speech_wer": 4.93, "speech_fad": 1.58, "audio_fad": 3.05, "speech_mos": "3.79±.06", "audio_mos": "3.64±.08"},
        {"method": "REPA @ Layer 12", "speech_wer": 5.38, "speech_fad": 1.72, "audio_fad": 3.28, "speech_mos": "3.68±.08", "audio_mos": "3.51±.09"},
        {"method": "REPA @ L4, 8, 12", "speech_wer": 4.21, "speech_fad": 1.45, "audio_fad": 2.88, "speech_mos": "3.92±.06", "audio_mos": "3.77±.07"},
        {"method": "REPA @ Deep (L20–L22)", "speech_wer": 5.60, "speech_fad": 1.79, "audio_fad": 3.39, "speech_mos": "3.64±.08", "audio_mos": "3.48±.09"},
        {"method": "REPA @ Shallow (L1–L3)", "speech_wer": 3.62, "speech_fad": 1.36, "audio_fad": 2.68, "speech_mos": "4.05±.06", "audio_mos": "3.87±.07"},
        {"method": "AG-REPA (Top-3)", "speech_wer": 3.45, "speech_fad": 1.29, "audio_fad": 2.56, "speech_mos": "4.12±.05", "audio_mos": "3.94±.07"},
    ]


def table3_selection_targets() -> list[dict[str, Any]]:
    """Knowing vs doing (Table 3)."""
    return [
        {"strategy": "Base (no layer align.)", "top3": "NONE", "steps": 500_000, "speech_fad": 1.84, "audio_fad": 3.45, "rel_gain_pct": 0.0, "convergence_steps": None},
        {"strategy": "Random Control", "top3": "L5, L14, L19", "steps": 500_000, "speech_fad": 1.75, "audio_fad": 3.32, "rel_gain_pct": 4.9, "convergence_steps": 850_000},
        {"strategy": "Highest LASP", "top3": "L22, L23, L24", "steps": 500_000, "speech_fad": 1.68, "audio_fad": 3.21, "rel_gain_pct": 8.7, "convergence_steps": 720_000},
        {"strategy": "Gradient Norm", "top3": "L1, L2, L4", "steps": 500_000, "speech_fad": 1.35, "audio_fad": 2.71, "rel_gain_pct": 26.6, "convergence_steps": 260_000},
        {"strategy": "Highest FoG-A (Ours)", "top3": "L1, L2, L7", "steps": 500_000, "speech_fad": 1.29, "audio_fad": 2.56, "rel_gain_pct": 29.9, "convergence_steps": 220_000},
    ]


def table4_generalization() -> list[dict[str, Any]]:
    return [
        {"model": "Voicebox", "wer": 2.05, "fad": 1.20, "mos": "4.15±.06", "variant": "baseline"},
        {"model": "Voicebox", "wer": 1.85, "fad": 0.95, "mos": "4.28±.05", "variant": "+ AG-REPA"},
        {"model": "CosyVoice", "wer": 1.95, "fad": 0.88, "mos": "4.25±.05", "variant": "baseline"},
        {"model": "CosyVoice", "wer": 1.78, "fad": 0.72, "mos": "4.39±.04", "variant": "+ AG-REPA"},
        {"model": "F5-TTS", "wer": 2.40, "fad": 1.45, "mos": "4.05±.07", "variant": "baseline"},
        {"model": "F5-TTS", "wer": 2.12, "fad": 1.15, "mos": "4.22±.06", "variant": "+ AG-REPA"},
    ]


def table5_cross_architecture() -> list[dict[str, Any]]:
    return [
        {"architecture": "Our DiT", "base_fad": 1.84, "shallow_repa_fad": 1.36, "ag_repa_fad": 1.29},
        {"architecture": "F5-TTS", "base_fad": 1.45, "shallow_repa_fad": 1.34, "ag_repa_fad": 1.15},
        {"architecture": "Voicebox", "base_fad": 1.20, "shallow_repa_fad": 1.12, "ag_repa_fad": 0.95},
        {"architecture": "CosyVoice", "base_fad": 0.88, "shallow_repa_fad": 0.85, "ag_repa_fad": 0.72},
    ]


def table6_efficiency() -> list[dict[str, Any]]:
    return [
        {"method": "LASP-selected REPA", "main_training": "T0", "diagnostic": "≈0", "end_to_end": "1.00 T0"},
        {"method": "AG-REPA (Ours)", "main_training": "≈0.30 T0", "diagnostic": "<0.005 T0", "end_to_end": "≈0.305 T0"},
    ]


def table7_fog_stability() -> list[dict[str, Any]]:
    return [
        {"probe": "Warm-up (5k)", "speech_top3": "L1, L2, L7", "audio_top3": "L1, L7, L2", "overlap": "3/3, 3/3"},
        {"probe": "Mid-training", "speech_top3": "L1, L2, L8", "audio_top3": "L1, L7, L9", "overlap": "2/3, 2/3"},
        {"probe": "Final checkpoint", "speech_top3": "L1, L2, L7", "audio_top3": "L1, L7, L3", "overlap": "3/3, 2/3"},
    ]


def table8_freeze_vs_refresh() -> list[dict[str, Any]]:
    return [
        {"schedule": "Freeze @ Epoch 1 (Ours)", "speech_fad": 1.29, "audio_fad": 2.56, "relative_time": 1.00},
        {"schedule": "Refresh set every epoch", "speech_fad": 1.28, "audio_fad": 2.55, "relative_time": 1.19},
    ]


def table9_static_vs_adaptive() -> list[dict[str, Any]]:
    return [
        {"strategy": "Static AG-REPA (Ours)", "complexity": "Low (fixed heads)", "speech_fad": 1.29, "audio_fad": 2.56},
        {"strategy": "Timestep-Adaptive", "complexity": "High (time-variant heads)", "speech_fad": 1.28, "audio_fad": 2.55},
    ]


def table10_k_sensitivity() -> list[dict[str, Any]]:
    return [
        {"setting": "K=2, FoG-A λ", "speech_fad": 1.31, "audio_fad": 2.60},
        {"setting": "K=3, equal λ", "speech_fad": 1.32, "audio_fad": 2.59},
        {"setting": "K=3, FoG-A λ (Ours)", "speech_fad": 1.29, "audio_fad": 2.56},
        {"setting": "K=4, FoG-A λ", "speech_fad": 1.30, "audio_fad": 2.58},
    ]


def table11_projection_heads() -> list[dict[str, Any]]:
    return [
        {"head": "2D Conv (iREPA-style)", "params_m": 0.29, "speech_fad": 1.32, "audio_fad": 2.61},
        {"head": "1D Conv (K=3)", "params_m": 0.24, "speech_fad": 1.31, "audio_fad": 2.58},
        {"head": "2-layer MLP (Ours)", "params_m": 0.18, "speech_fad": 1.29, "audio_fad": 2.56},
    ]


def table12_supervision_source() -> list[dict[str, Any]]:
    return [
        {"source": "Base (no layer align.)", "static_fad": "1.84 / 3.45", "ag_repa_fad": "—"},
        {"source": "Internal self-alignment", "static_fad": "1.53 / 2.97", "ag_repa_fad": "1.40 / 2.74"},
        {"source": "Whisper / BEATs teachers", "static_fad": "1.45 / 2.88", "ag_repa_fad": "1.29 / 2.56"},
    ]
