"""Paper Table 2 / Table 3 metric excerpts (arXiv:2602.12304)."""

from __future__ import annotations

from typing import Any


def table2_quantitative() -> list[dict[str, Any]]:
    """Main comparison (paper Tab. 2)."""
    return [
        {
            "setting": "typical_video",
            "model": "ID-Animator",
            "facesim_arc": 0.33,
            "facesim_cur": 0.35,
            "fid": 134.09,
            "fvd": 637.93,
            "clip_text": 25.40,
            "background_sound": False,
        },
        {
            "setting": "typical_video",
            "model": "ConsisID",
            "facesim_arc": 0.49,
            "facesim_cur": 0.50,
            "fid": 174.13,
            "fvd": 471.46,
            "clip_text": 27.05,
            "background_sound": False,
        },
        {
            "setting": "typical_video",
            "model": "Phantom",
            "facesim_arc": 0.56,
            "facesim_cur": 0.57,
            "fid": 169.69,
            "fvd": 478.36,
            "clip_text": 26.72,
            "background_sound": False,
        },
        {
            "setting": "typical_video",
            "model": "VACE",
            "facesim_arc": 0.22,
            "facesim_cur": 0.23,
            "fid": 183.85,
            "fvd": 575.09,
            "clip_text": 28.03,
            "background_sound": False,
        },
        {
            "setting": "tts",
            "model": "CosyVoice",
            "speaker_sim": 0.51,
            "fad": 5.20,
            "wer_pct": 4.77,
            "background_sound": False,
        },
        {
            "setting": "audio_driven",
            "model": "HunyuanCustom",
            "facesim_arc": 0.53,
            "facesim_cur": 0.55,
            "fid": 130.29,
            "fvd": 517.12,
            "clip_text": 23.98,
            "background_sound": False,
        },
        {
            "setting": "audio_driven",
            "model": "Humo",
            "facesim_arc": 0.50,
            "facesim_cur": 0.52,
            "fid": 181.01,
            "fvd": 475.94,
            "clip_text": 27.23,
            "background_sound": False,
        },
        {
            "setting": "sync_av",
            "model": "OmniCustom (baseline)",
            "facesim_arc": 0.39,
            "facesim_cur": 0.41,
            "fid": 137.18,
            "fvd": 559.96,
            "clip_text": 28.31,
            "speaker_sim": 0.29,
            "fad": 4.32,
            "wer_pct": 2.40,
            "background_sound": True,
        },
        {
            "setting": "sync_av",
            "model": "OmniCustom (+ embeddings)",
            "facesim_arc": 0.48,
            "facesim_cur": 0.49,
            "fid": 105.62,
            "fvd": 506.33,
            "clip_text": 27.68,
            "speaker_sim": 0.38,
            "fad": 3.67,
            "wer_pct": 2.78,
            "background_sound": True,
        },
        {
            "setting": "sync_av",
            "model": "OmniCustom (+ contrastive)",
            "facesim_arc": 0.60,
            "facesim_cur": 0.62,
            "fid": 95.57,
            "fvd": 440.49,
            "clip_text": 27.45,
            "speaker_sim": 0.47,
            "fad": 3.44,
            "wer_pct": 2.51,
            "background_sound": True,
        },
    ]


def table3_user_study() -> list[dict[str, Any]]:
    """Two-alternative forced-choice win rates vs OmniCustom (paper Tab. 3)."""
    return [
        {"competitor": "ID-Animator", "identity_consistency_pct": 95, "av_sync_pct": None, "video_quality_pct": 96},
        {"competitor": "ConsisID", "identity_consistency_pct": 91, "av_sync_pct": None, "video_quality_pct": 94},
        {"competitor": "Phantom", "identity_consistency_pct": 74, "av_sync_pct": None, "video_quality_pct": 86},
        {"competitor": "VACE", "identity_consistency_pct": 90, "av_sync_pct": None, "video_quality_pct": 91},
        {"competitor": "HunyuanCustom", "identity_consistency_pct": 81, "av_sync_pct": 88, "video_quality_pct": 85},
        {"competitor": "Humo", "identity_consistency_pct": 86, "av_sync_pct": 79, "video_quality_pct": 83},
    ]


def appendix_lse_comparison() -> list[dict[str, Any]]:
    """Lip-sync metrics (Appendix A-Tab. 1)."""
    return [
        {"model": "HunyuanCustom", "lse_c": 4.87, "lse_d": 10.47},
        {"model": "Humo", "lse_c": 5.42, "lse_d": 9.31},
        {"model": "OmniCustom", "lse_c": 5.76, "lse_d": 8.53},
    ]
