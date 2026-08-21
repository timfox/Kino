"""Paper table anchors (VOICEGIRAFFE arXiv:2605.27976)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig


def table_1_benchmark_comparison() -> list[dict[str, Any]]:
    return [
        {"benchmark": "MMAU", "duration": "~10 sec", "speech": True, "sound": True, "music": True, "bilingual": False, "multi_hop": False},
        {"benchmark": "MMAR", "duration": "~20 sec", "speech": True, "sound": True, "music": True, "bilingual": False, "multi_hop": True},
        {"benchmark": "AudioMarathon", "duration": "~4 min", "speech": True, "sound": True, "music": True, "bilingual": False, "multi_hop": True},
        {"benchmark": "LongSpeech", "duration": "~10 min", "speech": True, "sound": False, "music": False, "bilingual": True, "multi_hop": False},
        {"benchmark": "BLAB", "duration": "~51 min", "speech": True, "sound": False, "music": False, "bilingual": False, "multi_hop": False},
        {
            "benchmark": "VOICEGIRAFFE",
            "duration": "~1 hr",
            "speech": True,
            "sound": True,
            "music": True,
            "bilingual": True,
            "multi_hop": True,
        },
    ]


def table_2_leaderboard(cfg: VoiceGiraffeConfig | None = None) -> list[dict[str, Any]]:
    """Selected Table 2 rows (E2E + human reference)."""
    cfg = cfg or VoiceGiraffeConfig()
    return [
        {
            "model": cfg.best_e2e_model,
            "e2e": True,
            "lrm": False,
            "temp_loc": 92.0,
            "sem_con": 94.0,
            "aco_evt": 64.0,
            "paralin": 72.0,
            "causal": 77.2,
            "evt_trk": 59.6,
            "overall": cfg.best_e2e_overall_pct,
        },
        {
            "model": "Gemini-3.1-Pro",
            "e2e": False,
            "lrm": False,
            "temp_loc": 79.6,
            "sem_con": 97.5,
            "aco_evt": 57.0,
            "paralin": 66.3,
            "causal": 85.2,
            "evt_trk": 69.6,
            "overall": 75.13,
        },
        {
            "model": cfg.best_opensource_cascade_model,
            "e2e": False,
            "lrm": False,
            "temp_loc": 30.8,
            "sem_con": 69.0,
            "aco_evt": 48.5,
            "paralin": 61.4,
            "causal": 45.6,
            "evt_trk": 47.2,
            "overall": cfg.best_opensource_cascade_pct,
        },
        {
            "model": "Human Reference",
            "e2e": False,
            "lrm": False,
            "temp_loc": 63.89,
            "sem_con": 90.74,
            "aco_evt": 79.63,
            "paralin": 79.17,
            "causal": cfg.human_causal_pct,
            "evt_trk": cfg.human_event_tracking_pct,
            "overall": cfg.human_overall_pct,
        },
    ]


def table_3_lrm_ablation() -> list[dict[str, Any]]:
    """Selected LRM ablation rows (Table 3)."""
    return [
        {"lalm": "Qwen2.5-Omni (7B)", "no_lrm": 41.1, "gpt52_lrm": 60.3, "gemini31_lrm": 71.9},
        {"lalm": "MiniCPM-o-4.5 (9B)", "no_lrm": 3.7, "gpt52_lrm": 62.9, "gemini31_lrm": 76.5},
        {"lalm": "Gemini-3.1-Pro", "no_lrm": 75.1, "gpt52_lrm": 59.6, "gemini31_lrm": 75.1},
        {"lalm": "Qwen3.5-Omni-Plus", "no_lrm": 66.2, "gpt52_lrm": 59.5, "gemini31_lrm": 78.3},
    ]


def table_4_language_bias() -> list[dict[str, Any]]:
    """Selected language-bias rows (Table 4, overall Δ)."""
    return [
        {"model": "Qwen3.5-Omni-Plus", "origin": "Chinese", "delta_en_minus_zh": -5.2},
        {"model": "Qwen3-Omni-Instruct", "origin": "Chinese", "delta_en_minus_zh": -4.7},
        {"model": "Gemini-3.1-Pro", "origin": "American", "delta_en_minus_zh": -2.5},
        {"model": "Phi-4-Multimodal", "origin": "American", "delta_en_minus_zh": 4.7},
    ]


def memory_asymmetry(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    """Finding 4: CA vs ET gap (models vs human)."""
    cfg = cfg or VoiceGiraffeConfig()
    human_et_minus_ca = cfg.human_event_tracking_pct - cfg.human_causal_pct
    gemini_ca = 85.2
    gemini_et = 69.6
    return {
        "human_event_tracking_minus_causal_pct": round(human_et_minus_ca, 2),
        "human_pattern": "ET > CA (stronger episodic memory)",
        "gemini31_causal_minus_et_pct": round(gemini_ca - gemini_et, 2),
        "model_pattern": "CA > ET (memory bottleneck region)",
    }
