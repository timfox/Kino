"""F5-TTS-DPS dual-scoring filter smoke (arXiv:2605.23859)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.f5_tts_dps.config import F5TtsDpsConfig
from ltx_trainer.f5_tts_dps.dps import dual_score_filter


def evaluation_smoke(cfg: F5TtsDpsConfig | None = None) -> dict[str, Any]:
    c = cfg or F5TtsDpsConfig()
    candidates = [
        {
            "transcript": "I really love this amazing expressive song today!",
            "target_text": "hello world",
            "reference_texts": ["hello there world", "hi everyone"],
        },
        {"transcript": "ok", "target_text": "hello", "reference_texts": []},
    ]
    kept = dual_score_filter(candidates, audio_min=c.audio_score_min)
    return {
        "paper": c.paper_arxiv,
        "dev_utmos": c.dev_utmos,
        "dev_wer": c.dev_wer,
        "dps_candidates_kept": len(kept),
        "dps_candidates_in": len(candidates),
    }
