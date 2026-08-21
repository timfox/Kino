"""DEP audit SRDS + probe smoke (Zenodo depression benchmark audit)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dep_audit.config import DepAuditConfig
from ltx_trainer.dep_audit.srds import classify_chunk_stub, heavy_neutral_delta


def evaluation_smoke(cfg: DepAuditConfig | None = None) -> dict[str, Any]:
    c = cfg or DepAuditConfig()
    heavy = [classify_chunk_stub("I feel hopeless and cannot sleep")["topic_score"] for _ in range(5)]
    neutral = [classify_chunk_stub("the weather is nice today")["topic_score"] for _ in range(5)]
    text_delta = heavy_neutral_delta([float(h) for h in heavy], [float(n) for n in neutral])
    audio_heavy = [0.1, 0.12, 0.08, 0.11, 0.09]
    audio_neutral = [0.11, 0.10, 0.12, 0.09, 0.10]
    audio_delta = heavy_neutral_delta(audio_heavy, audio_neutral)
    return {
        "paper": c.paper_zenodo,
        "loso_tl_macro_f1": c.loso_tl_macro_f1,
        "text_mean_shift": text_delta["mean_shift"],
        "audio_mean_shift": audio_delta["mean_shift"],
    }
