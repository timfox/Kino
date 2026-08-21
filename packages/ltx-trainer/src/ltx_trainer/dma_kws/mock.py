"""DMA-KWS dual-stage scoring smoke (arXiv:2605.22120)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dma_kws.config import DmaKwsConfig
from ltx_trainer.dma_kws.ctc_streaming import (
    blank_insert,
    ctc_streaming_score,
    find_candidate_segments,
)
from ltx_trainer.dma_kws.enrollment import mam_fuse_cross_attention, text_enroll
from ltx_trainer.dma_kws.matcher import phoneme_match_score


def evaluation_smoke(cfg: DmaKwsConfig | None = None) -> dict[str, Any]:
    c = cfg or DmaKwsConfig()
    target = [1, 2]
    widened = blank_insert(target)
    post = np.full((10, 4), 0.2, dtype=np.float64)
    post[:, 0] = 0.1
    for t in range(3, 7):
        post[t, target[0]] = 0.7
        post[t, target[1]] = 0.65
    scores = ctc_streaming_score(post, target=target)
    segs = find_candidate_segments(scores, threshold=0.4)
    enroll = text_enroll([5, 7, 9], dim=16)
    fused = mam_fuse_cross_attention(enroll, np.zeros(16, dtype=np.float64))
    match = phoneme_match_score(enroll[:8], enroll[:8])
    combined = 0.5 * float(scores.max()) + 0.5 * float(match)
    return {
        "paper": c.paper_arxiv,
        "paper_headline_auc_lph": c.headline_auc_lph,
        "widened_len": len(widened),
        "n_candidate_segments": len(segs),
        "streaming_score_max": round(float(scores.max()), 4),
        "phoneme_match": round(float(match), 4),
        "combined_score": round(combined, 4),
        "enroll_dim": int(fused.shape[0]),
    }
