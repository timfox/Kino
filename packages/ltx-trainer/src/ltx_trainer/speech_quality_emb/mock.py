"""Speech quality embedding loss + mix-up smoke (arXiv:2605.21332)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.speech_quality_emb.config import SpeechQualityEmbConfig
from ltx_trainer.speech_quality_emb.losses import (
    frame_pseudo_l1,
    supervised_contrastive_stub,
    total_training_loss,
    utterance_mae_loss,
)
from ltx_trainer.speech_quality_emb.mixup import partial_mixup_waveform_mask, pseudo_frame_scores


def evaluation_smoke(cfg: SpeechQualityEmbConfig | None = None) -> dict[str, Any]:
    c = cfg or SpeechQualityEmbConfig()
    rng = np.random.default_rng(0)
    q_ref = np.linspace(3.0, 2.0, 20, dtype=np.float64)
    q_deg = np.linspace(2.5, 1.5, 20, dtype=np.float64)
    mask = partial_mixup_waveform_mask(20, rng=rng)
    q_pseudo = pseudo_frame_scores(q_ref, q_deg, mask)
    l_frame = frame_pseudo_l1(q_ref, q_pseudo)
    z = rng.standard_normal((8, 4))
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    l_scl = supervised_contrastive_stub(z, labels, temperature=c.tau_temperature)
    l_utt = utterance_mae_loss(q_ref, 2.5)
    l_total = total_training_loss(l_utt, l_frame, l_scl, tau=c.tau_temperature)
    return {
        "paper": c.paper_arxiv,
        "paper_con1_nisqa_zscl_embed_iauc": c.con1_nisqa_zscl_embed_iauc,
        "paper_baseline_nisqa_mos_iauc": c.baseline_nisqa_mos_iauc,
        "L_total": round(l_total, 4),
        "pseudo_frames": int(q_pseudo.size),
    }
