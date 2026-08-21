"""FC-TTS dual-reference synthesis smoke (arXiv:2605.24618)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fctts.config import FcttsConfig
from ltx_trainer.fctts.facodec import fctts_conditioning, split_facodec_streams
from ltx_trainer.fctts.stages import dual_reference_synthesis


def evaluation_smoke(cfg: FcttsConfig | None = None) -> dict[str, Any]:
    c = cfg or FcttsConfig()
    rng = np.random.default_rng(0)
    factors = split_facodec_streams(rng.standard_normal(80))
    cond = fctts_conditioning(factors)
    phoneme = rng.standard_normal((6, 8))
    timbre_ref = rng.standard_normal(32)
    style_ref = factors.prosody_tokens
    out = dual_reference_synthesis(phoneme, timbre_ref, style_ref, rng)
    return {
        "paper": c.paper_arxiv,
        "separate_references": True,
        "uses_only_cp_and_zspk": (not cond["uses_content"]) and (not cond["uses_detail"]),
        "output_len": int(out["x_hat"].size),
    }
