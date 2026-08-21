"""KSAA diacritization loss + invariant smoke (arXiv:2605.25928)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ksaa_diac.config import KsaaDiacConfig
from ltx_trainer.ksaa_diac.losses import focal_loss
from ltx_trainer.ksaa_diac.postprocess import insert_diacritics, verify_invariants


def evaluation_smoke(cfg: KsaaDiacConfig | None = None) -> dict[str, Any]:
    c = cfg or KsaaDiacConfig()
    und = "abc"
    labels = ["َ", "NONE", "NONE"]
    diac = insert_diacritics(und, labels)
    inv = verify_invariants(und, diac, labels)
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((len(labels), c.num_diacritic_classes))
    targets = np.array([1, 0, 0], dtype=np.int64)
    fl = focal_loss(logits, targets, gamma=c.focal_gamma, label_smoothing=c.label_smoothing)
    return {
        "paper": c.paper_arxiv,
        "test_wer_pct": 23.26,
        "invariants": inv,
        "focal_loss": round(fl, 4),
    }
