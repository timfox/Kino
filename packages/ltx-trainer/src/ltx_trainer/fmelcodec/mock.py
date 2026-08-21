"""FMelCodec mel coding loss smoke (arXiv:2605.25669)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fmelcodec.coding import coding_stage_loss
from ltx_trainer.fmelcodec.config import FMelCodecConfig


def evaluation_smoke(cfg: FMelCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or FMelCodecConfig()
    rng = np.random.default_rng(0)
    natural = rng.standard_normal((4, 32))
    coarse = natural + 0.1 * rng.standard_normal((4, 32))
    z = rng.standard_normal((4, 8))
    z_q = z + 0.05 * rng.standard_normal((4, 8))
    loss = coding_stage_loss(natural, coarse, z, z_q)
    return {
        "paper": c.paper_arxiv,
        "coding_loss": round(loss, 4),
    }
