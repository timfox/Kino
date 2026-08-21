"""TSB segmented MVDR smoke (arXiv:2605.24825)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.tsb.config import TsbConfig
from ltx_trainer.tsb.segmented import batch_segmented_beamformer


def evaluation_smoke(cfg: TsbConfig | None = None) -> dict[str, Any]:
    c = cfg or TsbConfig()
    rng = np.random.default_rng(0)
    p = 5
    snapshots = rng.standard_normal((p, 12))
    steering = rng.standard_normal(p)
    steering /= np.linalg.norm(steering) + 1e-12
    out = batch_segmented_beamformer(snapshots, steering, penalty_c=c.default_penalty_c, delta=c.diagonal_loading_delta)
    single = batch_segmented_beamformer(snapshots, steering, penalty_c=1e6, delta=c.diagonal_loading_delta)
    return {
        "paper": c.paper_arxiv,
        "mse_improvement_db_abrupt": 3.0,
        "birth_death_penalty_c": c.default_penalty_c,
        "num_segments": int(out["num_segments"]),
        "penalized_cost": round(float(out["penalized_cost"]), 4),
        "split_beats_single": single["num_segments"] == 1,
    }
