"""Toy smoke hooks for validate_paper_stubs."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.foley_omni.conditioning import toy_project_conditions
from ltx_trainer.foley_omni.training import training_step_smoke


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    cond = toy_project_conditions(seed=seed)
    train = training_step_smoke(seed=seed + 1)
    return {
        "c_uni_dim": int(cond["c_uni"].shape[-1]),
        "x_tilde_norm": round(float(np.linalg.norm(cond["x_tilde"])), 4),
        "flow_loss": round(train["loss"], 6),
    }
