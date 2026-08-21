"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.edm.augment import random_azimuth_augment
from ltx_trainer.edm.benchmarks import (
    IMPROVEMENT_MP_AUC5,
    IMPROVEMENT_S2D_AUC5,
    TABLE1_MATTERPORT3D,
    ablation_3d_full_best,
    edm_beats_dkm_matterport,
    edm_beats_all_s2d,
)
from ltx_trainer.edm.config import EdmConfig
from ltx_trainer.edm.edm_net import EdmStub
from ltx_trainer.edm.metrics import angular_error_deg, auc_at_threshold
from ltx_trainer.edm.spherical import erp_grid_to_cartesian


def evaluation_demo_run(cfg: EdmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EdmConfig()
    model = EdmStub(cfg)
    a = torch.rand(1, 3, 80, 160)
    b = torch.rand(1, 3, 80, 160)
    h, w = 20, 40
    tgt = erp_grid_to_cartesian(h, w).unsqueeze(0)
    cert = torch.ones(1, h, w)
    with torch.no_grad():
        out = model(a, b, tgt_s=tgt, certainty_gt=cert)
    err = angular_error_deg(out["match_s"], tgt)
    return {
        "match_shape": list(out["match_s"].shape),
        "loss": float(out["loss"].item()),
        "auc5_proxy": auc_at_threshold(err, 5.0),
        "paper_auc5_mp": TABLE1_MATTERPORT3D["EDM"]["auc_5"],
        "delta_mp_vs_dkm": IMPROVEMENT_MP_AUC5,
        "delta_s2d_vs_dkm": IMPROVEMENT_S2D_AUC5,
    }


def train_step(cfg: EdmConfig | None = None) -> dict[str, float]:
    cfg = cfg or EdmConfig()
    model = EdmStub(cfg)
    a = torch.rand(2, 3, 64, 128)
    b = torch.rand(2, 3, 64, 128)
    if cfg.use_azimuth_aug:
        a, _ = random_azimuth_augment(a)
    h, w = 16, 32
    tgt = erp_grid_to_cartesian(h, w).unsqueeze(0).expand(2, -1, -1, -1)
    cert = torch.ones(2, h, w)
    out = model(a, b, tgt_s=tgt, certainty_gt=cert)
    out["loss"].backward()
    return {"loss": float(out["loss"].detach())}


def ablation_table_check() -> dict[str, bool]:
    return {
        "beats_dkm_matterport": edm_beats_dkm_matterport(),
        "beats_all_stanford": edm_beats_all_s2d(),
        "ablation_3d_linear_best": ablation_3d_full_best(),
        "improvement_mp_is_26_72": abs(IMPROVEMENT_MP_AUC5 - 26.72) < 0.01,
        "improvement_s2d_is_42_62": abs(IMPROVEMENT_S2D_AUC5 - 42.62) < 0.01,
    }
