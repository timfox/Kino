"""Scoring utilities, checkpoint I/O, and paper evaluation stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.fdim.model import FDIM, FDIMConfig
from ltx_trainer.fdim.paper import evaluation_demo, framework_card


def score_video_pair(
    model: FDIM,
    ref: Tensor,
    dist: Tensor,
    *,
    sample_fps: float = 1.0,
    native_fps: float = 25.0,
) -> float:
    model.eval()
    if ref.dim() == 5:
        ref = ref.squeeze(0)
        dist = dist.squeeze(0)
    step = max(1, int(round(native_fps / max(sample_fps, 1e-3))))
    ref_s = ref[::step]
    dist_s = dist[::step]
    with torch.no_grad():
        q = model(ref_s, dist_s)
    return float(q.reshape(-1)[0].cpu())


def load_fdim_checkpoint(path: Path | str, *, device: str = "cpu") -> FDIM:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = FDIMConfig(
        use_pu21=bool(cfg_dict.get("use_pu21", False)),
        l_peak=float(cfg_dict.get("l_peak", 1000.0)),
    )
    model = FDIM(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
