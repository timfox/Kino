"""Checkpoint I/O and one-step HDR recovery."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.expo_cm.model import ExpoCM, ExpoCMConfig


def recover_hdr_one_step(model: ExpoCM, ldr: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model.one_step(ldr)


def load_expocm_checkpoint(path: Path | str, *, device: str = "cpu") -> ExpoCM:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = ExpoCMConfig(**ckpt.get("config", {}))
    model = ExpoCM(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
