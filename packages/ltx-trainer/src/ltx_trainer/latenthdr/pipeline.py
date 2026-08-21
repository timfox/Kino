"""Inference, checkpoints, HDR recovery, and paper evaluation stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.latenthdr.model import LatentHdr, LatentHdrConfig
from ltx_trainer.latenthdr.paper import evaluation_demo, framework_card


def recover_hdr_l2h(model: LatentHdr, ldr: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model(ldr)


def load_latenthdr_checkpoint(path: Path | str, *, device: str = "cpu") -> LatentHdr:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = LatentHdrConfig(**ckpt.get("config", {}))
    model = LatentHdr(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
