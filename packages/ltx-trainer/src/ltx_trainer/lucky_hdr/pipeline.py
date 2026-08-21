"""Merge CLI helpers and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ltx_trainer.lucky_hdr.capture_ae import bracket_evs
from ltx_trainer.lucky_hdr.model import LuckyHdr, LuckyHdrConfig


def _load_rgb(path: Path, *, size: int | None = None) -> Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    return arr.permute(2, 0, 1)


def load_lucky_hdr_checkpoint(path: str | Path, *, device: str = "cpu") -> LuckyHdr:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config") or {}
    cfg = LuckyHdrConfig(**{k: v for k, v in cfg_dict.items() if k in LuckyHdrConfig.__dataclass_fields__})
    model = LuckyHdr(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def merge_bracket_stack(model: LuckyHdr, stack: Tensor, evs: list[float] | Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        out, _, _ = model(stack, evs)
    return out


def merge_bracket_paths(
    paths: list[str | Path],
    *,
    device: str = "cpu",
    size: int | None = None,
    checkpoint: str | Path | None = None,
) -> Tensor:
    frames = [_load_rgb(Path(p), size=size) for p in paths]
    stack = torch.stack(frames, dim=0).to(device)
    evs = bracket_evs(n=len(frames), span=2.0)
    if checkpoint and Path(checkpoint).exists():
        model = load_lucky_hdr_checkpoint(checkpoint, device=device)
    else:
        model = LuckyHdr(LuckyHdrConfig()).to(device)
        model.eval()
    return merge_bracket_stack(model, stack, evs)


def pipeline_merge_smoke(*, seed: int = 0, size: int = 32) -> dict[str, Any]:
    from ltx_trainer.lucky_hdr.synthetic import synthesize_bracket_burst

    torch.manual_seed(seed)
    hdr = torch.rand(3, size, size)
    stack, gt, evs = synthesize_bracket_burst(hdr, num_frames=3, shake_px=0.5, seed=seed)
    model = LuckyHdr(LuckyHdrConfig())
    out = merge_bracket_stack(model, stack, evs)
    mse = float((out - gt).pow(2).mean())
    return {"merge_ran": True, "mse_vs_gt": mse, "out_shape": list(out.shape)}
