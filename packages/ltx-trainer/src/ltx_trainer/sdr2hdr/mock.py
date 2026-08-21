"""SDR2HDR evaluation smoke (computed bracket + merge path)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.sdr2hdr.pipeline import Sdr2HdrConfig, sdr_video_to_hdr, tone_map_hdr_for_preview
from ltx_trainer.sdr2hdr.vmm import VideoMergingModel, vmm_log_loss


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    sdr = torch.rand(3, 4, 32, 32)
    cfg = Sdr2HdrConfig(merge="debevec", device="cpu")
    hdr, brackets, evs = sdr_video_to_hdr(sdr, cfg)
    preview = tone_map_hdr_for_preview(hdr)
    model = VideoMergingModel(num_exposures=len(evs))
    pred = model(brackets, evs)
    loss = float(vmm_log_loss(pred, hdr).item())
    return {
        "package": "sdr2hdr",
        "paper": "arXiv:2605.14703",
        "num_brackets": len(evs),
        "hdr_finite": bool(torch.isfinite(hdr).all()),
        "preview_mean": round(float(preview.mean()), 4),
        "vmm_log_loss": round(loss, 5),
        "merge_mode": cfg.merge,
    }
