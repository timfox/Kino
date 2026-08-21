"""LTX dataset prep / validation bridge for multi-object slot consistency."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ltx_trainer.ocl_icc.config import OCLICCConfig
from ltx_trainer.ocl_icc.losses import total_objective
from ltx_trainer.ocl_icc.streams import VideoOCLStreams


def ltx_integration_notes(cfg: OCLICCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OCLICCConfig()
    return {
        "use_case": "Audit multi-object temporal consistency on pooled frame features before LTX training",
        "hook_point": "Optional DINO patch features per clip → VideoOCLStreams ICC loss as QA scalar",
        "does_not": "Does not replace SAM/mask prep or LTX latent encoding",
        "collapse_check": "Compare slot_diversity under ICC vs ECC modes on the same clip",
        "datasets_aligned": list(cfg.datasets),
    }


class OCLICCLTXBridge(nn.Module):
    """Run bidirectional OCL on (T, B, N, D) features; expose slots + ICC loss."""

    def __init__(self, cfg: OCLICCConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OCLICCConfig()
        self.cfg = cfg
        self.streams = VideoOCLStreams(cfg)

    def forward(self, frames: Tensor) -> dict[str, Tensor | float]:
        """``frames`` (T, B, N, D) → last forward slots and ICC total loss."""
        fw, bw = self.streams.run_bidirectional(frames)
        targets = [frames[t].mean(dim=1) for t in range(frames.shape[0])]
        losses = total_objective(fw, bw, targets, mode="icc")
        return {
            "slots": fw.slots[-1],
            "icc_loss": losses["total"],
            "l_recon_fw": losses["l_recon_fw"],
            "l_icc": losses["l_icc"],
        }
