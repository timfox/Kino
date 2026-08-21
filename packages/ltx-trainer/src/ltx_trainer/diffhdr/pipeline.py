"""Training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.diffhdr.config import DiffHDRConfig
from ltx_trainer.diffhdr.metrics import TABLE1_SI_HDR, TABLE4_LOG_GAMMA, TABLE5_ABLATION
from ltx_trainer.diffhdr.model import DiffHDR


def train_step(model: DiffHDR, *, ldr: Tensor, hdr: Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(ldr, hdr_target=hdr)
    loss = model._last_flow_loss
    if loss is None:
        loss = torch.zeros((), device=ldr.device)
    stats = {"loss_flow": float(loss.detach())}
    return loss, stats


def ablation_configs() -> list[DiffHDRConfig]:
    return [
        DiffHDRConfig(use_data_aug=False),
        DiffHDRConfig(use_mask=False),
        DiffHDRConfig(use_cfa=False),
        DiffHDRConfig(use_mask=True, use_cfa=True, use_data_aug=True),
    ]


def paper_report() -> dict[str, object]:
    return {
        "si_hdr": TABLE1_SI_HDR,
        "log_gamma_ablation": TABLE4_LOG_GAMMA,
        "mask_aug_ablation": TABLE5_ABLATION,
    }


def save_checkpoint(model: DiffHDR, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path | str, *, device: str = "cpu") -> DiffHDR:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = DiffHDRConfig(**ckpt.get("config", {}))
    model = DiffHDR(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
