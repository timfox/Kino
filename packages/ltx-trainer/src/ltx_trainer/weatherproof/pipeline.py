"""Training step, inference, checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.weatherproof.ema import update_ema_teacher
from ltx_trainer.weatherproof.losses import SemiSupervisedLoss
from ltx_trainer.weatherproof.model import UniMatchV2Config, UniMatchV2Seg


def train_step(
    model: UniMatchV2Seg,
    loss_fn: SemiSupervisedLoss,
    *,
    clean: Tensor,
    mask: Tensor,
    degraded: Tensor,
    ema_decay: float = 0.999,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One semi-supervised step: clean supervised + degraded consistency."""
    clean_logits = model.student(clean.unsqueeze(0) if clean.dim() == 3 else clean)
    pseudo, conf, _ = model.pseudo_from_teacher(degraded)
    s1, s2 = model.student.forward_strong_pair(degraded)
    if pseudo.dim() == 2:
        pseudo = pseudo.unsqueeze(0)
        conf = conf.unsqueeze(0)
    loss, stats = loss_fn(
        clean_logits,
        mask.unsqueeze(0) if mask.dim() == 2 else mask,
        strong_logits1=s1,
        strong_logits2=s2,
        pseudo1=pseudo,
        pseudo2=pseudo,
        conf1=conf,
        conf2=conf,
    )
    update_ema_teacher(model.teacher, model.student, decay=ema_decay)
    return loss, stats


@torch.no_grad()
def predict_segmentation(model: UniMatchV2Seg, img: Tensor) -> Tensor:
    model.eval()
    logits = model.student(img.unsqueeze(0) if img.dim() == 3 else img)
    return logits.argmax(dim=1).squeeze(0)


def load_weatherproof_checkpoint(path: Path | str, *, device: str = "cpu") -> UniMatchV2Seg:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = UniMatchV2Config(**ckpt.get("config", {}))
    model = UniMatchV2Seg(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()


def save_checkpoint(model: UniMatchV2Seg, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)
