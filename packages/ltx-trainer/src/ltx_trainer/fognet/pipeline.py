"""Training, inference, checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.fognet.losses import FogNetLoss
from ltx_trainer.fognet.model import FogNet, FogNetConfig


def train_step(
    model: FogNet,
    loss_fn: FogNetLoss,
    *,
    foggy: Tensor,
    clean: Tensor,
    labels: Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if foggy.dim() == 4:
        foggy = foggy.unsqueeze(0)
        clean = clean.unsqueeze(0)
        labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
    fog_emb, clean_emb, text_emb, temp_loss = model.forward_train(foggy, clean, labels)
    return loss_fn(fog_emb=fog_emb, clean_emb=clean_emb, text_emb=text_emb, temp_loss=temp_loss)


@torch.no_grad()
def predict_action(model: FogNet, foggy: Tensor) -> tuple[int, Tensor]:
    model.eval()
    logits = model.forward_infer(foggy)
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    return int(logits.argmax().item()), logits


def load_fognet_checkpoint(path: Path | str, *, device: str = "cpu") -> FogNet:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = FogNetConfig(**ckpt.get("config", {}))
    model = FogNet(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()


def save_checkpoint(model: FogNet, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)
