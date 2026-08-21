"""Inference and checkpoint I/O."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.eic_lie.events import events_to_sbt_voxel, synthetic_events_from_image
from ltx_trainer.eic_lie.model import EicLie, EicLieConfig


def load_eic_lie_checkpoint(path: str | Path, *, device: str = "cuda") -> EicLie:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict.get("eici_stages"), list):
        cfg_dict["eici_stages"] = tuple(cfg_dict["eici_stages"])
    cfg = EicLieConfig(**{k: v for k, v in cfg_dict.items() if k in EicLieConfig.__dataclass_fields__})
    model = EicLie(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def _load_rgb(path: Path, device: torch.device, size: tuple[int, int] | None = None) -> Tensor:
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, Image.Resampling.BILINEAR)
        return to_tensor(im).to(device)


def enhance_low_light(
    model: EicLie,
    low_image: Tensor,
    event_voxel: Tensor | None = None,
) -> Tensor:
    """Enhance ``[3,H,W]`` or ``[B,3,H,W]`` low-light frame."""
    single = low_image.dim() == 3
    if single:
        low_image = low_image.unsqueeze(0)
    device = low_image.device
    h, w = low_image.shape[-2:]
    if event_voxel is None:
        ev = synthetic_events_from_image(low_image.squeeze(0), num_events=h * w // 2)
        event_voxel = events_to_sbt_voxel(ev, height=h, width=w, num_bins=model.cfg.event_bins)
    if event_voxel.dim() == 3:
        event_voxel = event_voxel.unsqueeze(0)
    event_voxel = event_voxel.to(device)
    if event_voxel.abs().max() > 0:
        event_voxel = event_voxel / event_voxel.abs().max()
    model.eval()
    with torch.no_grad():
        out = model(low_image, event_voxel)
    return out.squeeze(0) if single else out


def save_checkpoint(model: EicLie, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps({"config": asdict(model.cfg)}, indent=2), encoding="utf-8")
