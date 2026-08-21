"""Training and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lightharmony3d.config import LightHarmony3DConfig
from ltx_trainer.lightharmony3d.metrics import TABLE1_LH3D_KU, TABLE3_LH3D_BLENDER, TABLE4_ABLATION
from ltx_trainer.lightharmony3d.model import LightHarmony3D


def train_step(
    model: LightHarmony3D,
    *,
    background: Tensor,
    object_rgb: Tensor,
    mask: Tensor,
    target: Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(background, object_rgb, mask)
    loss = F.l1_loss(out.composite, target) + 0.1 * F.mse_loss(out.hdr_env, out.ev0)
    return loss, {"loss_l1": float(loss.detach())}


def ablation_configs() -> list[LightHarmony3DConfig]:
    return [
        LightHarmony3DConfig(use_hdr_fusion=False),
        LightHarmony3DConfig(use_shadow_ratio=False),
        LightHarmony3DConfig(use_ray_decoupled=False),
        LightHarmony3DConfig(use_gen_env=True, use_hdr_fusion=True, use_shadow_ratio=True, use_ray_decoupled=True),
    ]


def paper_report() -> dict[str, object]:
    return {
        "lh3d_ku": TABLE1_LH3D_KU,
        "lh3d_blender": TABLE3_LH3D_BLENDER,
        "ablation": TABLE4_ABLATION,
    }


def save_checkpoint(model: LightHarmony3D, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path | str, *, device: str = "cpu") -> LightHarmony3D:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = LightHarmony3DConfig(**ckpt.get("config", {}))
    model = LightHarmony3D(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
