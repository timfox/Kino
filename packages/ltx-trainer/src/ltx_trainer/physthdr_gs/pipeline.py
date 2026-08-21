"""Training, inference, and evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.physthdr_gs.config import PhysHDRConfig
from ltx_trainer.physthdr_gs.gradient_scaling import scaling_factor
from ltx_trainer.physthdr_gs.losses import PhysHDRLoss, PhysHDRLossConfig
from ltx_trainer.physthdr_gs.metrics import TABLE2_EXP3, TABLE3_EFFICIENCY, TABLE4_ABLATION, psnr
from ltx_trainer.physthdr_gs.cameras import Camera, camera_to_device
from ltx_trainer.physthdr_gs.model import PhysHDRGS


def train_step(
    model: PhysHDRGS,
    loss_fn: PhysHDRLoss,
    *,
    target_ldr: Tensor,
    exposure: float,
    camera: Camera | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    model.set_iteration(getattr(model, "_train_iter", 0))
    t = torch.tensor(exposure, device=target_ldr.device, dtype=target_ldr.dtype)
    cam = camera_to_device(camera, target_ldr.device) if camera is not None else None
    out = model(
        t,
        lighting_level=t,
        camera=cam if model.cfg.use_perspective else None,
    )
    preds = {"ildr": out.ildr, "iig": out.iig, "igi": out.igi}
    loss, stats = loss_fn(
        preds=preds,
        target_ldr=target_ldr,
        ihdr_scaled=out.ihdr_scaled,
        ihdr_relit=out.ihdr_relit,
        ihdr=out.ihdr,
        exposure=exposure,
        use_cons=model.cfg.use_hdr_cons and model.cfg.use_gi_branch,
    )
    if model.cfg.use_igs and model.cfg.use_gi_branch:
        sa = scaling_factor(out.la, out.la_hat, s=model.cfg.scale_s)
        stats["mean_scale_factor"] = float(sa.mean().detach())
    return loss, stats


def evaluate_psnr(
    model: PhysHDRGS,
    target: Tensor,
    exposure: float = 1.0,
    *,
    camera: Camera | None = None,
) -> float:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(exposure, device=target.device, dtype=target.dtype)
        cam = camera_to_device(camera, target.device) if camera is not None else None
        out = model(
            t,
            lighting_level=t,
            camera=cam if model.cfg.use_perspective else None,
        )
        return psnr(out.ildr.clamp(0, 1), target.clamp(0, 1))


def ablation_configs() -> list[PhysHDRConfig]:
    base = PhysHDRConfig()
    return [
        PhysHDRConfig(use_gi_branch=False, use_hdr_cons=False, use_igs=False),
        PhysHDRConfig(use_gi_branch=True, use_hdr_cons=False, use_igs=False),
        PhysHDRConfig(use_gi_branch=True, use_hdr_cons=True, use_igs=False),
        PhysHDRConfig(use_gi_branch=True, use_hdr_cons=True, use_igs=True),
    ]


def paper_report() -> dict[str, object]:
    return {
        "table2_exp3": TABLE2_EXP3,
        "table3_efficiency": TABLE3_EFFICIENCY,
        "table4_ablation": TABLE4_ABLATION,
        "hdr_psnr_gain_over_hdr_gs_db": 2.04,
    }


def save_checkpoint(model: PhysHDRGS, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path | str, *, device: str = "cpu") -> PhysHDRGS:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg = PhysHDRConfig(**ckpt.get("config", {}))
    model = PhysHDRGS(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()
