"""Training / ablation demos."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.campvg.benchmarks import TABLE2_ABLATION, TABLE3_EPIPOLAR_K
from ltx_trainer.campvg.campvg_net import CamPVGStub
from ltx_trainer.campvg.config import CamPVGConfig, EPIPOLAR_SAMPLES_K
from ltx_trainer.campvg.epipolar import sample_epipolar_line
from ltx_trainer.campvg.losses import diffusion_noise_loss, frame_reconstruction_loss
from ltx_trainer.campvg.plucker_pano import panoramic_plucker_map
from ltx_trainer.campvg.synthetic import synthetic_erp_video, synthetic_poses


def evaluation_demo_run(cfg: CamPVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CamPVGConfig(height=64, width=128, num_frames=4, epipolar_k=16)
    model = CamPVGStub(cfg)
    frames = synthetic_erp_video(cfg)
    rots, trans = synthetic_poses(cfg)
    with torch.no_grad():
        out = model(frames, rots, trans)
    return {
        "frames_shape": list(frames.shape),
        "recon_shape": list(out["frames"].shape),
        "plucker_shape": list(out["plucker"].shape),
    }


def ablation_components(cfg: CamPVGConfig | None = None) -> dict[str, float]:
    cfg = cfg or CamPVGConfig(height=32, width=64, num_frames=3, epipolar_k=8)
    frames = synthetic_erp_video(cfg)
    rots, trans = synthetic_poses(cfg)
    full = CamPVGStub(cfg)
    no_epi = CamPVGStub(replace(cfg, use_spherical_epipolar=False))
    no_plk = CamPVGStub(replace(cfg, use_pano_plucker=False))
    with torch.no_grad():
        f = full(frames, rots, trans)["frames"].mean()
        e = no_epi(frames, rots, trans)["frames"].mean()
        p = no_plk(frames, rots, trans)["frames"].mean()
    return {
        "full_mean": float(f),
        "no_epipolar_mean": float(e),
        "no_plucker_mean": float(p),
        "table2_full_psnr": TABLE2_ABLATION["CamPVG_full"]["psnr"],
        "table2_no_plucker_psnr": TABLE2_ABLATION["w/o_pano_plucker"]["psnr"],
    }


def epipolar_k_demo(k: int = 250) -> dict[str, Any]:
    samples = sample_epipolar_line(0.1, 1.0, 64, 128, min(k, 32), torch.device("cpu"))
    return {
        "k": k,
        "paper_best_k": EPIPOLAR_SAMPLES_K,
        "table3_psnr_at_k": TABLE3_EPIPOLAR_K[k]["psnr"],
        "sample_directions_shape": list(samples.shape),
    }


def plucker_demo(cfg: CamPVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CamPVGConfig(height=32, width=64)
    r = torch.eye(3)
    t = torch.tensor([0.0, 0.0, 1.0])
    p = panoramic_plucker_map(r, t, cfg.height, cfg.width)
    return {"plucker_shape": list(p.shape), "moment_norm_mean": float(p[..., :3].norm(dim=-1).mean())}


def train_step(cfg: CamPVGConfig | None = None) -> dict[str, float]:
    cfg = cfg or CamPVGConfig(height=32, width=64, num_frames=3, epipolar_k=8)
    model = CamPVGStub(cfg)
    frames = synthetic_erp_video(cfg)
    rots, trans = synthetic_poses(cfg)
    out = model(frames, rots, trans)
    recon_loss = frame_reconstruction_loss(out["frames"], frames)
    noise = torch.randn_like(frames[:, :1])
    noise_pred = out["frames"][:, :1] - frames[:, :1]
    diff_loss = diffusion_noise_loss(noise_pred, noise)
    loss = recon_loss + diff_loss
    loss.backward()
    return {
        "recon_loss": float(recon_loss.detach()),
        "diff_loss": float(diff_loss.detach()),
        "loss": float(loss.detach()),
    }
