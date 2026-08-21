"""Training / ablation demos."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.daovi.benchmarks import TABLE2_ABLATION
from ltx_trainer.daovi.config import DaoviConfig
from ltx_trainer.daovi.daovi_net import DaoviStub
from ltx_trainer.daovi.distortion import distortion_map, erp_distortion_weight
from ltx_trainer.daovi.geodesic import flow_consistency_error, geodesic_distance_pixels
from ltx_trainer.daovi.losses import masked_reconstruction_loss
from ltx_trainer.daovi.synthetic import synthetic_depth, synthetic_erp_video, synthetic_flow, synthetic_masks


def evaluation_demo_run(cfg: DaoviConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DaoviConfig(height=48, width=96, num_frames=4)
    model = DaoviStub(cfg)
    frames = synthetic_erp_video(cfg)
    masks = synthetic_masks(cfg)
    fwd, bwd = synthetic_flow(cfg)
    depth = synthetic_depth(cfg)
    with torch.no_grad():
        out = model(frames, masks, fwd, bwd, depth)
    return {
        "frames_shape": list(frames.shape),
        "recon_shape": list(out["frames"].shape),
        "partial_shape": list(out["partial"].shape),
    }


def ablation_modules(cfg: DaoviConfig | None = None) -> dict[str, float]:
    cfg = cfg or DaoviConfig(height=32, width=64, num_frames=3)
    frames = synthetic_erp_video(cfg)
    masks = synthetic_masks(cfg)
    fwd, bwd = synthetic_flow(cfg)
    depth = synthetic_depth(cfg)
    full = DaoviStub(cfg)
    no_gfcip = DaoviStub(replace(cfg, use_gfcip=False))
    no_odafp = DaoviStub(replace(cfg, use_odafp=False))
    with torch.no_grad():
        f = full(frames, masks, fwd, bwd, depth)["frames"].mean()
        g = no_gfcip(frames, masks, fwd, bwd, depth)["frames"].mean()
        o = no_odafp(frames, masks, fwd, bwd, depth)["frames"].mean()
    return {
        "full_mean": float(f),
        "no_gfcip_mean": float(g),
        "no_odafp_mean": float(o),
        "table2_full_psnr": TABLE2_ABLATION["DAOVI_full"]["psnr"],
        "table2_no_odafp_psnr": TABLE2_ABLATION["w/o_ODAFP"]["psnr"],
    }


def geodesic_demo(width: int = 96, height: int = 48) -> dict[str, Any]:
    """Pole vs equator: same ERP pixel offset, different geodesic error."""
    p = (width // 2, height // 4)
    p_prime_equator = (p[0] + 4.0, p[1])
    p_prime_pole = (p[0] + 4.0, 2.0)
    d_eq = geodesic_distance_pixels(p, p_prime_equator, width, height)
    d_po = geodesic_distance_pixels(p, p_prime_pole, width, height)
    return {"equator_offset_deg": d_eq, "pole_offset_deg": d_po, "pole_larger": d_po > d_eq}


def distortion_demo(height: int = 48, width: int = 96) -> dict[str, Any]:
    w = erp_distortion_weight(height)
    dmap = distortion_map(height, width)
    return {
        "equator_weight": float(w[height // 2]),
        "pole_weight": float(w[0]),
        "map_shape": list(dmap.shape),
    }


def train_step(cfg: DaoviConfig | None = None) -> dict[str, float]:
    cfg = cfg or DaoviConfig(height=32, width=64, num_frames=3)
    model = DaoviStub(cfg)
    frames = synthetic_erp_video(cfg)
    masks = synthetic_masks(cfg)
    fwd, bwd = synthetic_flow(cfg)
    depth = synthetic_depth(cfg)
    out = model(frames, masks, fwd, bwd, depth)
    loss = masked_reconstruction_loss(out["frames"], frames, masks)
    valid, err = flow_consistency_error(fwd[:, 0], bwd[:, 0])
    return {
        "loss": float(loss.detach()),
        "flow_valid_fraction": float(valid.detach().mean()),
        "flow_err_mean_rad": float(err.detach().mean()),
    }
