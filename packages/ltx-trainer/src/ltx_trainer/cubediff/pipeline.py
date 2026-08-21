"""Training demo and table checks."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.cubediff.benchmarks import (
    TABLE1_LAVAL,
    TABLE2_DATA_ABLATION_LAVAL,
    cubediff_beats_mvdiffusion_laval,
)
from ltx_trainer.cubediff.config import CubeDiffConfig
from ltx_trainer.cubediff.cubediff_net import CubeDiffStub
from ltx_trainer.cubediff.cubemap import crop_face_overlap


def evaluation_demo_run() -> dict[str, Any]:
    model = CubeDiffStub(CubeDiffConfig())
    cond = torch.randn(1, 3, 64, 64)
    text = torch.randn(1, 64)
    with torch.no_grad():
        out = model(cond_face=cond, text_emb=text)
    return {
        "panorama_shape": list(out["panorama"].shape),
        "faces_shape": list(out["faces"].shape),
        "laval_fid_ours": TABLE1_LAVAL["Ours_img_txt"]["fid"],
        "beats_mvdiffusion": cubediff_beats_mvdiffusion_laval(),
    }


def train_step() -> dict[str, float]:
    cfg = CubeDiffConfig()
    model = CubeDiffStub(cfg)
    erp = torch.randn(1, 3, 128, 256, requires_grad=True)
    out = model(erp, text_emb=torch.randn(1, 64))
    loss = out["panorama"].pow(2).mean()
    loss.backward()
    return {"loss": float(loss.detach())}


def ablation_table_check() -> dict[str, bool]:
    faces = torch.randn(1, 6, 3, 32, 32)
    cropped = crop_face_overlap(faces, train_fov=95.0, crop_fov=90.0)
    return {
        "beats_mvdiffusion_laval": cubediff_beats_mvdiffusion_laval(),
        "ours_img_txt_best_among_baselines": TABLE1_LAVAL["Ours_img_txt"]["fid"]
        < TABLE1_LAVAL["Diffusion360"]["fid"],
        "multitxt_highest_cs_laval": TABLE1_LAVAL["Ours_img_multitxt"]["cs"]
        > TABLE1_LAVAL["Ours_img_txt"]["cs"],
        "overlap_crops_faces": cropped.shape[-1] < faces.shape[-1],
        "full_beats_tiny_ablation": TABLE2_DATA_ABLATION_LAVAL["Oursfull"]["fid"]
        < TABLE2_DATA_ABLATION_LAVAL["Ourstiny"]["fid"],
    }
