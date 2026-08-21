"""Light100K continuous pseudo-pair construction and filtering (Sec. 3.1, Appendix A)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.controllight.config import ControlLightConfig, Light100KGroup
from ltx_trainer.controllight.nr_metrics import interpolation_trajectory_metrics
from ltx_trainer.controllight.retinex import build_light100k_group, luminance_y


def sobel_magnitude(image: Tensor) -> Tensor:
    """Sobel edge magnitude for structural consistency filtering (Appendix A)."""
    if image.shape[-3] != 3:
        raise ValueError("expected CHW RGB")
    gray = luminance_y(image.clamp(0, 1))
    if gray.dim() == 2:
        gray = gray.unsqueeze(0).unsqueeze(0)
    elif gray.dim() == 3:
        gray = gray.unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(1)
    return mag.squeeze(0) if mag.shape[0] == 1 else mag


def edge_consistency_score(i0: Tensor, i1: Tensor, *, quantile: float = 0.9) -> float:
    """Mean |Sobel(I0) - Sobel(I1)| on strong edges — lower is more consistent."""
    e0 = sobel_magnitude(i0)
    e1 = sobel_magnitude(i1)
    thr = torch.quantile(e0.flatten(), quantile)
    mask = e0 >= thr
    if mask.sum() == 0:
        return float((e0 - e1).abs().mean().item())
    return float((e0 - e1).abs()[mask].mean().item())


def pair_passes_edge_filter(
    i0: Tensor,
    i1: Tensor,
    *,
    max_edge_diff: float = 0.35,
) -> bool:
    """Appendix A: drop pairs with obvious edge shifts."""
    return edge_consistency_score(i0, i1) <= max_edge_diff


@dataclass
class Light100KPairRecord:
    pair_id: str
    edge_score: float
    passed: bool
    strengths: tuple[float, ...]


def build_pair_record(
    pair_id: str,
    i0: Tensor,
    i1: Tensor,
    *,
    cfg: ControlLightConfig | None = None,
) -> Light100KPairRecord:
    cfg = cfg or ControlLightConfig()
    score = edge_consistency_score(i0, i1)
    passed = score <= cfg.edge_filter_max_diff
    return Light100KPairRecord(
        pair_id=pair_id,
        edge_score=score,
        passed=passed,
        strengths=cfg.enhancement_strengths,
    )


def make_training_group(
    i0: Tensor,
    i1: Tensor,
    *,
    cfg: ControlLightConfig | None = None,
    use_retinex: bool = True,
) -> Light100KGroup:
    cfg = cfg or ControlLightConfig()
    targets = build_light100k_group(
        i0,
        i1,
        strengths=cfg.enhancement_strengths,
        use_retinex=use_retinex,
        beta_scale=cfg.reflectance_beta_scale,
    )
    return Light100KGroup(i0=i0, targets=targets)


def compare_interpolation_strategies(
    i0: Tensor,
    i1: Tensor,
    *,
    cfg: ControlLightConfig | None = None,
) -> dict[str, Any]:
    """Table 6 style: Retinex vs alpha-blend trajectory on NIQE/MUSIQ proxies."""
    cfg = cfg or ControlLightConfig()
    grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    retinex_g = build_light100k_group(i0, i1, strengths=grid[1:-1], use_retinex=True)
    alpha_g = build_light100k_group(i0, i1, strengths=grid[1:-1], use_retinex=False)
    ret = interpolation_trajectory_metrics(retinex_g, strengths=grid)
    alp = interpolation_trajectory_metrics(alpha_g, strengths=grid)
    # Paper: Retinex keeps higher NIQE at low s (more degradation cue), smoother MUSIQ ramp
    retinex_monotone_musiq = all(ret["MUSIQ"][i] <= ret["MUSIQ"][i + 1] + 2.0 for i in range(len(ret["MUSIQ"]) - 2))
    return {
        "strengths": list(grid),
        "retinex": ret,
        "alpha_blend": alp,
        "retinex_monotone_musiq_tail": retinex_monotone_musiq,
        "i0_niqe_higher_than_mid_alpha": ret["NIQE"][0] >= alp["NIQE"][2] - 0.05,
    }


def export_group_manifest(
    pair_id: str,
    group: dict[float, Tensor],
    out_dir: str | Path,
) -> Path:
    """Write JSON manifest for offline weight-map precompute (Fig. 5)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pair_id": pair_id,
        "strengths": [float(s) for s in sorted(group.keys())],
        "shapes": {str(s): list(group[s].shape) for s in group},
    }
    path = out / f"{pair_id}_light100k.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_tensor_image(path: str | Path, *, max_side: int | None = 1024) -> Tensor:
    """Load RGB [0,1] CHW from file (PIL optional)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    import numpy as np

    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return arr


def build_light100k_from_paths(
    i0_path: str | Path,
    i1_path: str | Path,
    *,
    pair_id: str | None = None,
    cfg: ControlLightConfig | None = None,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    """End-to-end: load pair → filter → Retinex group G → optional manifest."""
    cfg = cfg or ControlLightConfig()
    i0 = load_tensor_image(i0_path, max_side=cfg.train_resolution)
    i1 = load_tensor_image(i1_path, max_side=cfg.train_resolution)
    pid = pair_id or Path(i0_path).stem
    rec = build_pair_record(pid, i0, i1, cfg=cfg)
    group = build_light100k_group(
        i0,
        i1,
        strengths=cfg.enhancement_strengths,
        use_retinex=True,
        beta_scale=cfg.reflectance_beta_scale,
    )
    out: dict[str, Any] = {
        "pair_id": pid,
        "edge_score": rec.edge_score,
        "passed_filter": rec.passed,
        "n_targets": len(group),
        "strengths": sorted(group.keys()),
    }
    if export_dir is not None:
        out["manifest"] = str(export_group_manifest(pid, group, export_dir))
    return out
