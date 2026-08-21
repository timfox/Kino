"""Fold VSR-VQA quality proxy scores into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vsr_vqa.config import VSRVQAConfig


def vsr_vqa_meta_block() -> dict[str, Any]:
    return {
        "vsr_vqa": {
            "arxiv_id": "2605.25940",
            "fold_role": "video_quality_proxy",
            "reference_metric": "LPIPS_Alex",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = VSRVQAConfig()
    latents = data.get("latents")
    out = dict(data)
    out.update(vsr_vqa_meta_block())
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    # Proxy objective score: lower spatial variance → higher perceived quality in toy correlation
    spatial_std = float(arr.std())
    mos_proxy = float(np.clip(5.0 - spatial_std * 2.0, 1.0, 5.0))
    lpips_proxy = float(np.clip(0.2 + spatial_std, 0.0, 1.0))

    try:
        import torch
        from ltx_trainer.vsr_vqa.correlation import pearson_correlation

        mos_t = torch.linspace(2.0, 4.5, 6)
        obj_t = 5.0 - mos_t + torch.randn(6) * 0.02
        plcc = pearson_correlation(obj_t, mos_t)
    except ImportError:
        plcc = -lpips_proxy

    out["vsr_vqa"].update(
        {
            "mos_proxy": round(mos_proxy, 3),
            "lpips_proxy": round(lpips_proxy, 4),
            "plcc_toy": round(float(plcc), 4),
            "num_pvs": cfg.num_pvs,
        }
    )
    return out
