"""Fold AVBench-style alignment proxies into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.avbench.config import AVBenchConfig


def avbench_meta_block() -> dict[str, Any]:
    return {
        "avbench": {
            "arxiv_id": "2605.24652",
            "fold_role": "short_t2av_alignment_proxy",
            "reference_suite": "ten human-aligned dimensions (VT/AT/AV)",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Toy cross-modal alignment score from latent statistics (not trained AVBench evaluators)."""
    cfg = AVBenchConfig()
    latents = data.get("latents")
    out = dict(data)
    out.update(avbench_meta_block())
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    # Temporal smoothness + bounded spatial energy as alignment proxies
    if arr.ndim >= 2:
        t_axis = 0 if arr.shape[0] <= 32 else 1
        temporal_diff = float(np.mean(np.abs(np.diff(arr, axis=t_axis)))) if arr.shape[t_axis] > 1 else 0.0
    else:
        temporal_diff = 0.0
    spatial_std = float(arr.std())
    vt_align = float(np.clip(1.0 - temporal_diff * 0.5, 0.0, 1.0))
    av_align = float(np.clip(0.85 * vt_align + 0.15 * (1.0 - min(spatial_std, 1.0)), 0.0, 1.0))
    yes_no_score = float(np.clip(0.5 + 0.5 * av_align, 0.0, 1.0))

    out["avbench"].update(
        {
            "vt_align_proxy": round(vt_align, 4),
            "av_align_proxy": round(av_align, 4),
            "yes_no_score_proxy": round(yes_no_score, 4),
            "n_prompts_reference": cfg.n_prompts_total,
        }
    )
    return out
