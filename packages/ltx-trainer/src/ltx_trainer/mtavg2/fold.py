"""Fold MTAVG-Bench 2.0 cinematic expressiveness proxies into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mtavg2.benchmarks import FIG4_FAILURE_RATE_PCT
from ltx_trainer.mtavg2.config import MTAVG2Config
from ltx_trainer.mtavg2.diagnosis import diagnose_clip_proxy


def mtavg2_meta_block() -> dict[str, Any]:
    return {
        "mtavg2": {
            "arxiv_id": "2605.28035",
            "fold_role": "cinematic_expressiveness_proxy",
            "benchmark": "MTAVG-Bench 2.0",
            "taxonomy": "acting,atmosphere,cinematography",
        }
    }


def _latent_stats(latents: Any) -> tuple[float, float]:
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    if arr.ndim >= 2:
        t_axis = 0 if arr.shape[0] <= 32 else 1
        temporal_diff = float(np.mean(np.abs(np.diff(arr, axis=t_axis)))) if arr.shape[t_axis] > 1 else 0.0
    else:
        temporal_diff = 0.0
    spatial_std = float(arr.std())
    return temporal_diff, spatial_std


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach per-shard proxies for scene-level failure risk (not trained omni evaluators)."""
    cfg = MTAVG2Config()
    latents = data.get("latents")
    out = dict(data)
    out.update(mtavg2_meta_block())
    ref = FIG4_FAILURE_RATE_PCT.get(cfg.ltx_model_label, {})

    if latents is None:
        out["mtavg2"].update({"expressiveness_risk_proxy": None})
        return out

    td, ss = _latent_stats(latents)
    # Sub-dimension risk proxies from latent heuristics (higher = more failure risk).
    acting_risk = float(np.clip(0.5 * td + 0.3 * min(ss, 1.0), 0.0, 1.0))
    atmosphere_risk = float(np.clip(0.4 * min(ss, 1.0) + 0.2 * (1.0 - min(td, 1.0)), 0.0, 1.0))
    cinema_risk = float(np.clip(0.6 * td + 0.2 * min(ss, 1.0), 0.0, 1.0))
    expressiveness_risk = float(np.clip((acting_risk + atmosphere_risk + cinema_risk) / 3.0, 0.0, 1.0))

    diag_dp = diagnose_clip_proxy(sub_dim="DP", latent_temporal_diff=td, latent_spatial_std=ss)
    diag_ct = diagnose_clip_proxy(sub_dim="CT", latent_temporal_diff=td, latent_spatial_std=ss)

    out["mtavg2"].update(
        {
            "temporal_diff": round(td, 4),
            "spatial_std": round(ss, 4),
            "acting_risk_proxy": round(acting_risk, 4),
            "atmosphere_risk_proxy": round(atmosphere_risk, 4),
            "cinematography_risk_proxy": round(cinema_risk, 4),
            "expressiveness_risk_proxy": round(expressiveness_risk, 4),
            "reference_ltx_avg_failure_rate": ref.get("average"),
            "sample_diagnosis_dp": diag_dp["predicted_failure_mode"],
            "sample_diagnosis_ct": diag_ct["predicted_failure_mode"],
            "n_qa_reference": cfg.n_qa_instances,
        }
    )
    return out
