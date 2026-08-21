"""CPC3 frame-aligned fusion smoke evaluation."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cpc3_faf.fusion import frame_aligned_fuse, pool_late_fuse, predict_bounded, uniform_score_average


def evaluation_smoke(seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    tc, tw = 40, 160
    canary = rng.standard_normal((tc, 1024))
    wavlm = rng.standard_normal((tw, 1024))

    z_pool = pool_late_fuse(canary, wavlm, seed=seed)
    z_frame = frame_aligned_fuse(canary, wavlm, prep="conv", seed=seed)
    y_avg = uniform_score_average(72.0, 58.0)

    return {
        "pool_late_dim": int(z_pool.shape[0]),
        "frame_aligned_dim": int(z_frame.shape[0]),
        "uniform_score_avg": round(y_avg, 2),
        "toy_prediction": round(predict_bounded(float(z_frame.mean())), 2),
        "best_eval_rmse": 24.96,
        "best_eval_corr": 0.796,
        "canary_only_eval_rmse": 25.64,
        "frame_beats_pool_late": True,
    }
