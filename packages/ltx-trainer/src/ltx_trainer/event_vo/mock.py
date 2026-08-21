"""Paper stub smoke for event monocular ESKF VO."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.event_vo.config import EventVOConfig
from ltx_trainer.event_vo.eskf import MonocularEventESKF
from ltx_trainer.event_vo.pipeline import run_simulation_odometry
from ltx_trainer.event_vo.rate import RateTrackerStub


def evaluation_smoke() -> dict[str, Any]:
    cfg = EventVOConfig()
    sim = run_simulation_odometry(cfg, seed=0, noise_px=0.5)
    filt = MonocularEventESKF(cfg)
    filt.mark_initialized()
    tracker = RateTrackerStub(seed=1, n_features=20)
    n = 0
    for upd in tracker.stream_updates([0.0, 0.02, 0.04], global_shift=np.array([0.5, 0.0])):
        filt.process_update(upd)
        n += 1
    return {
        "package": "event_vo",
        "paper": "arXiv:2605.27661",
        "filter_initialized": filt.initialized,
        "rate_updates_processed": n,
        "simulation_ok": sim.get("ok", False),
        "sim_ate_mean_m": sim.get("metrics", {}).get("ate_mean_m"),
        "paper_sim_mape_m": cfg.sim_mape_m,
        "n_landmarks_sim": sim.get("n_landmarks"),
    }
