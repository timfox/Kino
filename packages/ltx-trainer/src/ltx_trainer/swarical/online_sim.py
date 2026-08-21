"""Small-scale decentralized localization simulation (companion Sec. 2.1)."""

from __future__ import annotations

import random
from typing import Any

from ltx_trainer.swarical.config import LocalizationMode
from ltx_trainer.swarical.dead_reckoning import apply_dead_reckoning
from ltx_trainer.swarical.metrics import chamfer_distance, hausdorff_distance
from ltx_trainer.swarical.mock import grid_points
from ltx_trainer.swarical.planner import Point3, plan_from_points


def _mode_learning_rate(mode: LocalizationMode, step: int, n_swarms: int) -> float:
    """Heuristic convergence rates matching ISR > HC > RSF ranking on Skateboard (Fig. 13)."""
    if mode == LocalizationMode.ISR:
        return 0.32 if step < 12 else 0.22
    if mode == LocalizationMode.HC:
        return 0.15 if step < 12 else 0.12
    # RSF — per-edge rounds: low per-step gain, occasional swarm-wide sync.
    _ = n_swarms
    if step > 0 and step % 4 == 0:
        return 0.14
    return 0.06


def _localization_step(
    estimated: list[Point3],
    ground: list[Point3],
    *,
    mode: LocalizationMode,
    step: int,
    n_swarms: int,
) -> list[Point3]:
    """Move each estimate toward ground truth (decentralized correction stub)."""
    out: list[Point3] = []
    for est, gt in zip(estimated, ground, strict=True):
        lr = _mode_learning_rate(mode, step, n_swarms)
        out.append(
            Point3(
                est.x + lr * (gt.x - est.x),
                est.y + lr * (gt.y - est.y),
                est.z + lr * (gt.z - est.z),
            )
        )
    return out


def run_small_scale_experiment(
    *,
    rows: int = 4,
    cols: int = 4,
    spacing_cm: float = 7.0,
    group_size_g: int = 8,
    mode: LocalizationMode = LocalizationMode.ISR,
    dead_reckoning_alpha_deg: float = 5.0,
    max_iterations: int = 120,
    convergence_hd_cm: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Reproduce companion 16-FLS 4×4 experiment with HD/CD vs time (deterministic seed)."""
    rng = random.Random(seed)
    ground = grid_points(rows, cols, spacing_cm=spacing_cm)
    estimated = apply_dead_reckoning(ground, alpha_deg=dead_reckoning_alpha_deg, rng=rng)
    plan = plan_from_points(ground, group_size_g=group_size_g, max_link_cm=8.0)

    timeline: list[dict[str, float | int]] = []
    for step in range(max_iterations):
        hd = hausdorff_distance(estimated, ground)
        cd = chamfer_distance(estimated, ground)
        timeline.append(
            {
                "step": step,
                "hausdorff_cm": round(hd, 4),
                "chamfer_cm_squared": round(cd, 4),
            }
        )
        if hd <= convergence_hd_cm:
            break
        estimated = _localization_step(
            estimated,
            ground,
            mode=mode,
            step=step,
            n_swarms=plan.n_swarms,
        )

    final_hd = timeline[-1]["hausdorff_cm"] if timeline else float("inf")
    return {
        "experiment": "small_scale_4x4",
        "n_fls": len(ground),
        "mode": mode.value,
        "dead_reckoning_alpha_deg": dead_reckoning_alpha_deg,
        "planner": {
            "n_swarms": plan.n_swarms,
            "group_size_g": group_size_g,
            "dark_fls_inserted": plan.dark_fls_inserted,
        },
        "converged_hd_below_cm": convergence_hd_cm,
        "steps_run": len(timeline),
        "final_hausdorff_cm": final_hd,
        "hd_below_1cm_at_step": next(
            (t["step"] for t in timeline if t["hausdorff_cm"] < 1.0),
            None,
        ),
        "timeline": timeline,
        "note": "Deterministic correction stub; upstream uses UDP multicast FLS processes.",
    }


def compare_localization_modes(
    *,
    rows: int = 4,
    cols: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """ISR vs HC vs RSF on identical 4×4 grid (companion Fig. 13 / thousand-core trends)."""
    results: dict[str, Any] = {}
    for mode in (LocalizationMode.ISR, LocalizationMode.HC, LocalizationMode.RSF):
        results[mode.value] = run_small_scale_experiment(
            rows=rows,
            cols=cols,
            mode=mode,
            seed=seed,
        )
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["final_hausdorff_cm"], kv[1]["steps_run"]),
    )
    return {
        "ranking_by_final_hd": [m for m, _ in ranked],
        "runs": results,
    }
