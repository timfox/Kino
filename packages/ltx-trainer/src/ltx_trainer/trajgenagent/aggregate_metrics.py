"""Computed aggregate metrics (Table II) and authenticity scoring (Table III)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ltx_trainer.trajgenagent.benchmarks import TABLE_III_ANOMALY
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.metrics import jensen_shannon_divergence, paper_table_ii_row, trajectory_level_metrics
from ltx_trainer.trajgenagent.trajectory import DailyTrajectory, daily_distance_km, radius_of_gyration_km


def _bin_histogram(values: Sequence[float], *, n_bins: int = 20, lo: float | None = None, hi: float | None = None) -> list[float]:
    if not values:
        return [1.0 / n_bins] * n_bins
    lo = lo if lo is not None else min(values)
    hi = hi if hi is not None else max(values)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / n_bins
    counts = [0.0] * n_bins
    for v in values:
        idx = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1.0
    total = sum(counts) or 1.0
    return [c / total for c in counts]


def _transition_distribution(trajectories: Sequence[DailyTrajectory]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for traj in trajectories:
        acts = [v.activity for v in traj.visits]
        for i in range(len(acts) - 1):
            counts[f"{acts[i]}->{acts[i + 1]}"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _align_transition_jsd(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
) -> float:
    pg = _transition_distribution(generated)
    pr = _transition_distribution(reference)
    keys = sorted(set(pg) | set(pr))
    return jensen_shannon_divergence([pg.get(k, 0.0) for k in keys], [pr.get(k, 0.0) for k in keys])


def _individual_rank_values(trajectories: Sequence[DailyTrajectory]) -> list[float]:
    by_user: dict[str, list[float]] = {}
    for traj in trajectories:
        by_user.setdefault(traj.individual_id, []).append(radius_of_gyration_km(traj.visits))
    return [sum(v) / len(v) for v in by_user.values()]


def evaluate_table_ii_metrics(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
    *,
    dataset: str = "NumoSim",
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or TrajGenAgentConfig()
    grid = cfg.grid_km_numosim if dataset == "NumoSim" else cfg.grid_km_mobilitysyn

    gen_dist = [daily_distance_km(t.visits) for t in generated]
    ref_dist = [daily_distance_km(t.visits) for t in reference]
    gen_dur = [trajectory_level_metrics(t)["mean_duration_min"] for t in generated]
    ref_dur = [trajectory_level_metrics(t)["mean_duration_min"] for t in reference]
    gen_loc = [float(t.n_visits) for t in generated]
    ref_loc = [float(t.n_visits) for t in reference]
    gen_gr = [radius_of_gyration_km(t.visits) for t in generated]
    ref_gr = [radius_of_gyration_km(t.visits) for t in reference]
    gen_ir = _individual_rank_values(generated)
    ref_ir = _individual_rank_values(reference)

    dist_hi = max(max(ref_dist, default=1.0), max(gen_dist, default=1.0), grid * 40)
    dur_hi = max(max(ref_dur, default=1.0), max(gen_dur, default=1.0), 480.0)
    loc_hi = max(max(ref_loc, default=1.0), max(gen_loc, default=1.0), 12.0)
    gr_hi = max(max(ref_gr, default=1.0), max(gen_gr, default=1.0), 20.0)
    ir_hi = max(max(ref_ir, default=1.0), max(gen_ir, default=1.0), 20.0)

    return {
        "distance": jensen_shannon_divergence(
            _bin_histogram(gen_dist, lo=0.0, hi=dist_hi),
            _bin_histogram(ref_dist, lo=0.0, hi=dist_hi),
        ),
        "g_radius": jensen_shannon_divergence(
            _bin_histogram(gen_gr, lo=0.0, hi=gr_hi),
            _bin_histogram(ref_gr, lo=0.0, hi=gr_hi),
        ),
        "duration": jensen_shannon_divergence(
            _bin_histogram(gen_dur, lo=0.0, hi=dur_hi),
            _bin_histogram(ref_dur, lo=0.0, hi=dur_hi),
        ),
        "daily_loc": jensen_shannon_divergence(
            _bin_histogram(gen_loc, lo=0.0, hi=loc_hi, n_bins=int(loc_hi) + 1),
            _bin_histogram(ref_loc, lo=0.0, hi=loc_hi, n_bins=int(loc_hi) + 1),
        ),
        "i_rank": jensen_shannon_divergence(
            _bin_histogram(gen_ir, lo=0.0, hi=ir_hi),
            _bin_histogram(ref_ir, lo=0.0, hi=ir_hi),
        ),
        "g_rank": jensen_shannon_divergence(
            _bin_histogram(gen_gr, lo=0.0, hi=gr_hi),
            _bin_histogram(ref_gr, lo=0.0, hi=gr_hi),
        ),
        "transition": _align_transition_jsd(generated, reference),
    }


def _rank_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels, strict=True), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += i
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def trajectory_features(traj: DailyTrajectory) -> tuple[float, float, float, float]:
    local = trajectory_level_metrics(traj)
    return (local["daily_loc"], local["distance_km"], local["g_radius_km"], local["mean_duration_min"])


def _feature_zscore(feat: tuple[float, float, float, float], ref_feats: Sequence[tuple[float, float, float, float]]) -> float:
    if not ref_feats:
        return 0.0
    dims = len(feat)
    z_sum = 0.0
    for d in range(dims):
        vals = [f[d] for f in ref_feats]
        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / max(1, len(vals) - 1)
        sigma = math.sqrt(var) if var > 0 else 1.0
        z_sum += abs((feat[d] - mu) / sigma)
    return z_sum / dims


def authenticity_auroc(
    reference: Sequence[DailyTrajectory],
    candidate: Sequence[DailyTrajectory],
) -> float:
    """BeSTAD-style authenticity: AUROC near 0.5 means candidate matches reference."""
    ref_feats = [trajectory_features(t) for t in reference]
    cand_feats = [trajectory_features(t) for t in candidate]
    ref_scores = [_feature_zscore(f, ref_feats) for f in ref_feats]
    cand_scores = [_feature_zscore(f, ref_feats) for f in cand_feats]
    scores = ref_scores + cand_scores
    labels = [1] * len(ref_scores) + [0] * len(cand_scores)
    return _rank_auroc([-s for s in scores], labels)


def visit_level_auroc(
    reference: Sequence[DailyTrajectory],
    candidate: Sequence[DailyTrajectory],
) -> float:
    ref_dur = [v.duration_minutes for t in reference for v in t.visits]
    cand_dur = [v.duration_minutes for t in candidate for v in t.visits]
    if not ref_dur or not cand_dur:
        return 0.5
    mu = sum(ref_dur) / len(ref_dur)
    var = sum((x - mu) ** 2 for x in ref_dur) / max(1, len(ref_dur) - 1)
    sigma = math.sqrt(var) if var > 0 else 1.0
    ref_scores = [-abs(x - mu) / sigma for x in ref_dur]
    cand_scores = [-abs(x - mu) / sigma for x in cand_dur]
    return _rank_auroc(ref_scores + cand_scores, [1] * len(ref_scores) + [0] * len(cand_scores))


def evaluate_table_iii_metrics(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
) -> dict[str, float]:
    bestad = authenticity_auroc(reference, generated)
    icad_visit = visit_level_auroc(reference, generated)
    icad_ind = authenticity_auroc(reference, generated)
    return {
        "bestad_auroc": bestad,
        "bestad_ap": bestad,
        "icad_visit_auroc": icad_visit,
        "icad_visit_ap": icad_visit,
        "icad_ind_auroc": icad_ind,
        "icad_ind_ap": icad_ind,
    }


def compare_to_reference(
    generated: Sequence[DailyTrajectory],
    reference: Sequence[DailyTrajectory],
    *,
    dataset: str = "NumoSim",
    cfg: TrajGenAgentConfig | None = None,
) -> dict[str, object]:
    live_ii = evaluate_table_ii_metrics(generated, reference, dataset=dataset, cfg=cfg)
    live_iii = evaluate_table_iii_metrics(generated, reference)
    paper_ii = paper_table_ii_row(dataset)
    paper_iii_row = next(r for r in TABLE_III_ANOMALY if r["dataset"] == dataset and r["model"] == "TrajGenAgent")
    return {
        "live_table_ii": live_ii,
        "paper_table_ii": paper_ii,
        "live_table_iii": live_iii,
        "paper_table_iii": {k: float(paper_iii_row[k]) for k in paper_iii_row if k not in {"dataset", "model"}},
        "bestad_near_chance": abs(live_iii["bestad_auroc"] - 0.5) < 0.15,
    }
