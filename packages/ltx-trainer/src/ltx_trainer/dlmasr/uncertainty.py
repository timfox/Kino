"""Token-level NLL uncertainty trajectories (Sec. 3.3, Eqs. 4–6)."""

from __future__ import annotations

from typing import Any

import numpy as np


def token_nll(confidence: float, eps: float = 1e-12) -> float:
    """Eq. 4 — U_r^i = -log p(token|·) from confidence proxy."""
    return float(-np.log(max(confidence, eps)))


def cumulative_uncertainty(
    commit_rounds: dict[int, int],
    confidences: dict[int, float],
) -> list[tuple[float, float]]:
    """
    Eq. 5 — U_cum_r vs normalized progress P_r = |C_r|/L.

    Returns list of (progress, cumulative_nll) after each round boundary.
    """
    if not commit_rounds:
        return [(0.0, 0.0)]
    seq_len = max(commit_rounds) + 1
    max_round = max(commit_rounds.values())
    by_round: dict[int, list[int]] = {r: [] for r in range(max_round + 1)}
    for pos, rnd in commit_rounds.items():
        by_round[rnd].append(pos)

    committed: set[int] = set()
    cum = 0.0
    trajectory: list[tuple[float, float]] = [(0.0, 0.0)]
    for r in range(max_round + 1):
        for pos in by_round[r]:
            committed.add(pos)
            cum += token_nll(confidences.get(pos, 0.5))
        trajectory.append((len(committed) / seq_len, cum))
    return trajectory


def ar_reference_uncertainty(seq_len: int, confidences: list[float]) -> list[tuple[float, float]]:
    """Eq. 6 — left-to-right AR cumulative NLL reference."""
    cum = 0.0
    traj: list[tuple[float, float]] = [(0.0, 0.0)]
    for i, conf in enumerate(confidences, start=1):
        cum += token_nll(conf)
        traj.append((i / seq_len, cum))
    return traj


def uncertainty_gap_at_progress(
    parallel_traj: list[tuple[float, float]],
    ar_traj: list[tuple[float, float]],
    progress: float = 1.0,
) -> float:
    """Gap between parallel and AR cumulative uncertainty at given progress."""
    def _interp(traj: list[tuple[float, float]], p: float) -> float:
        for i in range(len(traj) - 1):
            p0, u0 = traj[i]
            p1, u1 = traj[i + 1]
            if p0 <= p <= p1:
                if p1 == p0:
                    return u1
                t = (p - p0) / (p1 - p0)
                return u0 + t * (u1 - u0)
        return traj[-1][1]

    return _interp(parallel_traj, progress) - _interp(ar_traj, progress)


def uncertainty_smoke(seq_len: int = 32, *, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    confidences = {i: float(rng.beta(20, 1)) for i in range(seq_len)}
    commit_rounds = {}
    masked = set(range(seq_len))
    rnd = 0
    while masked:
        rnd += 1
        batch = sorted(masked, key=lambda i: confidences[i], reverse=True)[: max(1, len(masked) // 4)]
        for i in batch:
            commit_rounds[i] = rnd
            masked.remove(i)

    parallel = cumulative_uncertainty(commit_rounds, confidences)
    ar = ar_reference_uncertainty(seq_len, [confidences[i] for i in range(seq_len)])
    return {
        "parallel_final_nll": parallel[-1][1],
        "ar_final_nll": ar[-1][1],
        "gap_at_full": uncertainty_gap_at_progress(parallel, ar, 1.0),
        "num_rounds": rnd,
    }
