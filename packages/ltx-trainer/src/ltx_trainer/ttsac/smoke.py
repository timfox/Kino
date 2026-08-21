"""Torch-free evaluation_smoke for TT-SAC."""

from __future__ import annotations


def evaluation_smoke(*, seed: int = 0) -> dict[str, object]:
    _ = seed
    sigma2 = 0.04
    k = 8
    var_agg = sigma2 / k
    best_k, best_obj = 1, float("inf")
    for trial_k in range(1, 11):
        obj = sigma2 / trial_k + (0.02 * trial_k) ** 2
        if obj < best_obj:
            best_obj = obj
            best_k = trial_k
    return {
        "iid_var_at_k8": round(var_agg, 6),
        "optimal_k": best_k,
        "torch_available": False,
    }
