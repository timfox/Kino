"""K-Means clustering stub + k selection (Sec. 4.4, Tables 6–7)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fairlend.benchmarks import TABLE_6_CLUSTER_METRICS, TABLE_7_CLUSTER_PROFILES
from ltx_trainer.fairlend.config import FairLendConfig


@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: int
    size: int
    avg_income_k: float
    avg_loan_k: float
    avg_dti: float
    avg_cltv: float
    denial_pct: float


def select_k(cfg: FairLendConfig | None = None) -> dict[str, object]:
    """Pick k=5: best silhouette among k ≥ 4 (Table 6)."""
    cfg = cfg or FairLendConfig()
    candidates = [r for r in TABLE_6_CLUSTER_METRICS if r["k"] >= cfg.kmeans_k_min]
    best = max(candidates, key=lambda r: r["silhouette"])
    return {
        "selected_k": best["k"],
        "silhouette": best["silhouette"],
        "wcss": best["wcss"],
        "candidates": TABLE_6_CLUSTER_METRICS,
    }


def cluster_profiles() -> tuple[ClusterProfile, ...]:
    return tuple(
        ClusterProfile(
            cluster_id=int(r["cluster"]),
            size=int(r["size"]),
            avg_income_k=float(r["avg_income_k"]),
            avg_loan_k=float(r["avg_loan_k"]),
            avg_dti=float(r["avg_dti"]),
            avg_cltv=float(r["avg_cltv"]),
            denial_pct=float(r["denial_pct"]),
        )
        for r in TABLE_7_CLUSTER_PROFILES
    )


def zscore_features(rows: list[dict[str, float]]) -> list[list[float]]:
    """StandardScaler stub: mean-center + unit variance per feature."""
    if not rows:
        return []
    keys = list(rows[0].keys())
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    stds = {
        k: max(1e-9, (sum((r[k] - means[k]) ** 2 for r in rows) / len(rows)) ** 0.5)
        for k in keys
    }
    return [[(r[k] - means[k]) / stds[k] for k in keys] for r in rows]
