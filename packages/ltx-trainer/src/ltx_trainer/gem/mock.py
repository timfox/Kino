"""Toy hypersphere clustering smoke for GEM."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.gem.config import GemConfig
from ltx_trainer.gem.balance import mixing_balance_value
from ltx_trainer.gem.gis import top_gis_indices
from ltx_trainer.gem.mm import fit_gem
from ltx_trainer.gem.tables import (
    ablation_clustering_avg,
    headline_results,
    student_distillation_accuracy,
    table1_gem_breakdown,
)
from ltx_trainer.gem.vmf import normalize_rows


def _toy_anisotropic_embeddings(
    n: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Strong hub (cone effect) + two minor semantic directions."""
    hub = np.zeros(dim)
    hub[0] = 1.0
    dir_a = np.zeros(dim)
    dir_a[1] = 1.0
    dir_b = np.zeros(dim)
    dir_b[2] = 1.0
    vecs = []
    n_hub = int(0.72 * n)
    n_a = int(0.14 * n)
    for _ in range(n_hub):
        vecs.append(hub + rng.normal(scale=0.08, size=dim))
    for _ in range(n_a):
        vecs.append(dir_a + rng.normal(scale=0.18, size=dim))
    for _ in range(n - n_hub - n_a):
        vecs.append(dir_b + rng.normal(scale=0.18, size=dim))
    return normalize_rows(np.stack(vecs, axis=0))


def _euclidean_kmeans_mass(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Lloyd on sphere for collapse comparison."""
    n = len(x)
    centroids = x[rng.choice(n, size=k, replace=False)]
    assign = np.zeros(n, dtype=int)
    for _ in range(20):
        sims = x @ centroids.T
        assign = sims.argmax(axis=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centroids[j] = normalize_rows(centroids[j : j + 1] + (x[mask].mean(axis=0, keepdims=True)))[0]
            else:
                centroids[j] = x[rng.integers(0, n)]
    counts = np.bincount(assign, minlength=k).astype(np.float64)
    return counts / counts.sum()


def evaluation_smoke(cfg: GemConfig | None = None) -> dict[str, Any]:
    c = cfg or GemConfig()
    rng = np.random.default_rng(7)
    n, dim, k = 180, 16, 6
    x = _toy_anisotropic_embeddings(n, dim, rng)

    cfg_balanced = GemConfig(
        n_clusters=k,
        balance_lambda=10.0,
        max_mm_iters=40,
        e_step_mirror_steps=12,
        gis_neighbors=c.gis_neighbors,
        gis_top_m=c.gis_top_m,
    )
    cfg_vanilla = GemConfig(
        n_clusters=k,
        balance_lambda=0.0,
        max_mm_iters=40,
        e_step_mirror_steps=6,
    )
    seed = np.random.default_rng(7)
    vanilla = fit_gem(x, cfg_vanilla, rng=seed)
    gem = fit_gem(x, cfg_balanced, rng=np.random.default_rng(7))
    eucl_mass = _euclidean_kmeans_mass(x, k, rng)

    mono = len(gem.objective) >= 2 and gem.objective[-1] >= gem.objective[0] - 1e-6
    lam = cfg_balanced.balance_lambda
    balance_beats_vanilla = mixing_balance_value(gem.pi, lam) > mixing_balance_value(
        vanilla.pi, lam
    )

    cluster0 = int(np.argmax(gem.pi))
    reps = top_gis_indices(
        x,
        gem.gamma,
        gem.mu,
        gem.kappa,
        cluster0,
        top_m=3,
        beta=c.gis_beta,
        neighbor_m=c.gis_neighbors,
    )

    distill = student_distillation_accuracy()
    t1 = table1_gem_breakdown()

    return {
        "paper": f"arXiv:{c.paper_arxiv}",
        "n_samples": n,
        "dim": dim,
        "n_clusters": k,
        "mm_iters": gem.n_iters,
        "objective_monotone": mono,
        "final_objective": round(gem.objective[-1], 4) if gem.objective else None,
        "pi_min": round(float(gem.pi.min()), 4),
        "pi_max": round(float(gem.pi.max()), 4),
        "mixing_balance_R_gem": round(mixing_balance_value(gem.pi, lam), 4),
        "mixing_balance_R_vanilla": round(mixing_balance_value(vanilla.pi, lam), 4),
        "balance_reduces_collapse": balance_beats_vanilla,
        "gis_representatives_cluster0": reps,
        "student_accuracy_gem": distill["gem_student_accuracy"],
        "student_accuracy_kmeans": distill["kmeans_student_accuracy"],
        "paper_average_doremi": t1["average_doremi"],
        "paper_average_perf": t1["average_perf"],
        "ablation_gem_average": next(
            r["average"] for r in ablation_clustering_avg() if r["method"] == "GEM (Ours)"
        ),
        "headlines": headline_results(),
    }
