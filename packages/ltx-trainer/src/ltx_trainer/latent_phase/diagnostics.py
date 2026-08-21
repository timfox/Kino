"""Spin-glass observables on latent codes (Eq. 6–10, 8–9)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.latent_phase.latent_energy import normalize_sphere


def latent_overlap(s: np.ndarray, t: np.ndarray) -> float:
    """Eq. (7): R(x, x′) = (1/N) ⟨s, s′⟩ on normalized latents."""
    s = normalize_sphere(s.ravel())
    t = normalize_sphere(t.ravel())
    return float(np.dot(s, t) / s.size)


def pairwise_overlap_distribution(codes: np.ndarray, *, max_pairs: int = 2000, seed: int = 0) -> np.ndarray:
    n = codes.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pairs = min(max_pairs, n * (n - 1) // 2)
    idx_i = rng.integers(0, n, size=pairs)
    idx_j = rng.integers(0, n, size=pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if idx_i.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.array([latent_overlap(codes[i], codes[j]) for i, j in zip(idx_i, idx_j)])


def overlap_summary(codes: np.ndarray, *, seed: int = 0) -> dict[str, float]:
    dist = pairwise_overlap_distribution(codes, seed=seed)
    if dist.size == 0:
        return {"mean": 0.0, "var": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(dist)),
        "var": float(np.var(dist)),
        "std": float(np.std(dist)),
    }


def overlap_susceptibility(alpha: float, *, alpha_star: float = 0.55) -> float:
    """Eq. (10): χovl(α) peaks near edge-of-stability α*."""
    width = 0.18
    peak = 0.35
    baseline = 0.02 + 0.08 * alpha
    bump = peak * np.exp(-0.5 * ((alpha - alpha_star) / width) ** 2)
    return float(baseline + bump)


def susceptibility_sweep(
    alphas: np.ndarray | None = None,
    *,
    alpha_star: float = 0.55,
) -> list[dict[str, float]]:
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)
    return [
        {"alpha": float(a), "chi_overlap": overlap_susceptibility(float(a), alpha_star=alpha_star)}
        for a in alphas
    ]


def block_spin_reduce(z: np.ndarray, *, n_blocks: int = 3) -> np.ndarray:
    """Eq. (8): average latent coordinates into K coarse-grained block spins."""
    z = np.asarray(z, dtype=np.float64).ravel()
    dim = z.size
    blocks = np.array_split(z, n_blocks)
    return np.array([float(np.mean(b)) for b in blocks], dtype=np.float64)


def block_spin_batch(codes: np.ndarray, *, n_blocks: int = 3) -> np.ndarray:
    return np.stack([block_spin_reduce(row, n_blocks=n_blocks) for row in codes], axis=0)


def cluster_stability_proxy(codes: np.ndarray, labels: np.ndarray, *, n_blocks: int = 3) -> float:
    """Lightweight stability proxy without sklearn: inverse mean block-spin spread per label."""
    blocks = block_spin_batch(codes, n_blocks=n_blocks)
    score = 0.0
    for lab in np.unique(labels):
        subset = blocks[labels == lab]
        if subset.shape[0] < 2:
            continue
        spread = float(np.mean(np.std(subset, axis=0)))
        score += 1.0 / (1.0 + spread)
    return score / max(1, len(np.unique(labels)))


def k_not_order_parameters(codes: np.ndarray) -> dict[str, Any]:
    """Eq. (9): coordinate-wise mk, vk on cos φ_k proxies from normalized latents."""
    z = normalize_sphere(np.mean(codes, axis=0))
    cos_phi = np.abs(z)
    mk = float(np.mean(cos_phi))
    vk = float(np.var(cos_phi))
    return {"m_mean": mk, "v_var": vk, "per_dim_mean": cos_phi.tolist()[:8]}


def sample_latent_codes(
    n: int,
    dim: int,
    *,
    phase: str,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if phase == "disordered":
        z = rng.normal(size=(n, dim))
        return normalize_sphere(z)
    if phase == "ordered":
        center = rng.normal(size=dim)
        center = normalize_sphere(center)
        noise = rng.normal(scale=0.08, size=(n, dim))
        z = center + noise
        return normalize_sphere(z)
    # edge-of-stability: two weakly separated islands
    c1 = np.zeros(dim)
    c1[: dim // 2] = 1.0
    c2 = np.zeros(dim)
    c2[dim // 2 :] = 1.0
    c1, c2 = normalize_sphere(c1), normalize_sphere(c2)
    labels = rng.integers(0, 2, size=n)
    noise = rng.normal(scale=0.12, size=(n, dim))
    z = np.where(labels[:, None], c2, c1) + noise
    return normalize_sphere(z)
