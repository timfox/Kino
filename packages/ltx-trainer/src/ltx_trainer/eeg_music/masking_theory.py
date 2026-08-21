"""Lemma 4.1 / Theorem 4.2 — channel vs block masking overlap."""

from __future__ import annotations


def retention_constants(rho: float) -> tuple[float, float]:
    """a = ρ² + (1-ρ)², b = (1-ρ)² / a."""
    a = rho * rho + (1.0 - rho) * (1.0 - rho)
    b = (1.0 - rho) * (1.0 - rho) / a
    return a, b


def block_concentration_score(diff_indices: set[int], blocks: list[set[int]]) -> int:
    """S(D) = |D| - |B(D)| per paper."""
    touched = {i for i, blk in enumerate(blocks) if blk & diff_indices}
    return len(diff_indices) - len(touched)


def kernel_ratio(
    diff_indices: set[int],
    *,
    rho: float,
    num_coords: int,
    blocks: list[set[int]],
) -> float:
    """K_cha / K_blk = a^{N-K} b^{S(D)} (Lemma 4.1)."""
    a, b = retention_constants(rho)
    s_d = block_concentration_score(diff_indices, blocks)
    k_blocks = len(blocks)
    return (a ** (num_coords - k_blocks)) * (b**s_d)


def cross_cluster_overlap_ratio(
    y_same: list[bool],
    s_d: list[int],
    *,
    b: float,
) -> tuple[float, float]:
    """
    Compare normalized overlap under channel vs block masking weights.

    Returns (r_channel, r_block) using reweighting identity; Theorem 4.2 holds when
    Cov(Y, b^S) < 0 under measure Q (simplified empirical check here).
    """
    if not y_same:
        return 0.0, 0.0
    weights = [b**s for s in s_d]
    cross_ch = sum(w * (0.0 if same else 1.0) for w, same in zip(weights, y_same, strict=True))
    cross_blk = sum(0.0 if same else 1.0 for same in y_same)
    z_ch = sum(weights)
    z_blk = float(len(y_same))
    return cross_ch / max(z_ch, 1e-9), cross_blk / max(z_blk, 1e-9)
