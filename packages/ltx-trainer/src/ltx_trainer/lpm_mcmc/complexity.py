"""Running-time bounds — Theorems 6.17–6.18."""

from __future__ import annotations

from ltx_trainer.lpm_mcmc.indexing import alpha_size


def fast_sweep_cost(n: int, num_edges: int, k_blocks: int, kappa: int) -> int:
    """Algorithm 9 inner loop: O(|E||A| + nK|A|²)."""
    a_size = alpha_size(kappa)
    return num_edges * a_size + n * k_blocks * a_size * a_size


def faster_sweep_cost(n: int, k_blocks: int) -> int:
    """Algorithm 10 inner loop: O(nK) when κ=1."""
    return n * k_blocks


def naive_sweep_cost(n: int) -> int:
    """Standard MwG with O(|V|) delta per node: O(n²) per sweep."""
    return n * n


def rmf24_sweep_cost(n: int, num_edges: int, k_blocks: int) -> int:
    """[RMF24] O(min(|V|b^{-4}+|E|, |V|²)) — use nK proxy with K ~ b^{-2}."""
    return min(k_blocks * k_blocks * n + num_edges, n * n)
