"""Two-layer weighted GCN phoneme lookup encoder (Sec. 2.3, Eq. 2)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.mdd_lssg.graph import ConfusionGraph


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    out_deg = adj.sum(axis=0, keepdims=True)
    out_deg = np.where(out_deg > 0, out_deg, 1.0)
    return adj / out_deg


def gcn_forward(
    graph: ConfusionGraph,
    *,
    embed_dim: int = 64,
    seed: int = 42,
    dropout: float = 0.0,
) -> np.ndarray:
    """Return H(l) ∈ R^{|V|×d} — shared GCN params, L1-specific adjacency."""
    rng = np.random.default_rng(seed)
    n = len(graph.phonemes)
    x0 = rng.standard_normal((n, embed_dim)) * 0.02
    w1 = rng.standard_normal((embed_dim, embed_dim)) * np.sqrt(2.0 / embed_dim)
    w2 = rng.standard_normal((embed_dim, embed_dim)) * np.sqrt(2.0 / embed_dim)

    a = _normalize_adjacency(graph.adjacency.T)  # message passing: neighbors → node
    h1 = np.tanh(a @ x0 @ w1)
    if dropout > 0:
        h1 *= rng.random(h1.shape) > dropout
    h2 = np.tanh(a @ h1 @ w2)
    return h1 + h2  # residual


def phoneme_embeddings(
    graph: ConfusionGraph,
    canonical: list[str],
    *,
    embed_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Eq. (3): L(l) = H(l)[c]."""
    h = gcn_forward(graph, embed_dim=embed_dim, seed=seed)
    idx = {p: i for i, p in enumerate(graph.phonemes)}
    rows = [idx[p] for p in canonical if p in idx]
    if not rows:
        return np.zeros((0, embed_dim), dtype=np.float64)
    return h[rows]


def embedding_distance(
    graph_a: ConfusionGraph,
    graph_b: ConfusionGraph,
    pair: tuple[str, str],
    *,
    embed_dim: int = 64,
    seed: int = 42,
) -> tuple[float, float]:
    """L2 distance between two phoneme embeddings under each graph."""
    ha = gcn_forward(graph_a, embed_dim=embed_dim, seed=seed)
    hb = gcn_forward(graph_b, embed_dim=embed_dim, seed=seed)
    idx_a = {p: i for i, p in enumerate(graph_a.phonemes)}
    idx_b = {p: i for i, p in enumerate(graph_b.phonemes)}
    p, q = pair
    da = float(np.linalg.norm(ha[idx_a[p]] - ha[idx_a[q]])) if p in idx_a and q in idx_a else float("inf")
    db = float(np.linalg.norm(hb[idx_b[p]] - hb[idx_b[q]])) if p in idx_b and q in idx_b else float("inf")
    return da, db
