"""Ego-decoupled MoE layer and context-informed gating (Eq. 2, 6–7)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.camera.experts import global_expert, graph_expert, semantic_expert

Array = np.ndarray


def local_context(h_neighbors: Array) -> Array:
    """Eq. (6): mean neighbor embedding."""
    if h_neighbors.size == 0:
        return np.zeros(1)
    return h_neighbors.mean(axis=0)


def context_informed_gating(h_i: Array, c_i: Array, linear_w: Array) -> Array:
    """
    Eq. (7): g_i = Softmax(Linear([h_i || c_i])) ∈ R^3.

    linear_w shape (3, 2*d) for three experts.
    """
    x = np.concatenate([h_i, c_i], axis=0)
    logits = linear_w @ x
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def ego_decoupled_moe_layer(
    h: Array,
    adj_neighbors: list[list[int]],
    *,
    sem_enc_w: Array,
    sem_dec_w: Array,
    global_mlp_w: Array,
    gate_w: Array,
    all_h: Array,
) -> tuple[Array, Array, dict[str, Array]]:
    """
    Eq. (2): H[l] = H[l-1] + sum_k diag(g_k) * e_k(...).

    Returns updated embedding, gating weights, expert residuals.
    """
    n, d = h.shape
    h_global = h.mean(axis=0)
    out = h.copy()
    gates = np.zeros((n, 3))
    residuals: dict[str, Array] = {"graph": np.zeros_like(h), "semantic": np.zeros_like(h), "global": np.zeros_like(h)}

    for i in range(n):
        neigh_idx = adj_neighbors[i]
        neigh_h = h[neigh_idx] if neigh_idx else np.zeros((0, d))
        e_g = graph_expert(h[i], neigh_h)
        e_s = semantic_expert(h[i], sem_enc_w, sem_dec_w)
        e_gl = global_expert(h[i], h_global, global_mlp_w)
        c_i = local_context(neigh_h) if len(neigh_idx) else np.zeros(d)
        g_i = context_informed_gating(h[i], c_i, gate_w)
        gates[i] = g_i
        residuals["graph"][i] = e_g
        residuals["semantic"][i] = e_s
        residuals["global"][i] = e_gl
        out[i] = h[i] + g_i[0] * e_g + g_i[1] * e_s + g_i[2] * e_gl

    return out, gates, residuals
