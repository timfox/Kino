"""Specialized MoE experts: graph, semantic, global (Eq. 3–5)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def graph_expert(h_i: Array, neighbor_hs: Array) -> Array:
    """
    Eq. (3): e_graph = h_i − GCN(h_i, neighbors).

    Toy GCN: mean aggregation of neighbors.
    """
    if neighbor_hs.size == 0:
        agg = np.zeros_like(h_i)
    else:
        agg = neighbor_hs.mean(axis=0)
    return h_i - agg


def semantic_expert(h_i: Array, encoder_w: Array, decoder_w: Array) -> Array:
    """
    Eq. (4): e_semantic = h_i − Decoder(Encoder(h_i)).

    Toy linear encoder/decoder with tied-ish weights.
    """
    hidden = np.tanh(encoder_w @ h_i)
    recon = decoder_w @ hidden
    return h_i - recon


def global_expert(h_i: Array, h_global: Array, mlp_w: Array) -> Array:
    """
    Eq. (5): e_global = h_i − MLP(h_global).

    h_global = mean over all node embeddings.
    """
    proto = mlp_w @ h_global
    return h_i - proto
