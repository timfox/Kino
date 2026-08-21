"""CAMERA graph expert smoke."""

from __future__ import annotations

from typing import Any

import numpy as np


def toy_tag_graph(n: int, d: int, *, seed: int = 0) -> tuple[np.ndarray, list[list[int]]]:
    """Node embeddings ``[n, d]`` and adjacency as neighbor index lists."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, d))
    neighbors = [[(i + 1) % n, (i + 2) % n] for i in range(n)]
    return h, neighbors


def init_expert_weights(d: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "sem_enc": rng.standard_normal((d, d)) * 0.1,
        "sem_dec": rng.standard_normal((d, d)) * 0.1,
        "global_mlp": rng.standard_normal((d, d)) * 0.1,
        "gate_w": rng.standard_normal((3, 2 * d)) * 0.01,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.camera.experts import graph_expert

    h, neighbors = toy_tag_graph(16, 8, seed=seed)
    out = graph_expert(h[0], h[neighbors[0]])
    return {"num_nodes": int(h.shape[0]), "expert_out_dim": int(out.shape[0])}
