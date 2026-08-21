"""Cosine-similarity retrieval stub (Sec. 3.4, FAISS in full system)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ssl_art.classify import cosine_similarity


def top_k_retrieve(
    query_feat: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_ids: list[str],
    *,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k (id, similarity) pairs for query embedding."""
    sims = [cosine_similarity(query_feat, gallery_feats[i]) for i in range(len(gallery_ids))]
    order = np.argsort(sims)[::-1][:k]
    return [(gallery_ids[int(i)], float(sims[int(i)])) for i in order]
