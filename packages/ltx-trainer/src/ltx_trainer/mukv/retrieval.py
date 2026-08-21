"""Semi-hierarchical KV-cache retrieval — Sec. 3.4, Eq. (5)–(8)."""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    D = len(vectors[0])
    out = [0.0 for _ in range(D)]
    for v in vectors:
        for d in range(D):
            out[d] += float(v[d])
    n = float(len(vectors))
    return [x / n for x in out]


def stage1_parallel_retrieval(query: list[float], block_keys: list[list[float]], *, k: int) -> list[int]:
    """Return top-k block indices by cosine similarity (Eq. 7 style)."""
    if k <= 0 or not block_keys:
        return []
    sims = [cosine(query, bk) for bk in block_keys]
    order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
    return order[: min(k, len(order))]


def rerank_with_global_query(
    *,
    base_query: list[float],
    cand_indices: list[int],
    cand_keys: list[list[float]],
    global_query: list[float],
    lambda_g: float,
) -> list[int]:
    """e_s = (1-λ_g)s + λ_g γ (Eq. 8)."""
    if not cand_indices:
        return []
    scored: list[tuple[float, int]] = []
    for idx in cand_indices:
        s = cosine(base_query, cand_keys[idx])
        gamma = cosine(global_query, cand_keys[idx])
        s2 = (1.0 - lambda_g) * s + lambda_g * gamma
        scored.append((s2, idx))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _, idx in scored]


def semi_hierarchical_retrieval(
    *,
    q: list[float],
    patch_blocks: list[list[float]],
    frame_blocks: list[list[float]],
    segment_blocks: list[list[float]],
    k2_patch: int,
    k2_frame: int,
    k2_segment: int,
    k_patch: int,
    k_frame: int,
    k_segment: int,
    lambda_g_patch: float,
    lambda_g_frame: float,
) -> dict[str, list[int]]:
    """Two-stage retrieval: parallel top-2k then rerank by segment-derived global query (Sec. 3.4)."""
    seg_stage1 = stage1_parallel_retrieval(q, segment_blocks, k=k2_segment)
    frame_stage1 = stage1_parallel_retrieval(q, frame_blocks, k=k2_frame)
    patch_stage1 = stage1_parallel_retrieval(q, patch_blocks, k=k2_patch)

    # Global query = average of top-N segment blocks from stage1 (paper: derived from segments).
    topN = max(1, min(5, len(seg_stage1)))
    global_q = mean_pool([segment_blocks[i] for i in seg_stage1[:topN]])

    seg_final = seg_stage1[: min(k_segment, len(seg_stage1))]
    frame_reranked = rerank_with_global_query(
        base_query=q,
        cand_indices=frame_stage1,
        cand_keys=frame_blocks,
        global_query=global_q,
        lambda_g=lambda_g_frame,
    )
    patch_reranked = rerank_with_global_query(
        base_query=q,
        cand_indices=patch_stage1,
        cand_keys=patch_blocks,
        global_query=global_q,
        lambda_g=lambda_g_patch,
    )

    return {
        "segment": seg_final,
        "frame": frame_reranked[: min(k_frame, len(frame_reranked))],
        "patch": patch_reranked[: min(k_patch, len(patch_reranked))],
    }

