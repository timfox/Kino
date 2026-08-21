"""Spatial Video-Audio Contrastive Learning (Sec. 3.2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def info_nce_loss(
    query: np.ndarray,
    positive: np.ndarray,
    negatives: Sequence[np.ndarray],
    *,
    temperature: float = 0.07,
) -> float:
    """Eq. 2 — symmetric contrastive term for one direction."""
    pos = cosine_similarity(query, positive) / temperature
    neg_terms = [cosine_similarity(query, n) / temperature for n in negatives]
    logits = np.array([pos, *neg_terms], dtype=np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return float(-np.log(exp[0] / exp.sum()))


def build_negative_sets(
    batch_size: int,
    index: int,
    *,
    include_temporal: bool = True,
    include_audio_rot: bool = True,
    include_video_rot: bool = True,
) -> tuple[list[int], list[str]]:
    """Eq. 3 — semantic batch negatives + optional hard-negative tags."""
    semantic = [j for j in range(batch_size) if j != index]
    tags: list[str] = ["semantic"] * len(semantic)
    if include_temporal:
        tags.append("temporal")
    if include_audio_rot:
        tags.append("audio_rotation")
    if include_video_rot:
        tags.append("video_rotation")
    return semantic, tags


def svac_batch_loss(
    video_feats: list[np.ndarray],
    audio_feats: list[np.ndarray],
    *,
    temperature: float = 0.07,
    full_physics: bool = True,
) -> float:
    """Eq. 1 — mean symmetric InfoNCE over batch (toy features)."""
    n = len(video_feats)
    total = 0.0
    count = 0
    for i in range(n):
        neg_a, _ = build_negative_sets(
            n,
            i,
            include_temporal=full_physics,
            include_audio_rot=full_physics,
            include_video_rot=full_physics,
        )
        neg_v, _ = build_negative_sets(
            n,
            i,
            include_temporal=full_physics,
            include_audio_rot=full_physics,
            include_video_rot=full_physics,
        )
        negatives_a = [audio_feats[j] for j in neg_a]
        negatives_v = [video_feats[j] for j in neg_v]
        if full_physics:
            negatives_a.append(-audio_feats[i])
            negatives_v.append(-video_feats[i])
        total += info_nce_loss(video_feats[i], audio_feats[i], negatives_a, temperature=temperature)
        total += info_nce_loss(audio_feats[i], video_feats[i], negatives_v, temperature=temperature)
        count += 2
    return total / max(count, 1)
