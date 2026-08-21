"""Per-modality cosine scores and late fusion (arXiv:2606.05931)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def max_pool_file_score(
    query_emb: np.ndarray,
    archive_embs: list[np.ndarray],
) -> float:
    """Eq. (1)/(2): max cosine over identities in one archive file."""
    if not archive_embs:
        return 0.0
    return max(cosine_similarity(query_emb, e) for e in archive_embs)


def fused_score(
    speaker_score: float,
    face_score: float,
    *,
    lam: float,
) -> float:
    """Eq. (3): sMM = λ sspk + (1−λ) sface."""
    return lam * speaker_score + (1.0 - lam) * face_score


def lambda_for_presence(presence: str, *, avp_lam: float = 0.5) -> float:
    """§3.1 optimal λ by presence type."""
    norm = presence.strip().upper()
    if norm in {"AOP", "AO", "AUDIO_ONLY"}:
        return 1.0
    if norm in {"VOP", "VO", "VISUAL_ONLY"}:
        return 0.0
    return avp_lam
