"""DLM parallel decoding strategies (Sec. 3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dlmasr.config import DecodingStrategy, DlmAsrConfig


def token_confidence(logits: np.ndarray) -> tuple[int, float]:
    """Eq. 2 proxy — argmax token and confidence c(i) = p(v|·)."""
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum() + 1e-12
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def fixed_number_commit(
    masked_indices: list[int],
    confidences: dict[int, float],
    k: int,
) -> list[int]:
    """Top-k most confident masked positions (fixed-number scheme)."""
    if not masked_indices:
        return []
    ranked = sorted(masked_indices, key=lambda i: confidences.get(i, 0.0), reverse=True)
    return ranked[: min(k, len(ranked))]


def static_threshold_commit(
    masked_indices: list[int],
    confidences: dict[int, float],
    threshold: float,
) -> list[int]:
    """Static confidence threshold C; fallback to single most confident token."""
    if not masked_indices:
        return []
    above = [i for i in masked_indices if confidences.get(i, 0.0) >= threshold]
    if above:
        return above
    best = max(masked_indices, key=lambda i: confidences.get(i, 0.0))
    return [best]


def dynamic_threshold_commit(
    masked_indices: list[int],
    confidences: dict[int, float],
    factor: float,
) -> list[int]:
    """Dynamic threshold — largest k with (k+1)(1 - c(k)) < f (Eq. 3)."""
    if not masked_indices:
        return []
    ranked = sorted(masked_indices, key=lambda i: confidences.get(i, 0.0), reverse=True)
    scores = [confidences[i] for i in ranked]
    best_k = 0
    for k, c_k in enumerate(scores, start=1):
        if (k + 1) * (1.0 - c_k) < factor:
            best_k = k
    if best_k == 0:
        return [ranked[0]]
    return ranked[:best_k]


def decode_round(
    masked: np.ndarray,
    logits_by_pos: dict[int, np.ndarray],
    *,
    strategy: DecodingStrategy,
    k: int = 8,
    threshold: float = 0.95,
    factor: float = 0.2,
) -> dict[str, Any]:
    """Single diffusion unmasking round over masked positions."""
    masked_indices = [int(i) for i in np.flatnonzero(masked)]
    confidences: dict[int, float] = {}
    for i in masked_indices:
        _, conf = token_confidence(logits_by_pos[i])
        confidences[i] = conf

    if strategy == DecodingStrategy.FIXED_NUMBER:
        commit = fixed_number_commit(masked_indices, confidences, k)
    elif strategy == DecodingStrategy.STATIC_THRESHOLD:
        commit = static_threshold_commit(masked_indices, confidences, threshold)
    elif strategy == DecodingStrategy.DYNAMIC_THRESHOLD:
        commit = dynamic_threshold_commit(masked_indices, confidences, factor)
    else:
        commit = [masked_indices[0]] if masked_indices else []

    return {
        "committed_indices": commit,
        "num_committed": len(commit),
        "confidences": {i: confidences[i] for i in commit},
    }


def block_decode_smoke(
    seq_len: int = 32,
    *,
    strategy: DecodingStrategy = DecodingStrategy.STATIC_THRESHOLD,
    cfg: DlmAsrConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Toy block diffusion decode with skewed ASR-like confidences."""
    cfg = cfg or DlmAsrConfig()
    rng = np.random.default_rng(seed)
    masked = np.ones(seq_len, dtype=bool)
    rounds = 0
    total_committed = 0
    # ASR-like skew: most tokens highly confident (paper Fig. 4: 91% >= 0.95)
    position_confidence = {i: float(rng.beta(25, 1)) for i in range(seq_len)}

    while masked.any() and rounds < seq_len * 4:
        masked_indices = [int(i) for i in np.flatnonzero(masked)]
        confidences = {i: position_confidence[i] for i in masked_indices}

        if strategy == DecodingStrategy.FIXED_NUMBER:
            commit = fixed_number_commit(masked_indices, confidences, k=1)
        elif strategy == DecodingStrategy.STATIC_THRESHOLD:
            commit = static_threshold_commit(masked_indices, confidences, threshold=0.95)
        elif strategy == DecodingStrategy.DYNAMIC_THRESHOLD:
            commit = dynamic_threshold_commit(masked_indices, confidences, factor=0.2)
        else:
            commit = [masked_indices[0]] if masked_indices else []

        for idx in commit:
            masked[idx] = False
        total_committed += len(commit)
        rounds += 1

    return {
        "strategy": strategy.value,
        "rounds": rounds,
        "seq_len": seq_len,
        "total_committed": total_committed,
    }
