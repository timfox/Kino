"""Predicate-aware re-ranking at inference (arXiv:2606.05812 §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.forte.fol import FolForm, parse_query_fallback, predicate_overlap


@dataclass
class RankedAudio:
    audio_id: str
    caption: str
    embedding: np.ndarray
    phi_hat: FolForm
    sim: float
    s_pred: float
    score: float


def caption_to_fol(caption: str) -> FolForm:
    return parse_query_fallback(caption)


def rerank(
    q_star_emb: np.ndarray,
    candidates: list[tuple[str, str, np.ndarray]],
    phi_star: FolForm,
    *,
    alpha: float = 0.3,
    projected: bool = True,
    project_fn=None,
) -> list[RankedAudio]:
    """Blend cosine similarity with predicate overlap (Eq. 8)."""
    ranked: list[RankedAudio] = []
    for aid, caption, emb in candidates:
        e = project_fn(emb) if project_fn and projected else emb
        sim = float(np.dot(q_star_emb, e) / (np.linalg.norm(q_star_emb) * np.linalg.norm(e) + 1e-8))
        phi_hat = caption_to_fol(caption)
        s_pred = predicate_overlap(phi_star, phi_hat)
        score = (1 - alpha) * sim + alpha * s_pred
        ranked.append(
            RankedAudio(
                audio_id=aid,
                caption=caption,
                embedding=e,
                phi_hat=phi_hat,
                sim=sim,
                s_pred=s_pred,
                score=score,
            )
        )
    ranked.sort(key=lambda r: -r.score)
    return ranked
