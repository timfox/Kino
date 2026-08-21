"""Standard and Utility-Aware InfoNCE losses (Eq. 1, 4)."""

from __future__ import annotations

import numpy as np


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def infonce_text_to_image(
    scores: np.ndarray,
    *,
    temperature: float = 0.07,
) -> float:
    """
    Lt→v: rows are texts, columns are images; diagonal are positives.

    scores[i,k] = similarity between image k and text i.
    """
    s = np.asarray(scores, dtype=np.float64) / temperature
    n = s.shape[0]
    log_probs = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return float(-np.mean(np.diag(log_probs)))


def infonce_image_to_text(
    scores: np.ndarray,
    *,
    temperature: float = 0.07,
) -> float:
    """Lv→t: transpose of text-to-image."""
    return infonce_text_to_image(scores.T, temperature=temperature)


def bidirectional_infonce(
    scores: np.ndarray,
    *,
    temperature: float = 0.07,
    utility_aware: bool = False,
) -> dict[str, float]:
    """Average of text→image and image→text InfoNCE."""
    lt = infonce_text_to_image(scores, temperature=temperature)
    lv = infonce_image_to_text(scores, temperature=temperature)
    loss = 0.5 * (lt + lv)
    return {
        "loss": loss,
        "text_to_image": lt,
        "image_to_text": lv,
        "utility_aware": utility_aware,
    }


def build_score_matrix(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    visual_utilities: np.ndarray,
    *,
    alpha_visual: float,
    beta_semantic: float,
    utility_aware: bool,
) -> np.ndarray:
    """N×N matrix for matched batch {(v_i, t_i)}."""
    from ltx_trainer.uaclip.similarity import clip_similarity, utility_aware_score

    n = image_embs.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            s = clip_similarity(image_embs[k], text_embs[i])
            if utility_aware:
                out[i, k] = utility_aware_score(
                    s,
                    visual_utilities[k],
                    alpha_visual=alpha_visual,
                    beta_semantic=beta_semantic,
                )
            else:
                out[i, k] = s
    return out
