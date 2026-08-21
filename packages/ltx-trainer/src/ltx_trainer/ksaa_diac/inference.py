"""MC Dropout ensemble inference — §3.3."""

from __future__ import annotations

import numpy as np

from ltx_trainer.ksaa_diac.losses import softmax


def stochastic_forward(
    logits_fn,
    rng: np.random.Generator,
    dropout_p: float = 0.1,
) -> np.ndarray:
    """Single MC pass with dropout active in CATT encoder."""
    logits = logits_fn(rng, dropout_p)
    return softmax(logits)


def mc_dropout_average(
    logits_fn,
    num_passes: int,
    rng: np.random.Generator,
    dropout_p: float = 0.1,
) -> np.ndarray:
    """Average softmax over num_passes stochastic forwards."""
    acc = None
    for _ in range(num_passes):
        p = stochastic_forward(logits_fn, rng, dropout_p)
        acc = p if acc is None else acc + p
    assert acc is not None
    return acc / num_passes


def ensemble_softmax_average(
    checkpoint_logits_fns: list,
    passes_per_model: int,
    rng: np.random.Generator,
    dropout_p: float = 0.1,
) -> np.ndarray:
    """4 checkpoints × 50 passes = 200 averaged softmax distributions."""
    acc = None
    total = 0
    for fn in checkpoint_logits_fns:
        avg = mc_dropout_average(fn, passes_per_model, rng, dropout_p)
        acc = avg if acc is None else acc + avg
        total += 1
    assert acc is not None
    return acc / total


def predict_diacritics(avg_probs: np.ndarray) -> np.ndarray:
    return np.argmax(avg_probs, axis=-1)
