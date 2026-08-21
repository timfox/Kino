"""Factorized conditioning: content, melody, timbre, task (Sec 3.3–3.4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TaskModality(str, Enum):
    SPEECH = "speech"
    SINGING = "singing"


@dataclass
class FactorizedCondition:
    content: np.ndarray
    melody: np.ndarray
    timbre: np.ndarray
    task: TaskModality


def null_melody_token(dim: int, learned: np.ndarray | None = None) -> np.ndarray:
    """Learned null token e_∅ for speech (broadcast over time)."""
    if learned is not None:
        return learned.copy()
    return np.zeros(dim, dtype=np.float64)


def melody_from_midi(midi_len: int, dim: int, seed: int = 0) -> np.ndarray:
    """Stub Conformer MIDI encoding c_mel ∈ R^{T×D}."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((midi_len, dim))


def concat_conditions(
    noisy_latent: np.ndarray,
    content: np.ndarray,
    melody: np.ndarray,
    timbre: np.ndarray,
) -> np.ndarray:
    """Channel-wise concat [x_t | c_cnt | c_mel | c_tmb] before linear projection."""
    t = noisy_latent.shape[0]
    mel = np.broadcast_to(melody.reshape(-1, melody.shape[-1]), (t, melody.shape[-1]))
    tmb = timbre[:t] if timbre.shape[0] >= t else np.pad(timbre, ((0, t - timbre.shape[0]), (0, 0)))
    return np.concatenate([noisy_latent, content[:t], mel, tmb], axis=-1)


def build_condition(
    *,
    dim: int,
    seq_len: int,
    task: TaskModality,
    null_token: np.ndarray | None = None,
    seed: int = 0,
) -> FactorizedCondition:
    rng = np.random.default_rng(seed)
    content = rng.standard_normal((seq_len, dim))
    timbre = rng.standard_normal((seq_len, dim))
    if task == TaskModality.SPEECH:
        melody = null_melody_token(dim, null_token)
    else:
        melody = melody_from_midi(seq_len, dim, seed=seed + 1)
    return FactorizedCondition(content=content, melody=melody, timbre=timbre, task=task)


def axis_guidance_delta(
    full_v: np.ndarray,
    dropped_v: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Axis-selective classifier-free guidance (Eq. 7)."""
    return full_v + weight * (full_v - dropped_v)
