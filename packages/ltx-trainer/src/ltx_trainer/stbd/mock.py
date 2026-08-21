"""Toy array geometry and likelihood comparison for subspace TBD smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.stbd.likelihood import (
    aggregate_bingham_log_likelihood,
    aggregate_deterministic_log_likelihood,
    bingham_log_likelihood,
    snr_to_noise_variance,
)
from ltx_trainer.stbd.steering import mixing_matrix, normalize_observation, subspace_projector


def ring_microphones(num_mics: int, room_size: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_mics, endpoint=False)
    r = room_size / 2.0
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def simulate_normalized_mixture(
    mic: np.ndarray,
    positions: list[np.ndarray],
    frequency_hz: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    h = mixing_matrix(mic, positions, frequency_hz)
    s = rng.standard_normal(h.shape[1]) + 1j * rng.standard_normal(h.shape[1])
    ez = h @ s
    z = normalize_observation(ez)
    return [z] if z is not None else []


def compare_likelihoods_at_true_state(
    mic: np.ndarray,
    positions: list[np.ndarray],
    frequency_hz: float,
    rng: np.random.Generator,
    *,
    snr_db: float = -10.0,
) -> dict[str, float | bool]:
    h_true = mixing_matrix(mic, positions, frequency_hz)
    p_true = subspace_projector(h_true)
    wrong_pos = [positions[0] + np.array([5.0, 5.0])]
    h_wrong = mixing_matrix(mic, wrong_pos, frequency_hz)
    p_wrong = subspace_projector(h_wrong)

    s = rng.standard_normal(h_true.shape[1]) + 1j * rng.standard_normal(h_true.shape[1])
    ez = h_true @ s
    z = normalize_observation(ez)
    if z is None:
        return {"subspace_prefers_true": False, "subspace_ll": 0.0, "conventional_ll": 0.0}

    ll_sub_true = bingham_log_likelihood(z, p_true)
    ll_sub_wrong = bingham_log_likelihood(z, p_wrong)
    sigma = snr_to_noise_variance(float(np.linalg.norm(ez) ** 2), snr_db)
    pred_true = h_true @ s
    pred_wrong = h_wrong @ s[: h_wrong.shape[1]]
    ll_det_true = aggregate_deterministic_log_likelihood([ez], [pred_true], sigma)
    ll_det_wrong = aggregate_deterministic_log_likelihood([ez], [pred_wrong], sigma)

    return {
        "subspace_prefers_true": ll_sub_true > ll_sub_wrong,
        "subspace_ll": float(ll_sub_true),
        "conventional_prefers_true": ll_det_true > ll_det_wrong,
        "aggregate_subspace": float(aggregate_bingham_log_likelihood([z], [p_true])),
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mic = ring_microphones(8, room_size=10.0)
    positions = [np.array([2.0, 3.0]), np.array([7.0, 6.0])]
    stats = compare_likelihoods_at_true_state(mic, positions, frequency_hz=1000.0, rng=rng)
    return {
        k: (bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 4))
        for k, v in stats.items()
    }
