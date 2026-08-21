"""Toy RCR-GRPO reward components (arXiv:2605.23261)."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

PairLabel = Literal["A", "B"]


def format_reward(output: str, has_thinking: bool = True, has_answer: bool = True) -> float:
    """R_fmt ∈ {-1, 0}: penalize invalid structured output."""
    if not output.strip():
        return -1.0
    thinking_ok = bool(re.search(r"<think>.*?</think>", output, re.S)) if has_thinking else True
    answer_ok = bool(re.search(r"<answer>.*?</answer>", output, re.S)) if has_answer else True
    return 0.0 if (thinking_ok and answer_ok) else -1.0


def accuracy_reward_pairwise(predicted: PairLabel, ground_truth: PairLabel) -> float:
    """R_acc = 1[y == y*] for Tasks 1/3/4."""
    return 1.0 if predicted == ground_truth else 0.0


def accuracy_reward_mos(
    predicted_overall: float,
    ground_truth_overall: float,
    m_min: float = 1.0,
    m_max: float = 5.0,
) -> float:
    """Task 2 normalized distance reward, clamped to [0, 1] (Eq. 9)."""
    denom = max(m_max - m_min, 1e-6)
    raw = 1.0 - abs(predicted_overall - ground_truth_overall) / denom
    return float(np.clip(raw, 0.0, 1.0))


def reasoning_consistent_pairwise(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
) -> float:
    """Dimension-wise sign consistency R_rc for pairwise tasks (Eq. 10)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    ga = np.asarray(gt_a, dtype=np.float64)
    gb = np.asarray(gt_b, dtype=np.float64)
    if not (a.shape == b.shape == ga.shape == gb.shape):
        raise ValueError("score vectors must share shape")
    d = a.shape[0]
    if d == 0:
        return 0.0
    pred_sign = np.sign(a - b)
    gt_sign = np.sign(ga - gb)
    return float(np.mean(pred_sign == gt_sign))


def reasoning_consistent_mos(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    m_min: float = 1.0,
    m_max: float = 5.0,
) -> float:
    """Task 2 aspect-wise normalized reward (Eq. 11)."""
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError("predicted and ground_truth must share shape")
    d = pred.size
    if d == 0:
        return 0.0
    denom = max(m_max - m_min, 1e-6)
    err = np.abs(pred - gt).mean() / denom
    return float(np.clip(1.0 - err, 0.0, 1.0))


def combined_reward(
    r_fmt: float,
    r_acc: float,
    r_rc: float,
    lam_fmt: float = 1.0,
    lam_acc: float = 1.0,
    lam_rc: float = 1.0,
) -> float:
    """R = λ_fmt R_fmt + λ_acc R_acc + λ_rc R_rc (Eq. 7)."""
    return lam_fmt * r_fmt + lam_acc * r_acc + lam_rc * r_rc


def grpo_advantages(rewards: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Group-wise normalized advantages A^(g) = (R^(g) - μ) / (σ + ε) (Eq. 12)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return r
    mu = float(r.mean())
    sigma = float(r.std())
    return (r - mu) / (sigma + epsilon)
