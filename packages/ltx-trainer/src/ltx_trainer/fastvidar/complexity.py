"""AHA vs full-attention complexity (Eq. 11–13)."""

from __future__ import annotations


def aha_vs_full_ratio(
    num_tokens_per_frame: int,
    num_frames: int,
    window_tokens: int,
) -> float:
    """
    Eq. (13): AHA/Full = P/(SN) + 1/P² with P = window size, N = tokens/frame, S = frames.
    """
    sn = num_frames * num_tokens_per_frame
    p = window_tokens
    if sn <= 0 or p <= 0:
        return 1.0
    return p / sn + 1.0 / (p * p)


def theoretical_speedup(
    num_tokens_per_frame: int = 200,
    num_frames: int = 4,
    window_tokens: int = 49,
) -> dict[str, float]:
    """Paper example: 640×320, 7×7 windows → ~16× at N≈200, S=4."""
    ratio = aha_vs_full_ratio(num_tokens_per_frame, num_frames, window_tokens)
    return {
        "aha_over_full": ratio,
        "speedup_factor": 1.0 / ratio if ratio > 0 else 1.0,
        "num_tokens_per_frame": num_tokens_per_frame,
        "num_frames": num_frames,
        "window_tokens": window_tokens,
    }
