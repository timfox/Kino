"""Highlight alignment: PSNR matching + temporal post-processing (Eq. 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ltx_trainer.svhighlights.config import SvHighlightsConfig


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    o = a.astype(np.float64)
    r = b.astype(np.float64)
    if o.shape != r.shape:
        raise ValueError(f"shape mismatch: {o.shape} vs {r.shape}")
    mse = float(np.mean((o - r) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def downsample_frame(frame: np.ndarray, size: int = 144) -> np.ndarray:
    """Nearest-neighbor downsample to square `size` (144p proxy)."""
    h, w = frame.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(np.int32)
    xs = (np.linspace(0, w - 1, size)).astype(np.int32)
    return frame[np.ix_(ys, xs)]


def best_match_frame(highlight: np.ndarray, full_frames: Sequence[np.ndarray]) -> tuple[int, float]:
    best_i, best_score = 0, -1.0
    for i, f in enumerate(full_frames):
        score = psnr(highlight, f)
        if score > best_score:
            best_i, best_score = i, score
    return best_i, best_score


def align_highlight_sequence(
    highlight_frames: Sequence[np.ndarray],
    full_frames: Sequence[np.ndarray],
    *,
    fps: float = 30.0,
    tau: float = 5.0,
    psnr_threshold: float = 20.0,
) -> list[tuple[int | None, float]]:
    """Align each highlight frame to full video; return (frame_idx, psnr) or (None, score) if filtered."""
    if not highlight_frames:
        return []
    rate = max(int(round(fps)), 1)
    aligned: list[tuple[int | None, float]] = []
    prev_pos = 0
    for h in highlight_frames:
        f_star, psnr_star = best_match_frame(h, full_frames)
        expected = min(prev_pos + rate, len(full_frames) - 1)
        f_plus = expected
        psnr_plus = psnr(h, full_frames[f_plus])
        if psnr_star - psnr_plus <= tau:
            chosen = f_plus
            score = psnr_plus
        else:
            chosen = f_star
            score = psnr_star
        if score < psnr_threshold:
            aligned.append((None, score))
        else:
            aligned.append((chosen, score))
            prev_pos = chosen
    return aligned


@dataclass
class AlignmentStats:
    n_total: int
    n_retained: int
    mean_psnr: float
    f_star_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_total": self.n_total,
            "n_retained": self.n_retained,
            "remain_rate_pct": round(100.0 * self.n_retained / max(self.n_total, 1), 2),
            "mean_psnr": round(self.mean_psnr, 3),
            "f_star_ratio": round(self.f_star_ratio, 3),
        }


def alignment_stats(
    highlight_frames: Sequence[np.ndarray],
    full_frames: Sequence[np.ndarray],
    cfg: SvHighlightsConfig | None = None,
) -> AlignmentStats:
    c = cfg or SvHighlightsConfig()
    aligned = align_highlight_sequence(
        highlight_frames,
        full_frames,
        tau=c.alignment_tau,
        psnr_threshold=c.psnr_filter_threshold,
    )
    retained = [(i, s) for i, s in aligned if i is not None]
    n_star = 0
    rate = 30
    prev = 0
    for h in highlight_frames:
        f_star, psnr_star = best_match_frame(h, full_frames)
        expected = min(prev + rate, len(full_frames) - 1)
        psnr_plus = psnr(h, full_frames[expected])
        if psnr_star - psnr_plus > c.alignment_tau:
            n_star += 1
        prev = f_star if psnr_star - psnr_plus > c.alignment_tau else expected
    scores = [s for _, s in retained]
    return AlignmentStats(
        n_total=len(aligned),
        n_retained=len(retained),
        mean_psnr=float(np.mean(scores)) if scores else 0.0,
        f_star_ratio=n_star / max(len(highlight_frames), 1),
    )
