"""Continuous scoring helpers for AVBench (Sec. 3.3, 3.4)."""

from __future__ import annotations

import math


def yes_no_alignment_score(p_yes: float, p_no: float, *, eps: float = 1e-12) -> float:
    r"""Normalized SFT evaluator score: ``S = P(\mathrm{Yes}) / (P(\mathrm{Yes}) + P(\mathrm{No}))`` (Sec. 3.3)."""
    py = max(float(p_yes), 0.0)
    pn = max(float(p_no), 0.0)
    return py / (py + pn + eps)


def speech_content_score(
    s_comp: float,
    s_acc: float,
    s_hall: float,
    *,
    w_comp: float = 1.0 / 3.0,
    w_acc: float = 1.0 / 3.0,
    w_hall: float = 1.0 / 3.0,
) -> float:
    """Whisper-based speech content aggregate (Sec. 3.4): keyword completeness, lexical accuracy, hallucination penalty.

    Paper weights components (S_comp, S_acc, S_hall); here equal weights by default for a reference stub.
    ``s_hall`` should be a *penalty* (higher = worse); contribution is ``-w_hall * s_hall`` if caller passes penalty magnitude.
    """
    return w_comp * s_comp + w_acc * s_acc - w_hall * s_hall


def audiobox_aesthetic_score(ce: float, cu: float, pq: float, pc: float) -> float:
    r"""Audiobox-Aesthetics aggregate: ``(CE + CU + PQ - PC) / 4`` (Sec. 3.4, inverse weighting for production complexity)."""
    return (ce + cu + pq - pc) / 4.0


def model_win_ratio(wins: int, ties: int, losses: int) -> float:
    r"""Human or metric win ratio for 2AFC: ``(W + 0.5 T) / (W + T + L)`` (Eq. 1)."""
    w, t, l = int(wins), int(ties), int(losses)
    denom = w + t + l
    if denom == 0:
        return float("nan")
    return (w + 0.5 * t) / denom


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation (Sec. 4.3); small utility for smoke tests."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)
