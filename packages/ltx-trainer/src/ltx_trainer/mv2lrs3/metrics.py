"""WER, IWER, linear-fit and error-profile metrics."""

from __future__ import annotations

from typing import Any


def predict_mv2lrs3_wer(wer_lrs3_pct: float, *, slope: float = 10.4, intercept: float = 8.1) -> float:
    """Eq. (1): linear generalisation sensitivity."""
    return slope * wer_lrs3_pct + intercept


def iwer(substitutions: int, deletions: int, hits: int) -> float:
    """Individual Word Error Rate: (S+D)/(H+S+D)."""
    denom = hits + substitutions + deletions
    if denom == 0:
        return 0.0
    return 100.0 * (substitutions + deletions) / denom


def error_profile(sub_pct: float, del_pct: float, ins_pct: float) -> dict[str, float]:
    return {"substitution": sub_pct, "deletion": del_pct, "insertion": ins_pct}


def modality_delta(av: float, ao: float) -> dict[str, Any]:
    """AV minus AO WER; negative means visual hurts (paper §7)."""
    return {"av_wer": av, "ao_wer": ao, "delta_av_minus_ao": round(av - ao, 2), "visual_hurts": av > ao}
