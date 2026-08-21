"""AVBench smoke helpers."""

from __future__ import annotations

import math
from typing import Any


def softmax_pair(yes_logit: float, no_logit: float) -> tuple[float, float]:
    m = max(yes_logit, no_logit)
    ey = math.exp(yes_logit - m)
    en = math.exp(no_logit - m)
    s = ey + en
    return ey / s, en / s


def toy_logits_aligned() -> tuple[float, float]:
    return 2.0, 0.1


def toy_logits_misaligned() -> tuple[float, float]:
    return 0.1, 2.0


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.avbench.scoring import yes_no_alignment_score

    ly, ln = toy_logits_aligned()
    py, pn = softmax_pair(ly, ln)
    my, mn = toy_logits_misaligned()
    pym, pnm = softmax_pair(my, mn)
    return {
        "aligned_score": round(yes_no_alignment_score(py, pn), 4),
        "misaligned_score": round(yes_no_alignment_score(pym, pnm), 4),
        "aligned_prefers_yes": py > pn,
    }
