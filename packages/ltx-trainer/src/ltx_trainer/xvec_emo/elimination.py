"""Four-step elimination study (Sec. 3.1, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EliminationStep:
    step: int
    operand: str
    intervention: str
    outcome: str
    supports_arithmetic: bool


ELIMINATION_STEPS: tuple[EliminationStep, ...] = (
    EliminationStep(
        1,
        "weights (backbone)",
        "FT / LoRA task vectors",
        "noise or calm speech; no controllable emotion",
        False,
    ),
    EliminationStep(
        2,
        "codec embeddings",
        "per-codebook centroid τ_k",
        "no effect → degenerate noise; no emotional regime",
        False,
    ),
    EliminationStep(
        3,
        "discrete tokens",
        "full_swap (angry tokens + neutral x-vector)",
        "calm coherent output; LM follows x-vector",
        False,
    ),
    EliminationStep(
        4,
        "x-vector (ECAPA)",
        "centroid τ_emo (Eq. 1)",
        "dominant emotion carrier; training-free control",
        True,
    ),
)


def localized_operand() -> str:
    return next(s.operand for s in ELIMINATION_STEPS if s.supports_arithmetic)
