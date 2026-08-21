"""Unified formulation stubs (Eq. 1 and factorizations in Sec. II)."""

from __future__ import annotations


def joint_factorization_exists(has_r: bool) -> bool:
    """Return whether the reasoning-augmented path applies."""
    return bool(has_r)


def audio_to_text_factorization_steps() -> tuple[str, ...]:
    """Stepwise autoregressive view (Eq. 4 style): R then Y."""
    return (
        "Sample/plan reasoning tokens R conditioned on (A, X)",
        "Sample answer tokens Y conditioned on (A, X, R)",
    )


def sequential_audio_to_speech_factorization_steps() -> tuple[str, ...]:
    """Eq. (5): P(R,S|A,X) = P(R|A,X) P(S|R,A,X)."""
    return (
        "Complete internal reasoning R from full audio A",
        "Generate speech tokens S conditioned on R (sequential bottleneck)",
    )
