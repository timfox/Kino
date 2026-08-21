"""Concept encoder on sparse codebook outputs (Sec. 3.2, Eq. 2)."""

from __future__ import annotations


def encode_concept_vector(sparse_vec: list[float], *, depth: int = 2) -> list[float]:
    """Transformer encoder stack — toy residual mixing."""
    state = list(sparse_vec)
    for layer in range(depth):
        scale = 1.0 / (layer + 1)
        state = [v + scale * (sum(state) / len(state)) for v in state]
    return state


def encode_slot_concepts(
    slots: list[list[float]],
    *,
    depth: int = 2,
) -> list[list[float]]:
    r"""R_k = Encoder(C(h_j)) aggregated per slot (Eq. 2)."""
    return [encode_concept_vector(slot, depth=depth) for slot in slots]
