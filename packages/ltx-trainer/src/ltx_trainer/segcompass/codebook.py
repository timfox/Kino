"""Query codebook decoding from sparse SAE features (Sec. 3.2)."""

from __future__ import annotations


def decode_sparse_pairs(
    indices: list[int],
    activations: list[float],
    *,
    concept_dim: int = 256,
) -> list[tuple[int, float, list[float]]]:
    r"""Map {(j, h_j)} to dense concept vectors C(h_j) — toy hash projection."""
    out: list[tuple[int, float, list[float]]] = []
    for j, h in zip(indices, activations, strict=True):
        vec = [0.0] * concept_dim
        vec[j % concept_dim] = h
        if concept_dim > 1:
            vec[(j + 1) % concept_dim] = 0.5 * h
        out.append((j, h, vec))
    return out


def aggregate_concept_slots(
    decoded: list[tuple[int, float, list[float]]],
    num_slots: int,
) -> list[list[float]]:
    """Round-robin aggregate sparse concepts into Ks slot representations."""
    if num_slots <= 0:
        return []
    dim = len(decoded[0][2]) if decoded else 256
    slots = [[0.0] * dim for _ in range(num_slots)]
    counts = [0] * num_slots
    for i, (_, h, vec) in enumerate(decoded):
        k = i % num_slots
        for d, v in enumerate(vec):
            slots[k][d] += h * v
        counts[k] += 1
    for k in range(num_slots):
        if counts[k] > 0:
            slots[k] = [v / counts[k] for v in slots[k]]
    return slots
