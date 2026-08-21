"""Physics Emergence Zone identification (Sec. 3.2, Eq. 3)."""

from __future__ import annotations

from torch import Tensor


def identify_pez_layers(
    layer_accuracies: dict[int, float] | list[float],
    *,
    epsilon: float = 0.05,
) -> list[int]:
    """PEZ = { l : a_l ≥ max(a) − ε }."""
    if isinstance(layer_accuracies, dict):
        items = sorted(layer_accuracies.items())
        layers = [l for l, _ in items]
        accs = [a for _, a in items]
    else:
        layers = list(range(len(layer_accuracies)))
        accs = list(layer_accuracies)
    peak = max(accs)
    return [layers[i] for i, a in enumerate(accs) if a >= peak - epsilon]


def top_pez_layers(
    layer_accuracies: dict[int, float],
    *,
    epsilon: float = 0.05,
    k: int = 3,
) -> list[int]:
    """Top-k layers by accuracy among PEZ members (paper: L* = {5, 0, 1})."""
    pez = identify_pez_layers(layer_accuracies, epsilon=epsilon)
    ranked = sorted(pez, key=lambda l: layer_accuracies[l], reverse=True)
    return ranked[:k]
