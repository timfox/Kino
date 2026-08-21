"""SAM-style mask decoder stub (Sec. 3.1, Eq. 3)."""

from __future__ import annotations


def resample_heatmap(heatmap_peak: float, *, target_resolution: int = 256) -> float:
    """Three stacked conv blocks — scalar toy for resampled activation mass."""
    return heatmap_peak * (target_resolution / 64.0)


def two_way_cross_attention(feature: float, key: float, *, layers: int = 2) -> float:
    """Bidirectional cross-attention between image keys and heatmap features."""
    state = 0.5 * (feature + key)
    for _ in range(layers):
        state = 0.6 * state + 0.4 * key
    return state


def decode_masks(
    keys: list[float],
    heatmaps: list[float],
    confidences: list[float],
    *,
    resolution: int = 256,
) -> list[float]:
    r"""M̂ = F_dec(K, (H_k, c_k)) — per-slot mask confidence (Eq. 3)."""
    masks: list[float] = []
    for h, c, k in zip(heatmaps, confidences, keys, strict=False):
        feat = resample_heatmap(h * c, target_resolution=resolution)
        masks.append(two_way_cross_attention(feat, k))
    return masks
