"""Timestamp label smoothing and training utilities (Sec. 3.3)."""

from __future__ import annotations

from ltx_trainer.rubato.config import RubatoConfig


def timestamp_smoothing_distribution(
    target_bin: int,
    *,
    w: int | None = None,
    p_center: float | None = None,
    cfg: RubatoConfig | None = None,
) -> dict[int, float]:
    r"""Ordinal label smoothing over timestamp bins (Sec. 3.3).

    ``P_center`` on true bin ``y``; remaining ``1 - P_center`` spread over ``i ∈ N_y`` with
    quadratic weights ``(w + 1 - |i-y|)^2``, normalized by ``Z_y``.
    """
    cfg = cfg or RubatoConfig()
    w = w if w is not None else cfg.label_smoothing_window_bins
    p_c = p_center if p_center is not None else cfg.label_smoothing_p_center
    ny = [i for i in range(target_bin - w, target_bin + w + 1) if i != target_bin and abs(i - target_bin) <= w]
    weights: dict[int, float] = {}
    quad_sum = 0.0
    for i in ny:
        q = (w + 1 - abs(i - target_bin)) ** 2
        weights[i] = q
        quad_sum += q
    if quad_sum < 1e-12:
        return {target_bin: 1.0}
    rem = 1.0 - p_c
    z = rem / quad_sum
    dist = {target_bin: p_c}
    for i in ny:
        dist[i] = z * weights[i]
    # normalize drift from float
    s = sum(dist.values())
    return {k: v / s for k, v in dist.items()}


def inverse_sequence_length_weight(seq_len: int) -> float:
    r"""Loss scale ``1/|T|`` for multitask token weighting (Sec. 3.3)."""
    if seq_len <= 0:
        return 0.0
    return 1.0 / float(seq_len)
