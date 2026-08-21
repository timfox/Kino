"""Decoupling Spatio-Temporal (DST) module within DSTA (Sec. IV-C, Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChannelSplit:
    """Split C channels into αC (spatial-temporal) and (1-α)C (identity)."""

    spatial_temporal_channels: int
    identity_channels: int


def split_channels(total_channels: int, *, alpha: float) -> ChannelSplit:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    st = int(round(total_channels * alpha))
    st = max(1, min(total_channels - 1, st))
    return ChannelSplit(spatial_temporal_channels=st, identity_channels=total_channels - st)


def dst_spatial_temporal_sum(
    temporal: float,
    height: float,
    width: float,
) -> float:
    r"""x̂_11 = Conv_{3×1×1} + Conv_{1×3×1} + Conv_{1×1×3} (Eq. 2a), toy scalar."""
    return temporal + height + width


def dst_forward(
    dwconv_out: float,
    conv_t: float,
    conv_h: float,
    conv_w: float,
    identity_skip: float,
) -> float:
    r"""Full DST: concat fused spatial-temporal branch + DWConv stream (Eq. 2b–2d)."""
    x11 = dst_spatial_temporal_sum(conv_t, conv_h, conv_w)
    x1 = x11 + identity_skip
    return x1 + dwconv_out
