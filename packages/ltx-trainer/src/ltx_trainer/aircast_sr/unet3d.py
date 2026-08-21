"""Toy 3D U-Net denoiser matching AirCast-SR channel layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.aircast_sr.conv3d_np import conv3d, group_norm, init_conv_weight, silu
from ltx_trainer.aircast_sr.config import AirCastSRConfig


@dataclass
class UNet3DWeights:
    in_conv: np.ndarray
    in_bias: np.ndarray
    down_convs: list[np.ndarray]
    down_biases: list[np.ndarray]
    mid_conv: np.ndarray
    mid_bias: np.ndarray
    up_convs: list[np.ndarray]
    up_biases: list[np.ndarray]
    out_conv: np.ndarray
    out_bias: np.ndarray


def build_toy_unet3d(cfg: AirCastSRConfig, *, seed: int = 0, smoke: bool = True) -> UNet3DWeights:
    blocks = cfg.smoke_block_channels if smoke else cfg.block_channels
    c_in = cfg.denoiser_in_channels
    c_out = cfg.denoiser_out_channels
    rng = np.random.default_rng(seed)
    ch0 = blocks[0]
    in_conv = init_conv_weight(ch0, c_in, 3, seed)
    in_bias = rng.standard_normal(ch0) * 0.01
    down_convs, down_biases = [], []
    prev = ch0
    for i, ch in enumerate(blocks[1:]):
        w = init_conv_weight(ch, prev, 3, seed + 10 + i)
        down_convs.append(w)
        down_biases.append(rng.standard_normal(ch) * 0.01)
        prev = ch
    mid_conv = init_conv_weight(prev, prev, 3, seed + 99)
    mid_bias = rng.standard_normal(prev) * 0.01
    up_convs, up_biases = [], []
    for i, ch in enumerate(reversed(blocks[:-1])):
        w = init_conv_weight(ch, prev + ch, 3, seed + 50 + i)
        up_convs.append(w)
        up_biases.append(rng.standard_normal(ch) * 0.01)
        prev = ch
    out_conv = init_conv_weight(c_out, prev, 1, seed + 200)
    out_bias = rng.standard_normal(c_out) * 0.01
    return UNet3DWeights(
        in_conv=in_conv,
        in_bias=in_bias,
        down_convs=down_convs,
        down_biases=down_biases,
        mid_conv=mid_conv,
        mid_bias=mid_bias,
        up_convs=up_convs,
        up_biases=up_biases,
        out_conv=out_conv,
        out_bias=out_bias,
    )


def _pool3d(x: np.ndarray) -> np.ndarray:
    c, d, h, w = x.shape
    d2, h2, w2 = d - d % 2, h - h % 2, w - w % 2
    x = x[:, :d2, :h2, :w2]
    return 0.125 * (
        x[:, 0::2, 0::2, 0::2]
        + x[:, 1::2, 0::2, 0::2]
        + x[:, 0::2, 1::2, 0::2]
        + x[:, 1::2, 1::2, 0::2]
        + x[:, 0::2, 0::2, 1::2]
        + x[:, 1::2, 0::2, 1::2]
        + x[:, 0::2, 1::2, 1::2]
        + x[:, 1::2, 1::2, 1::2]
    )


def _upsample3d(x: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    c = x.shape[0]
    d, h, w = target_shape
    out = np.zeros((c, d, h, w), dtype=np.float64)
    sd, sh, sw = x.shape[1:]
    for di in range(d):
        for hi in range(h):
            for wi in range(w):
                out[:, di, hi, wi] = x[:, min(di * sd // d, sd - 1), min(hi * sh // h, sh - 1), min(wi * sw // w, sw - 1)]
    return out


def unet3d_forward(x: np.ndarray, weights: UNet3DWeights, *, groups: int = 8) -> np.ndarray:
    """Predict noise ε for target channels. x: (C_in, D, H, W)."""
    h = conv3d(x, weights.in_conv, weights.in_bias, padding=1)
    h = silu(group_norm(h, groups=groups))
    skips = [h]
    for w, b in zip(weights.down_convs, weights.down_biases):
        if min(h.shape[1:]) < 4:
            break
        h = conv3d(h, w, b, padding=1)
        h = silu(group_norm(h, groups=groups))
        skips.append(h)
        h = _pool3d(h)
    h = conv3d(h, weights.mid_conv, weights.mid_bias, padding=1)
    h = silu(group_norm(h, groups=groups))
    for w, b, skip in zip(weights.up_convs, weights.up_biases, reversed(skips[:-1])):
        h = _upsample3d(h, skip.shape[1:])
        h = np.concatenate([h, skip], axis=0)
        h = conv3d(h, w, b, padding=1)
        h = silu(group_norm(h, groups=groups))
    if h.shape != skips[0].shape:
        h = _upsample3d(h, skips[0].shape[1:])
    return conv3d(h, weights.out_conv, weights.out_bias, padding=0)
