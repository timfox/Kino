"""Skeletal feature configurations G(s) (Sec. III-A, Fig. 2)."""

from __future__ import annotations

from typing import Literal

FeatureConfig = Literal[
    "baseline",
    "K3D",
    "K3D_C3D",
    "K3D_U",
    "K3D_U_C3D_U",
    "K3D_L",
    "K3D_L_C3D_L",
    "K2D",
    "K2D_L",
]

# MediaPipe 33-keypoint indices (lower / upper subsets — illustrative stub indices)
LOWER_BODY_3D_IDX: tuple[int, ...] = tuple(range(23, 33))  # 10 lower-body
UPPER_BODY_3D_IDX: tuple[int, ...] = tuple(range(11, 21))  # 10 upper-body
# COCO 17: hips=11,12; knees=13,14; ankles=15,16
LOWER_BODY_2D_IDX: tuple[int, ...] = (11, 12, 13, 14, 15, 16)


def keypoint_dim(config: FeatureConfig) -> int:
    if config == "baseline":
        return 0
    if config in ("K3D", "K3D_C3D"):
        return 33 * 3
    if config in ("K3D_L", "K3D_L_C3D_L"):
        return 10 * 3
    if config in ("K3D_U", "K3D_U_C3D_U"):
        return 10 * 3
    if config == "K2D":
        return 17 * 2
    if config == "K2D_L":
        return 6 * 2
    return 0


def biomech_dim(config: FeatureConfig) -> int:
    if config in ("K3D_C3D",):
        return 12  # C3D_L + C3D_U compact
    if config == "K3D_L_C3D_L":
        return 6  # leg angles + step length
    if config == "K3D_U_C3D_U":
        return 6  # arm angles + head orientation
    return 0


def total_feature_dim(config: FeatureConfig) -> int:
    return keypoint_dim(config) + biomech_dim(config)
