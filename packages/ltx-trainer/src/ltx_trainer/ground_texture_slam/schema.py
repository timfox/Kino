"""Poses, observations, and loop-closure methods (Sec. III–IV)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LoopClosureMethod(str, Enum):
    ORIGINAL_SINGLE = "original_single"
    ORIGINAL_MANY = "original_many"
    KLD = "kld"
    KLD_GRAY = "kld_gray"
    VISUAL_OVERLAP = "visual_overlap"
    JIH = "jih"
    ODOMETRY = "odometry"


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0
    session: int = 0
    t: int = 0


@dataclass
class Observation:
    image_id: str
    session: int
    t: int
    pose_gt: Pose2D | None = None


DATASET_SESSIONS = 5
DATASET_IMAGES_PER_SESSION = (225, 249)
IMAGE_SIZE = (711, 1266)  # H x W per paper
CAMERA_HEIGHT_M = 0.72
