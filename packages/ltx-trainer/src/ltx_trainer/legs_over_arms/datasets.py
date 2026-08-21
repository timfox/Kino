"""JRDB and social-navigation dataset cards (Sec. III-B)."""

from __future__ import annotations

from typing import Any

JRDB_SCENES = "large-scale mobile robot, 360° stereo, pedestrian zones"
SOCIAL_NAV_HOURS = 3.0
CAMERA_HEIGHT_M = 0.85


def jrdb_card() -> dict[str, Any]:
    return {
        "name": "JRDB",
        "modalities": ["LiDAR", "360° stereo RGB"],
        "pose": {
            "2D": "JRDB-Pose COCO-17",
            "3D": "HST 33-keypoint (GHUM fitting)",
        },
        "notes": "Limited vertical FOV often crops legs on nearby pedestrians",
    }


def social_nav_card() -> dict[str, Any]:
    return {
        "name": "GMU Social Navigation (forthcoming)",
        "platform": "AgileX Scout Mini + Insta360 X4 ERP",
        "duration_hours": SOCIAL_NAV_HOURS,
        "pose_pipeline": "HRNet + MMPose → Bacchin ERP→robot tracks",
        "camera_height_m": CAMERA_HEIGHT_M,
        "notes": "Dense indoor crowds; full vertical FOV for leg keypoints",
    }


def datasets_bundle() -> dict[str, Any]:
    return {"jrdb": jrdb_card(), "social_navigation": social_nav_card()}
