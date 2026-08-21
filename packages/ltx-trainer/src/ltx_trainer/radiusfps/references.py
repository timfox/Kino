"""Key citation anchors."""

from __future__ import annotations

from typing import Any


def reference_anchors() -> list[dict[str, Any]]:
    return [
        {"id": 4, "cite": "Qi et al. PointNet++", "topic": "FPS in hierarchical point cloud learning"},
        {"id": 23, "cite": "Han et al. QuickFPS", "topic": "KD-tree exact FPS accelerator baseline"},
        {"id": 24, "cite": "Lee et al. FastPoint", "topic": "Learning-based sample distance prediction"},
        {"id": 2, "cite": "Lin et al. PointMetaBase", "topic": "E2E segmentation backbone"},
        {"id": 19, "cite": "S3DIS", "topic": "Indoor semantic parsing benchmark"},
        {"id": 40, "cite": "SemanticKITTI", "topic": "Outdoor LiDAR sequences"},
    ]
