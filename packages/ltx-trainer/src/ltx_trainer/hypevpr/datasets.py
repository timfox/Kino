"""Dataset / benchmark cards (Sec. 5)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "Pitts250K-P2E": {
            "query": "perspective",
            "database": "equirectangular",
            "mining": "GPS KNN, 10 m positive radius, 1 hard pos + 10 hard neg",
            "resize": "query 224×224, pano W'=224×8",
        },
        "YQ360": {
            "source": "PanoVPR benchmark companion",
            "notes": "smaller DB → narrower speed gap vs sliding-window baselines",
        },
        "SF-XL_panoramic_test": {
            "panoramas": 230_000,
            "covers_pv_test": 2_800_000,
            "note": "panoramic DB replaces per-view PV images for storage efficiency",
        },
        "training": {
            "optimizer": "RiemannianAdam",
            "lr": 1e-5,
            "margin": 0.1,
            "batch": 2,
            "epochs": 60,
            "curvature_c": 1.0,
            "hierarchy_L": 5,
        },
    }
