"""HDR-NeRF / HDR-Plenoxels dataset metadata (Sec. 5.1)."""

from __future__ import annotations

DATASETS = {
    "hdr_nerf_real": {
        "paper": "HDR-NeRF-Real [12]",
        "train_views_ldr_oe": 18,
        "eval_views": 17,
        "exposures": 5,
        "downsample": 4,
        "resolution": "804×534",
    },
    "hdr_plenoxels_real": {
        "paper": "HDR-Plenoxels-Real [15]",
        "train_views_ldr_oe": 27,
        "eval_views": 13,
        "exposures": 5,
        "downsample": 6,
        "resolution": "992×746",
    },
    "hdr_nerf_syn": {
        "paper": "HDR-NeRF-Syn [12]",
        "train_views_ldr_oe": 18,
        "eval_views": 17,
        "exposures": 5,
        "hdr_gt_eval_only": True,
        "downsample": 2,
        "resolution": "400×400",
    },
}


def dataset_summary() -> dict[str, object]:
    return {"datasets": DATASETS, "exposure_groups": {"ldr_oe": "t1,t3,t5", "ldr_ne": "t2,t4"}}
