"""360VDS / 360UHD dataset card (Sec. IV)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3po.config import ERP_TRAIN_HEIGHT, ERP_TRAIN_WIDTH


def datasets_card() -> dict[str, Any]:
    return {
        "360VDS": {
            "clips_total": 590,
            "train_clips": 545,
            "test_clips": 45,
            "frames_per_clip_max": 20,
            "erp_resolution": [ERP_TRAIN_WIDTH, ERP_TRAIN_HEIGHT],
            "sources": ["psych-360", "open 360° QA/compression/salience corpora"],
        },
        "360UHD": {
            "test_clips": 8,
            "resolutions": ["1280x720", "2560x1280", "3840x1920", "4096x2048"],
            "note": "HD–4K subset of 360VDS test without downscale",
        },
        "MiG_Panorama": {
            "total_clips": 204,
            "public_test_clips": 4,
        },
        "pretrain": "Vimeo90K then fine-tune on 360VDS",
    }
