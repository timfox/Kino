"""ODI-SR + SUN360 training card."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "train": "ODI-SR + SUN360 (1151 ERP images @ 2048×1024)",
        "test_odisr": 100,
        "test_sun360": 100,
        "lr_patch": "128×128 cubic downsample",
        "hr_patch": "128r×128r arbitrary r",
        "metrics": ["WS-PSNR", "WS-SSIM"],
    }
