"""Paper Table 1–2 reference anchors."""

from __future__ import annotations

from typing import Any

TABLE1_SYNTHETIC_3FRAME: dict[str, dict[str, float | int | str]] = {
    "lucky_hdr": {
        "params_k": 66,
        "itpi_ms_1888x1280": 62,
        "psnr_mu_3ev": 36.5,
        "hdr_vdp2_3ev": 43.7,
    },
    "hdrflow_retrained": {"psnr_mu_3ev": 35.0, "hdr_vdp2_3ev": 41.2},
    "kalantari": {"psnr_mu_3ev": 34.1, "hdr_vdp2_3ev": 40.5},
}

TABLE2_ITPI_512: dict[str, float] = {"lucky_hdr_ms": 7.0, "hdrflow_ms": 18.0}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_synthetic_3frame": TABLE1_SYNTHETIC_3FRAME,
        "table2_itpi_512": TABLE2_ITPI_512,
    }
