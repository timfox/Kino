"""GS-NFS paper table anchors."""

from __future__ import annotations

from typing import Any


def table2_latency_ms() -> dict[str, dict[str, float]]:
    """Table 2 aggregate encode/decode latency (ms)."""
    return {
        "HiFi4G": {"encode": 18.0, "decode": 14.0},
        "N3DV": {"encode": 23.0, "decode": 21.0},
        "V3-2D": {"encode": 370.0, "decode": 290.0},
        "LTS-Draco": {"encode": 400.0, "decode": 158.0},
        "G-PCC": {"encode": 8804.0, "decode": 6928.0},
        "MesonGS": {"encode": 28996.0, "decode": 1096.0},
        "GS-NFS": {"encode": 23.0, "decode": 21.0},
    }


def table3_mean_comparison() -> dict[str, dict[str, float]]:
    """Table 3 mean ΔPSNR and RCR vs GS-NFS (HiFi4G + N3DV pooled)."""
    return {
        "V3-2D": {"delta_psnr": 0.00, "rcr": 0.55},
        "MesonGS": {"delta_psnr": -0.04, "rcr": 1.91},
        "LTS-Draco": {"delta_psnr": -0.05, "rcr": 2.98},
        "G-PCC": {"delta_psnr": 0.01, "rcr": 4.85},
    }


def table_a8_klt_sizes_mb() -> dict[str, dict[str, float]]:
    """Appendix Table A.8 compressed size (MB)."""
    return {
        "flame_salmon": {"rgb": 11.94, "yuv": 11.37, "klt": 8.35},
        "sear_steak": {"rgb": 9.29, "yuv": 8.99, "klt": 6.78},
        "Actor1": {"rgb": 10.97, "yuv": 10.43, "klt": 9.90},
    }


def jetson_decode_ms() -> dict[str, float]:
    """Table 4 Jetson Orin decode latency samples."""
    return {
        "Actor1_SH0": 40.0,
        "Actor1_SH3": 57.0,
        "flame_salmon_SH0": 59.0,
        "flame_salmon_SH2": 120.0,
    }


def dataset_catalog() -> dict[str, Any]:
    return {
        "HiFi4G": {"sequences": 7, "frames": 200, "sh_degree": "0..3"},
        "N3DV": {"sequences": 6, "frames": 300, "sh_degree": "0..2"},
    }
