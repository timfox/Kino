"""Table 2/4/5 and summary anchors (§5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.radiusfps.constants import (
    CPU_SPEEDUP_MAX,
    E2E_SPEEDUP_MAX_FP_RADIUS_G,
    E2E_SPEEDUP_MAX_GPU_FPS,
    FIG1_FPS_LATENCY_SHARE,
    FIG15_PEAK_SPEEDUP,
    GPU_SPEEDUP_MAX,
    QUICKFPS_MEMORY_RATIO,
    SAMPLING_SPEEDUP_MAX_FP_RADIUS_G,
    TABLE2_POINTMETABASE,
    TABLE4_ABLATION,
)


def table_2_pointmetabase() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_POINTMETABASE]


def table_4_ablation() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE4_ABLATION]


def fig1_fps_share() -> dict[str, float]:
    return dict(FIG1_FPS_LATENCY_SHARE)


def fig15_peak_speedups() -> dict[str, Any]:
    return dict(FIG15_PEAK_SPEEDUP)


def kitti_e2e_speedup_radiusfps_g() -> float:
    gpu_fps = 1734.057
    radius_g = 691.230
    return round(gpu_fps / radius_g, 2)


def summary_anchors() -> dict[str, Any]:
    return {
        "cpu_speedup_max": CPU_SPEEDUP_MAX,
        "gpu_speedup_max": GPU_SPEEDUP_MAX,
        "e2e_speedup_vs_gpu_fps": E2E_SPEEDUP_MAX_GPU_FPS,
        "e2e_speedup_fp_radius_g": E2E_SPEEDUP_MAX_FP_RADIUS_G,
        "sampling_speedup_fp_radius_g": SAMPLING_SPEEDUP_MAX_FP_RADIUS_G,
        "quickfps_memory_ratio": QUICKFPS_MEMORY_RATIO,
        "kitti_e2e_speedup_radiusfps_g": kitti_e2e_speedup_radiusfps_g(),
        "s3dis_radiusfps_g_mIoU": 68.28,
        "scannet_radiusfps_g_mIoU": 71.21,
    }
