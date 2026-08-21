"""Paper anchors — RadiusFPS (Yu et al., arXiv:2606.06255)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06255"
PAPER_TITLE = (
    "RadiusFPS: Efficient Farthest Point Sampling on CPUs and GPUs via Spherical Voxel Pruning"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

DATASETS: tuple[str, ...] = ("S3DIS", "ScanNet", "SemanticKITTI")
BACKBONES: tuple[str, ...] = ("PointMetaBase", "PointVector")

BASELINE_SAMPLERS: tuple[str, ...] = (
    "FPS (CPU)",
    "FPS + NPDU",
    "FPS (GPU)",
    "QuickFPS",
    "RadiusFPS",
    "RadiusFPS-G",
    "FastPoint + RadiusFPS-G",
)

# Fig. 1 — FPS latency share on PointMetaBase validation (25% subset).
FIG1_FPS_LATENCY_SHARE = {"S3DIS": 0.776, "ScanNet": 0.890}

# Headline speedups from abstract / Sec. 1.
CPU_SPEEDUP_MAX = 186.0
GPU_SPEEDUP_MAX = 52.0
E2E_SPEEDUP_MAX_GPU_FPS = 2.5
E2E_SPEEDUP_MAX_FP_RADIUS_G = 3.3
SAMPLING_SPEEDUP_MAX_FP_RADIUS_G = 11.7
QUICKFPS_MEMORY_RATIO = 0.5  # ~half GPU memory vs QuickFPS

# Table 2 — PointMetaBase E2E (OA %, mIoU %, Runtime s).
TABLE2_POINTMETABASE: tuple[dict[str, str | float], ...] = (
    {
        "method": "FPS (CPU)",
        "S3DIS_OA": 89.59,
        "S3DIS_mIoU": 67.90,
        "S3DIS_s": 351.846,
        "ScanNet_OA": 89.68,
        "ScanNet_mIoU": 70.93,
        "ScanNet_s": 2861.794,
        "KITTI_OA": 89.53,
        "KITTI_mIoU": 49.72,
        "KITTI_s": 14697.007,
    },
    {
        "method": "FPS (GPU)",
        "S3DIS_OA": 89.60,
        "S3DIS_mIoU": 67.90,
        "S3DIS_s": 40.176,
        "ScanNet_OA": 89.63,
        "ScanNet_mIoU": 70.80,
        "ScanNet_s": 296.860,
        "KITTI_OA": 89.59,
        "KITTI_mIoU": 49.71,
        "KITTI_s": 1734.057,
    },
    {
        "method": "QuickFPS",
        "S3DIS_OA": 89.60,
        "S3DIS_mIoU": 67.90,
        "S3DIS_s": 18.363,
        "ScanNet_OA": 89.60,
        "ScanNet_mIoU": 70.96,
        "ScanNet_s": 106.466,
        "KITTI_OA": 89.54,
        "KITTI_mIoU": 49.54,
        "KITTI_s": 859.166,
    },
    {
        "method": "RadiusFPS-G",
        "S3DIS_OA": 89.75,
        "S3DIS_mIoU": 68.28,
        "S3DIS_s": 15.335,
        "ScanNet_OA": 89.75,
        "ScanNet_mIoU": 71.21,
        "ScanNet_s": 92.346,
        "KITTI_OA": 89.58,
        "KITTI_mIoU": 49.26,
        "KITTI_s": 691.230,
    },
    {
        "method": "FastPoint + RadiusFPS-G",
        "S3DIS_OA": 89.25,
        "S3DIS_mIoU": 67.58,
        "S3DIS_s": 11.248,
        "ScanNet_OA": 89.08,
        "ScanNet_mIoU": 69.43,
        "ScanNet_s": 68.860,
        "KITTI_OA": 88.75,
        "KITTI_mIoU": 47.55,
        "KITTI_s": 531.202,
    },
)

# Table 4 — S3DIS ablation (mIoU %, speedup vs GPU FPS).
TABLE4_ABLATION: tuple[dict[str, str | float | bool], ...] = (
    {"variant": "FPS (GPU)", "radius_pruning": False, "point_skip": False, "fk1": False, "fk2": False, "mIoU": 67.90, "speedup": 1.00},
    {"variant": "RadiusFPS full", "radius_pruning": True, "point_skip": True, "fk1": False, "fk2": False, "mIoU": 67.90, "speedup": 2.58},
    {"variant": "Radius pruning only", "radius_pruning": True, "point_skip": False, "fk1": False, "fk2": False, "mIoU": 67.90, "speedup": 2.42},
    {"variant": "Point skip only", "radius_pruning": False, "point_skip": True, "fk1": False, "fk2": False, "mIoU": 19.73, "speedup": 3.68},
    {"variant": "RadiusFPS-G full", "radius_pruning": True, "point_skip": True, "fk1": True, "fk2": True, "mIoU": 68.28, "speedup": 2.68},
    {"variant": "Fusion kernel 1 only", "radius_pruning": True, "point_skip": True, "fk1": True, "fk2": False, "mIoU": 68.28, "speedup": 2.24},
    {"variant": "Fusion kernel 2 only", "radius_pruning": True, "point_skip": True, "fk1": False, "fk2": True, "mIoU": 68.28, "speedup": 2.11},
    {"variant": "GPU baseline no fusion", "radius_pruning": True, "point_skip": True, "fk1": False, "fk2": False, "mIoU": 68.28, "speedup": 1.87},
)

# Fig. 15 peak speedups vs vanilla FPS.
FIG15_PEAK_SPEEDUP = {
    "gpu": {"Stanford Dragon": 1.6, "SemanticKITTI Velodyne": 4.7, "S3DIS Room": 52.4},
    "cpu": {"Stanford Dragon": 38.94, "SemanticKITTI Velodyne": 46.57, "S3DIS Room": 186.56},
}

SPHERICAL_RADIUS_FACTOR = 0.8660254037844386  # sqrt(3)/2
