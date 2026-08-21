"""AVstack + CARLA MS/MA dataset scaling (Hallyburton et al., arXiv:2606.04444)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SensingDomain(str, Enum):
    GROUND = "ground"
    AERIAL = "aerial"
    INFRASTRUCTURE = "infrastructure"


class SensorModality(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"
    SEMANTIC = "semantic"
    LIDAR = "lidar"
    RADAR = "radar"
    GNSS = "gnss"
    IMU = "imu"


@dataclass
class SensorMount:
    """Per-sensor reference frame relative to actor."""

    modality: SensorModality
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    fov_deg: float = 90.0
    rate_hz: float = 10.0


@dataclass
class AvstackMsmaConfig:
    paper_arxiv: str = "arXiv:2606.04444"
    companion_arxiv: str = "arXiv:2312.04970"
    title: str = "Scaling Datasets for Multi-Sensor, Multi-Agent, and Multi-Domain Learning in Autonomous Systems"
    repo_url: str = "https://github.com/avstack-lab/carla-sandbox.git"
    avstack_ref: str = "AVstack open autonomy platform (ACM/IEEE CPS-IoT Week 2023)"

    sim_fps: float = 20.0
    default_duration_s: float = 600.0
    default_seed: int = 0

    # Representative generated runs (Table 1, §3.1)
    multi_sensor_duration_s: float = 600.0  # 10 min
    multi_agent_duration_s: float = 1500.0  # 25 min
    num_ego_ground: int = 1
    num_ego_multi_agent: int = 4
    num_infra_platforms: int = 5
    num_npc_traffic: int = 30

    ground_sensors: tuple[SensorModality, ...] = (
        SensorModality.RGB,
        SensorModality.SEMANTIC,
        SensorModality.DEPTH,
        SensorModality.LIDAR,
        SensorModality.RADAR,
    )

    # Fusion study defaults (Table 3, §4.2)
    communication_radius_m: float = 150.0
    fusion_local_mAP: float = 0.60
    fusion_naive_no_corr: float = 0.86
    fusion_corr_aware_no_corr: float = 0.90

    export_formats: tuple[str, ...] = field(
        default_factory=lambda: ("avstack_native", "coco", "mmdet", "mmdet3d")
    )
