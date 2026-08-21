"""RFDT-Channel RF digital twin workflow (Yao et al., arXiv:2606.01261)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RfdtChannelConfig:
    paper_arxiv: str = "arXiv:2606.01261"
    title: str = (
        "RFDT-Channel: RGB-LiDAR-Based RF Digital Twin Scene "
        "Construction for 28 GHz Indoor Ray-Tracing Channel Simulation"
    )
    scene_name: str = "scene_camera_lidar"

    # Hardware (Table I, §III)
    edge_platform: str = "NVIDIA Jetson Orin"
    lidar_model: str = "Hesai PandarXT-32"
    camera_interface: str = "GMSL"
    middleware: str = "ROS 2"

    # Visual reconstruction chain (§IV-B)
    reconstruction_pipeline: tuple[str, ...] = ("COLMAP", "3DGS", "SuGaR")

    # Sionna RT setup (§V-A)
    carrier_ghz: float = 28.0
    max_depth: int = 8
    enable_diffraction: bool = True
    enable_scattering: bool = True
    path_types: tuple[str, ...] = (
        "LOS",
        "specular_reflection",
        "diffuse_reflection",
        "refraction",
    )

    tx_position_m: Tuple[float, float, float] = (1.5, 0.0, 0.8)
    rx_position_m: Tuple[float, float, float] = (1.5, 4.0, 0.8)
    link_distance_m: float = 4.0

    num_subcarriers: int = 1024
    subcarrier_spacing_hz: float = 30_000.0
    bandwidth_mhz: float = 30.72

    # Material configs (§V)
    material_configs: tuple[str, ...] = ("All_Concrete", "Multi_Material")
    semantic_materials: tuple[str, ...] = ("concrete", "glass", "wood", "metal")
    itu_material_ref: str = "ITU-R P.2040"

    # Headline results (Fig. 3)
    paths_all_concrete: int = 742
    paths_multi_material: int = 52
    max_cir_magnitude: float = 4.63e-4

    # Regularization ops (§IV-C)
    blender_regularization_ops: tuple[str, ...] = (
        "gravity_alignment",
        "axis_normalization",
        "wall_solidify",
        "door_window_boolean",
        "topology_repair",
    )

    openscene_categories: tuple[str, ...] = field(
        default_factory=lambda: (
            "wall",
            "floor",
            "ceiling",
            "window",
            "door",
            "wooden_furniture",
            "metal_cabinet",
        )
    )
