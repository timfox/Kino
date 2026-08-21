"""RFDT-Channel framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig
from ltx_trainer.rfdt_channel.materials import material_binding_summary
from ltx_trainer.rfdt_channel.reconstruction import reconstruction_pipeline_smoke
from ltx_trainer.rfdt_channel.regularization import regularization_summary
from ltx_trainer.rfdt_channel.sionna_rt import (
    compare_material_configs,
    radio_map_delta_smoke,
    simulate_link,
)


def framework_card(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "scene": cfg.scene_name,
        "workflow": [
            "RGB video + LiDAR acquisition (Jetson Orin)",
            "COLMAP → 3DGS → SuGaR mesh",
            "LiDAR-guided Blender RF regularization",
            "OpenScene semantic → ITU-R P.2040 materials",
            "Sionna RT @ 28 GHz → CIR / CFR / Radio Map",
        ],
        "hardware": {
            "edge": cfg.edge_platform,
            "lidar": cfg.lidar_model,
            "camera": cfg.camera_interface,
            "middleware": cfg.middleware,
        },
        "simulation": {
            "tool": "Sionna RT",
            "frequency_ghz": cfg.carrier_ghz,
            "max_depth": cfg.max_depth,
            "bandwidth_mhz": cfg.bandwidth_mhz,
            "tx_m": list(cfg.tx_position_m),
            "rx_m": list(cfg.rx_position_m),
        },
        "headline": {
            "paths_all_concrete": cfg.paths_all_concrete,
            "paths_multi_material": cfg.paths_multi_material,
            "max_cir": cfg.max_cir_magnitude,
        },
    }


def table1_hardware() -> list[dict[str, str]]:
    """Paper Table I."""
    return [
        {"component": "NVIDIA Jetson Orin", "function": "Edge-side data acquisition and organization"},
        {"component": "GMSL camera", "function": "Indoor video and RGB image sequences"},
        {"component": "Hesai LiDAR", "function": "Metric-scale point clouds and geometric reference"},
        {"component": "ROS 2", "function": "Point-cloud organization and timestamp recording"},
    ]


def fig3_statistics(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    return {
        "effective_path_count": {
            "All_Concrete": cfg.paths_all_concrete,
            "Multi_Material": cfg.paths_multi_material,
        },
        "max_cir_magnitude": {
            "All_Concrete": cfg.max_cir_magnitude,
            "Multi_Material": cfg.max_cir_magnitude,
        },
        "interpretation": (
            "Material binding suppresses weak reflection/transmission/scattering paths; "
            "dominant path amplitude unchanged."
        ),
    }


def evaluation_demo(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    return {
        "framework": framework_card(cfg),
        "table1_hardware": table1_hardware(),
        "reconstruction": reconstruction_pipeline_smoke(cfg),
        "regularization": regularization_summary(cfg),
        "materials": material_binding_summary(cfg),
        "material_comparison": compare_material_configs(cfg),
        "fig3": fig3_statistics(cfg),
        "radio_map": radio_map_delta_smoke(cfg),
        "cir_smoke": {
            "All_Concrete": simulate_link("All_Concrete", cfg),
            "Multi_Material": simulate_link("Multi_Material", cfg),
        },
    }
