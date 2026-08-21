"""Simulation + logging stage stubs (§2.1, Fig. 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.avstack_msma.config import AvstackMsmaConfig
from ltx_trainer.avstack_msma.scenario import (
    ScenarioSpec,
    estimate_collection_scale,
    ground_vehicle_scenario,
    multi_agent_smart_city_scenario,
)


@dataclass
class RunManifest:
    run_id: str
    scenario: ScenarioSpec
    log_root: str
    stages_completed: list[str] = field(default_factory=list)

    def mark(self, stage: str) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)


def pipeline_stages() -> list[str]:
    return [
        "configuration",
        "carla_simulation",
        "avstack_logging",
        "postprocessing",
        "dataset_collation",
    ]


def execute_run_manifest(manifest: RunManifest) -> dict[str, Any]:
    """Non-CARLA smoke: walk the four-stage workflow."""
    for stage in pipeline_stages():
        manifest.mark(stage)
    scale = estimate_collection_scale(manifest.scenario)
    return {
        "run_id": manifest.run_id,
        "log_root": manifest.log_root,
        "stages": manifest.stages_completed,
        "estimated_scale": scale,
        "sync_mode": manifest.scenario.sync_mode,
        "num_actors": len(manifest.scenario.actors),
    }


def table_dataset_scale(cfg: AvstackMsmaConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — benchmarks vs representative generated instances."""
    cfg = cfg or AvstackMsmaConfig()
    ms_spec = ground_vehicle_scenario(cfg)
    ma_spec = multi_agent_smart_city_scenario(cfg)
    ms_scale = estimate_collection_scale(ms_spec, cfg)
    ma_scale = estimate_collection_scale(ma_spec, cfg)
    return [
        {
            "dataset": "KITTI",
            "sensing": "1 LiDAR + 4 RGB",
            "images": 7_500,
            "lidar_frames": 7_500,
            "objects": 51_000,
            "size_gb": 40,
        },
        {
            "dataset": "nuScenes",
            "sensing": "1 LiDAR, 6 RGB, 5 radars",
            "images": 240_000,
            "lidar_frames": 40_000,
            "objects": None,
            "size_gb": 280,
        },
        {
            "dataset": "OPV2V",
            "sensing": "~3 connected agents, LiDAR+RGB each",
            "images": 40_000,
            "lidar_frames": 40_000,
            "objects": 232_000,
            "size_gb": 250,
        },
        {
            "dataset": "Generated multi-sensor (10 min.)",
            "sensing": "RGB, semantic, depth, LiDAR, radar on ego",
            "images": 200_000,
            "lidar_frames": 50_000,
            "objects": 3_100_000,
            "size_gb": 500,
            "estimated_from_spec": ms_scale,
        },
        {
            "dataset": "Generated multi-agent (25 min.)",
            "sensing": "4 ego + 5 infra RGB/LiDAR",
            "images": 300_000,
            "lidar_frames": 100_000,
            "objects": 2_000_000,
            "size_gb": 500,
            "estimated_from_spec": ma_scale,
        },
    ]
