"""Scenario configuration layer (§2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.avstack_msma.config import AvstackMsmaConfig, SensingDomain, SensorModality, SensorMount


@dataclass
class ActorSpec:
    name: str
    domain: SensingDomain
    mobile: bool = True
    sensors: list[SensorMount] = field(default_factory=list)


@dataclass
class ScenarioSpec:
    """Human-readable scenario: actors, NPCs, weather, map, seed, duration."""

    map_name: str = "Town10HD"
    weather: str = "ClearNoon"
    seed: int = 0
    duration_s: float = 600.0
    num_npc: int = 30
    actors: list[ActorSpec] = field(default_factory=list)
    sync_mode: bool = True

    def to_runner_dict(self) -> dict[str, Any]:
        return {
            "map": self.map_name,
            "weather": self.weather,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "num_npc": self.num_npc,
            "sync": self.sync_mode,
            "actors": [
                {
                    "name": a.name,
                    "domain": a.domain.value,
                    "mobile": a.mobile,
                    "sensors": [
                        {
                            "modality": s.modality.value,
                            "pose": [s.x, s.y, s.z, s.roll, s.pitch, s.yaw],
                            "fov_deg": s.fov_deg,
                            "rate_hz": s.rate_hz,
                        }
                        for s in a.sensors
                    ],
                }
                for a in self.actors
            ],
        }


def default_sensor_mounts(modalities: tuple[SensorModality, ...], *, rate_hz: float = 10.0) -> list[SensorMount]:
    mounts: list[SensorMount] = []
    for m in modalities:
        fov = 110.0 if m in (SensorModality.RGB, SensorModality.DEPTH, SensorModality.SEMANTIC) else 0.0
        mounts.append(SensorMount(modality=m, z=1.6 if m != SensorModality.RADAR else 0.5, fov_deg=fov, rate_hz=rate_hz))
    return mounts


def ground_vehicle_scenario(cfg: AvstackMsmaConfig | None = None) -> ScenarioSpec:
    cfg = cfg or AvstackMsmaConfig()
    ego = ActorSpec(
        name="ego_0",
        domain=SensingDomain.GROUND,
        mobile=True,
        sensors=default_sensor_mounts(cfg.ground_sensors),
    )
    return ScenarioSpec(
        seed=cfg.default_seed,
        duration_s=cfg.multi_sensor_duration_s,
        num_npc=cfg.num_npc_traffic,
        actors=[ego],
    )


def multi_agent_smart_city_scenario(cfg: AvstackMsmaConfig | None = None) -> ScenarioSpec:
    cfg = cfg or AvstackMsmaConfig()
    actors: list[ActorSpec] = []
    for i in range(cfg.num_ego_multi_agent):
        actors.append(
            ActorSpec(
                name=f"ego_{i}",
                domain=SensingDomain.GROUND,
                sensors=default_sensor_mounts(
                    (SensorModality.RGB, SensorModality.LIDAR, SensorModality.RADAR),
                    rate_hz=10.0,
                ),
            )
        )
    for j in range(cfg.num_infra_platforms):
        actors.append(
            ActorSpec(
                name=f"infra_{j}",
                domain=SensingDomain.INFRASTRUCTURE,
                mobile=False,
                sensors=default_sensor_mounts((SensorModality.RGB, SensorModality.LIDAR), rate_hz=5.0),
            )
        )
    return ScenarioSpec(
        seed=cfg.default_seed + 1,
        duration_s=cfg.multi_agent_duration_s,
        num_npc=cfg.num_npc_traffic,
        actors=actors,
    )


def aerial_overhead_scenario(cfg: AvstackMsmaConfig | None = None) -> ScenarioSpec:
    cfg = cfg or AvstackMsmaConfig()
    aerial = ActorSpec(
        name="uav_0",
        domain=SensingDomain.AERIAL,
        mobile=True,
        sensors=[
            SensorMount(SensorModality.RGB, z=80.0, pitch=-75.0, fov_deg=90.0, rate_hz=5.0),
            SensorMount(SensorModality.LIDAR, z=80.0, pitch=-90.0, rate_hz=5.0),
        ],
    )
    return ScenarioSpec(map_name="Town04", duration_s=300.0, actors=[aerial])


def estimate_collection_scale(spec: ScenarioSpec, cfg: AvstackMsmaConfig | None = None) -> dict[str, int]:
    """Approximate logged volume from duration × sensors × rates (Table 1 style)."""
    cfg = cfg or AvstackMsmaConfig()
    n_frames = int(spec.duration_s * cfg.sim_fps)
    n_images = 0
    n_lidar = 0
    n_radar = 0
    for actor in spec.actors:
        for s in actor.sensors:
            n = int(spec.duration_s * s.rate_hz)
            if s.modality in (SensorModality.RGB, SensorModality.DEPTH, SensorModality.SEMANTIC):
                n_images += n
            elif s.modality == SensorModality.LIDAR:
                n_lidar += n
            elif s.modality == SensorModality.RADAR:
                n_radar += n
    # Heuristic objects: NPC + ego traffic interactions
    n_objects = n_frames * max(1, spec.num_npc) * max(1, len(spec.actors))
    return {
        "frames": n_frames,
        "images": n_images,
        "lidar_frames": n_lidar,
        "radar_frames": n_radar,
        "objects_labeled": n_objects,
    }
