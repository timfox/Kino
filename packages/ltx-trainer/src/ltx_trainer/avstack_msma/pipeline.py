"""Framework card, paper tables, and evaluation demos (arXiv:2606.04444)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.avstack_msma.collation import build_coco_skeleton, downstream_training_targets
from ltx_trainer.avstack_msma.config import AvstackMsmaConfig
from ltx_trainer.avstack_msma.fusion import fusion_demo, table_collaborative_fusion
from ltx_trainer.avstack_msma.ltx_plan import ltx_integration_plan
from ltx_trainer.avstack_msma.perception import domain_shift_summary, table_infrastructure_perception
from ltx_trainer.avstack_msma.postprocess import ObjectState, SensorCalibration, label_objects_for_sensor
from ltx_trainer.avstack_msma.scenario import aerial_overhead_scenario, ground_vehicle_scenario, multi_agent_smart_city_scenario
from ltx_trainer.avstack_msma.simulation import RunManifest, execute_run_manifest, table_dataset_scale


def framework_card(cfg: AvstackMsmaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AvstackMsmaConfig()
    return {
        "name": "AVstack MS/MA Dataset Generator",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "repo": cfg.repo_url,
        "companion": cfg.companion_arxiv,
        "stack": ["CARLA", "AVstack", "carla-sandbox runner"],
        "pipeline_stages": [
            "Scenario configuration (actors, NPCs, sensors, weather, seed)",
            "Synchronous CARLA simulation + AVstack logging",
            "Reference-frame labels + FOV + depth-based occlusion",
            "COCO / AVstack / MMDet export",
        ],
        "domains": ["ground_vehicle", "aerial_overhead", "infrastructure"],
        "downstream": [t["library"] for t in downstream_training_targets()],
        "terabyte_scale": "Representative ~1 TB collections in under one hour (paper §3.1)",
    }


def labeling_demo() -> dict[str, Any]:
    """Per-sensor labels with occlusion filter smoke."""
    cal_ego = SensorCalibration(x=0.0, y=0.0, z=1.6, yaw=0.0, fov_deg=90.0)
    cal_infra = SensorCalibration(x=30.0, y=0.0, z=8.0, yaw=3.14, pitch=-0.4, fov_deg=70.0)
    objects = [
        ObjectState(1, 25.0, 0.0, 0.0, class_name="car"),
        ObjectState(2, 28.0, 0.0, 0.0, class_name="truck"),
        ObjectState(3, 5.0, 2.0, 0.0, class_name="bicycle"),
    ]
    depth = np.full((1080, 1920), 30.0, dtype=np.float32)
    depth[500:520, 960:1000] = 25.0  # nearer surface occludes far car projection region
    ego_labels = label_objects_for_sensor(objects, cal_ego, depth_map=depth)
    infra_labels = label_objects_for_sensor(objects, cal_infra, depth_map=None)
    return {
        "ego_visible_count": len(ego_labels),
        "infra_visible_count": len(infra_labels),
        "ego_subset_of_infra": len(ego_labels) <= len(infra_labels),
    }


def evaluation_demo(cfg: AvstackMsmaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AvstackMsmaConfig()
    spec = multi_agent_smart_city_scenario(cfg)
    manifest = RunManifest(run_id="demo_ma_001", scenario=spec, log_root="/tmp/avstack_msma/demo_ma_001")
    run = execute_run_manifest(manifest)
    coco = build_coco_skeleton(spec, max_frames_per_sensor=2)
    return {
        "framework": framework_card(cfg),
        "ltx_plan": ltx_integration_plan(cfg),
        "run": run,
        "labeling": labeling_demo(),
        "fusion": fusion_demo(),
        "domain_shift": domain_shift_summary(),
        "coco_export": {"num_images": len(coco.images), "num_annotations": len(coco.annotations)},
        "scenarios": {
            "ground": ground_vehicle_scenario(cfg).to_runner_dict(),
            "multi_agent": spec.to_runner_dict(),
            "aerial": aerial_overhead_scenario(cfg).to_runner_dict(),
        },
        "paper_tables": {
            "dataset_scale": table_dataset_scale(cfg),
            "perception_ap": table_infrastructure_perception(),
            "collaborative_fusion": table_collaborative_fusion(),
        },
    }
