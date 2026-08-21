"""AVstack + CARLA scalable MS/MA/Multi-domain dataset generation (arXiv:2606.04444)."""

from ltx_trainer.avstack_msma.config import AvstackMsmaConfig, SensingDomain, SensorModality, SensorMount
from ltx_trainer.avstack_msma.collation import CocoExport, build_coco_skeleton, downstream_training_targets
from ltx_trainer.avstack_msma.fusion import (
    CorrelationAssumption,
    TrackEstimate,
    covariance_intersection,
    fusion_demo,
    table_collaborative_fusion,
)
from ltx_trainer.avstack_msma.ltx_plan import ltx_integration_plan
from ltx_trainer.avstack_msma.perception import domain_shift_summary, table_infrastructure_perception
from ltx_trainer.avstack_msma.pipeline import evaluation_demo, framework_card, labeling_demo
from ltx_trainer.avstack_msma.postprocess import (
    ObjectState,
    SensorCalibration,
    label_objects_for_sensor,
    visibility_from_depth,
)
from ltx_trainer.avstack_msma.scenario import (
    ScenarioSpec,
    aerial_overhead_scenario,
    estimate_collection_scale,
    ground_vehicle_scenario,
    multi_agent_smart_city_scenario,
)
from ltx_trainer.avstack_msma.simulation import RunManifest, execute_run_manifest, table_dataset_scale

__all__ = [
    "AvstackMsmaConfig",
    "CocoExport",
    "CorrelationAssumption",
    "ObjectState",
    "RunManifest",
    "ScenarioSpec",
    "SensorCalibration",
    "SensorModality",
    "SensorMount",
    "SensingDomain",
    "TrackEstimate",
    "aerial_overhead_scenario",
    "build_coco_skeleton",
    "covariance_intersection",
    "domain_shift_summary",
    "downstream_training_targets",
    "estimate_collection_scale",
    "evaluation_demo",
    "execute_run_manifest",
    "framework_card",
    "fusion_demo",
    "ground_vehicle_scenario",
    "label_objects_for_sensor",
    "labeling_demo",
    "ltx_integration_plan",
    "multi_agent_smart_city_scenario",
    "table_collaborative_fusion",
    "table_dataset_scale",
    "table_infrastructure_perception",
    "visibility_from_depth",
]
