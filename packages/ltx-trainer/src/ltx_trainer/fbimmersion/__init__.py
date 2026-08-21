"""FBImmersion: quantifying full-body immersion (Bakir et al., arXiv:2605.22521)."""

from ltx_trainer.fbimmersion.config import (
    ACTIVITIES,
    CONDITIONS,
    FBImmersionConfig,
    ActivityName,
    ConditionName,
    OperatingMode,
)
from ltx_trainer.fbimmersion.gyration import (
    GyrationCircle,
    gyration_circle_from_iqrs,
    gyration_circle_from_submetrics,
    iqr_to_polygon_vertices,
    radius_of_gyration,
)
from ltx_trainer.fbimmersion.immersion import (
    circle_intersection_area,
    circle_union_area,
    immersion_index_percent,
)
from ltx_trainer.fbimmersion.miros import MIROSPlatformSpec, operating_mode_description, platform_spec, scale_ladder
from ltx_trainer.fbimmersion.pipeline import (
    acceleration_tracking_table,
    compute_immersion_for_activity,
    dataset_card,
    demo_boating_standing_series,
    demo_skiing_series,
    immersion_layers,
    reference_immersion_table,
)
from ltx_trainer.fbimmersion.submetrics import (
    SKIING_SUBMETRICS,
    SUBMETRICS_BY_ACTIVITY,
    extract_submetric_values,
    interquartile_range,
    minimum_enclosing_circle_radius,
    submetric_variability,
)

__all__ = [
    "ACTIVITIES",
    "CONDITIONS",
    "FBImmersionConfig",
    "GyrationCircle",
    "MIROSPlatformSpec",
    "SKIING_SUBMETRICS",
    "SUBMETRICS_BY_ACTIVITY",
    "ActivityName",
    "ConditionName",
    "OperatingMode",
    "acceleration_tracking_table",
    "circle_intersection_area",
    "circle_union_area",
    "compute_immersion_for_activity",
    "dataset_card",
    "demo_boating_standing_series",
    "demo_skiing_series",
    "extract_submetric_values",
    "gyration_circle_from_iqrs",
    "gyration_circle_from_submetrics",
    "immersion_index_percent",
    "immersion_layers",
    "interquartile_range",
    "iqr_to_polygon_vertices",
    "minimum_enclosing_circle_radius",
    "operating_mode_description",
    "platform_spec",
    "radius_of_gyration",
    "reference_immersion_table",
    "scale_ladder",
    "submetric_variability",
]
