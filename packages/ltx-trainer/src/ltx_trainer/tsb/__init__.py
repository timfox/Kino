"""Time Segmented Beamforming (BSB / OSB) via dynamic programming."""

from ltx_trainer.tsb.config import TsbConfig
from ltx_trainer.tsb.layout import LIMITATIONS
from ltx_trainer.tsb.mock import evaluation_smoke
from ltx_trainer.tsb.mvdr import (
    mvdr_weights,
    sample_covariance,
    segment_output_power,
    woodbury_inverse_update,
)
from ltx_trainer.tsb.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_birth_death,
    table_distributed_mic,
    table_piecewise_bearing,
    table_simulation_abrupt_change,
    table_swellex96,
    theorem_regret,
)
from ltx_trainer.tsb.regret import universal_regret_bound
from ltx_trainer.tsb.segmented import (
    batch_segmented_beamformer,
    bellman_segment_cost,
    online_segment_step,
    traceback_partitions,
)

__all__ = [
    "LIMITATIONS",
    "TsbConfig",
    "batch_segmented_beamformer",
    "bellman_segment_cost",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "mvdr_weights",
    "online_segment_step",
    "pipeline_demo",
    "sample_covariance",
    "segment_output_power",
    "table_birth_death",
    "table_distributed_mic",
    "table_piecewise_bearing",
    "table_simulation_abrupt_change",
    "table_swellex96",
    "theorem_regret",
    "traceback_partitions",
    "universal_regret_bound",
    "woodbury_inverse_update",
]
