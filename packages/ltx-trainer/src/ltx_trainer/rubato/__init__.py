"""Rubato: time-aligned piano score transcription; InterMo text notation (Tamer et al., arXiv:2605.24291)."""

from ltx_trainer.rubato.config import (
    InterMoDialect,
    RubatoConfig,
    RubatoTrainingMix,
    dialect_registry,
    training_data_rows,
)
from ltx_trainer.rubato.encoding import inverse_sequence_length_weight, timestamp_smoothing_distribution
from ltx_trainer.rubato.intermo import (
    BARLINE_PATTERN,
    STAFF_LEFT,
    STAFF_RIGHT,
    metric_interval_tokens,
    parse_barline,
    pitch_case_is_onset,
    sum_metric_intervals,
    tast_first_bar_example,
    validate_measure_metric_sum,
)
from ltx_trainer.rubato.layout import LIMITATIONS
from ltx_trainer.rubato.pipeline import (
    encoding_demo,
    evaluation_demo,
    framework_card,
    intermo_validation_demo,
    table_omr_ned,
    table_temporal_f1,
    table_version_matching,
)

__all__ = [
    "BARLINE_PATTERN",
    "LIMITATIONS",
    "InterMoDialect",
    "RubatoConfig",
    "RubatoTrainingMix",
    "STAFF_LEFT",
    "STAFF_RIGHT",
    "dialect_registry",
    "encoding_demo",
    "evaluation_demo",
    "framework_card",
    "intermo_validation_demo",
    "inverse_sequence_length_weight",
    "metric_interval_tokens",
    "parse_barline",
    "pitch_case_is_onset",
    "sum_metric_intervals",
    "table_omr_ned",
    "table_temporal_f1",
    "table_version_matching",
    "tast_first_bar_example",
    "timestamp_smoothing_distribution",
    "training_data_rows",
    "validate_measure_metric_sum",
]
