"""CPC3 frame-aligned Canary–WavLM fusion for hearing-aid intelligibility."""

from ltx_trainer.cpc3_faf.align import (
    adaptive_map_to_length,
    conv_stride_downsample,
    masked_average_downsample,
    temporal_shift,
)
from ltx_trainer.cpc3_faf.config import Cpc3FafConfig
from ltx_trainer.cpc3_faf.fusion import (
    frame_aligned_fuse,
    pool_late_fuse,
    predict_bounded,
    uniform_score_average,
)
from ltx_trainer.cpc3_faf.layout import LIMITATIONS
from ltx_trainer.cpc3_faf.mock import evaluation_smoke
from ltx_trainer.cpc3_faf.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_i_main_comparison,
    table_ii_diagnostics,
    table_iii_robustness,
)
from ltx_trainer.cpc3_faf.pooling import attention_pool

__all__ = [
    "Cpc3FafConfig",
    "LIMITATIONS",
    "adaptive_map_to_length",
    "attention_pool",
    "benchmarks_bundle",
    "conv_stride_downsample",
    "evaluation_demo",
    "evaluation_smoke",
    "frame_aligned_fuse",
    "framework_card",
    "headline_results",
    "masked_average_downsample",
    "pool_late_fuse",
    "predict_bounded",
    "table_i_main_comparison",
    "table_ii_diagnostics",
    "table_iii_robustness",
    "temporal_shift",
    "uniform_score_average",
]
