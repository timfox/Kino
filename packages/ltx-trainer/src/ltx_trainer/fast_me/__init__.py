"""FAST-ME: foundation-aware adaptive stopping for motion estimation (arXiv:2605.23428)."""

from ltx_trainer.fast_me.blend import blended_cost, should_stop_fast_me, stopping_boundary
from ltx_trainer.fast_me.config import BlockCandidate, FastMEConfig
from ltx_trainer.fast_me.layout import LIMITATIONS
from ltx_trainer.fast_me.ost import (
    empirical_cdf,
    exponential_cdf,
    exponential_stop_threshold,
    ost_stop_index,
)
from ltx_trainer.fast_me.pipeline import (
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_alpha_tradeoff,
    table_foreman_me,
    table_foreman_psnr_scs,
    table_multi_sequence,
)
from ltx_trainer.fast_me.sad import sad_block

__all__ = [
    "BlockCandidate",
    "FastMEConfig",
    "LIMITATIONS",
    "blended_cost",
    "empirical_cdf",
    "evaluation_demo",
    "exponential_cdf",
    "exponential_stop_threshold",
    "framework_card",
    "ost_stop_index",
    "pipeline_demo",
    "sad_block",
    "should_stop_fast_me",
    "stopping_boundary",
    "table_alpha_tradeoff",
    "table_foreman_me",
    "table_foreman_psnr_scs",
    "table_multi_sequence",
]
