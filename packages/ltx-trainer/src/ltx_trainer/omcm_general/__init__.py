"""Online min-cost matching, general arrivals (arXiv:2606.05546)."""

from ltx_trainer.omcm_general.config import OmcmGeneralConfig
from ltx_trainer.omcm_general.matching import (
    competitive_ratio_bound,
    matching_cost,
    simulate_general_arrivals,
    toy_arrival_stream,
)
from ltx_trainer.omcm_general.mock import evaluation_smoke
from ltx_trainer.omcm_general.pipeline import benchmarks_bundle, evaluation_demo, framework_card

__all__ = [
    "OmcmGeneralConfig",
    "benchmarks_bundle",
    "competitive_ratio_bound",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "matching_cost",
    "simulate_general_arrivals",
    "toy_arrival_stream",
]
