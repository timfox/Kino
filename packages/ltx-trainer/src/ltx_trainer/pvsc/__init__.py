"""PVSC — perception-aware video semantic communication (arXiv:2605.19397)."""

from ltx_trainer.pvsc.channel import (
    awgn_receive,
    channel_bandwidth_ratio,
    nmse_db,
    zf_equalize,
)
from ltx_trainer.pvsc.config import PVSCConfig
from ltx_trainer.pvsc.entropy import DEFAULT_RATE_SET, quantize_rate, symbol_length_factor
from ltx_trainer.pvsc.layout import LIMITATIONS
from ltx_trainer.pvsc.loss import overall_objective, rate_loss, reconstruction_loss_l1
from ltx_trainer.pvsc.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_savings,
    pipeline_demo,
    table_i_bd_cbr_awgn_snr6,
    table_i_pvsc_by_gop,
    table_iv_complexity,
    table_vi_loss_ablation,
)

__all__ = [
    "DEFAULT_RATE_SET",
    "LIMITATIONS",
    "PVSCConfig",
    "awgn_receive",
    "benchmarks_bundle",
    "channel_bandwidth_ratio",
    "evaluation_demo",
    "framework_card",
    "headline_savings",
    "nmse_db",
    "overall_objective",
    "pipeline_demo",
    "quantize_rate",
    "rate_loss",
    "reconstruction_loss_l1",
    "symbol_length_factor",
    "table_i_bd_cbr_awgn_snr6",
    "table_i_pvsc_by_gop",
    "table_iv_complexity",
    "table_vi_loss_ablation",
    "zf_equalize",
]
