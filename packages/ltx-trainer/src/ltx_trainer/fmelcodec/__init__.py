"""FMelCodec — ultra-low-bitrate mel-spectrogram speech codec (arXiv:2605.25669)."""

from ltx_trainer.fmelcodec.cfm import (
    cfm_loss,
    euler_step,
    ideal_terminal_operator,
    linear_path,
    path_velocity,
    refine_mel_euler,
    refinement_loss,
    self_consistency_loss,
)
from ltx_trainer.fmelcodec.coding import coding_stage_loss, mel_reconstruction_loss, vq_loss
from ltx_trainer.fmelcodec.config import FMelCodecConfig
from ltx_trainer.fmelcodec.layout import LIMITATIONS
from ltx_trainer.fmelcodec.oc_vq import (
    bitrate_bps,
    ema_usage,
    quantize_nearest,
    refresh_coefficient,
    update_codevector,
)
from ltx_trainer.fmelcodec.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_equal_bitrate,
    table_ii_complexity,
    table_iii_public_checkpoints,
    table_iv_stage_complexity,
)

__all__ = [
    "FMelCodecConfig",
    "LIMITATIONS",
    "benchmarks_bundle",
    "bitrate_bps",
    "cfm_loss",
    "coding_stage_loss",
    "ema_usage",
    "euler_step",
    "evaluation_demo",
    "framework_card",
    "headline_results",
    "ideal_terminal_operator",
    "linear_path",
    "mel_reconstruction_loss",
    "path_velocity",
    "pipeline_demo",
    "quantize_nearest",
    "refine_mel_euler",
    "refinement_loss",
    "refresh_coefficient",
    "self_consistency_loss",
    "table_i_equal_bitrate",
    "table_ii_complexity",
    "table_iii_public_checkpoints",
    "table_iv_stage_complexity",
    "update_codevector",
    "vq_loss",
]
