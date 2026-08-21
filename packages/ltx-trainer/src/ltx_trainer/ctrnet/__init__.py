"""CTRnet + PuLSS — cross-talk reduction for real-recorded conversational separation (arXiv:2605.19695)."""

from ltx_trainer.ctrnet.config import Chime6Split, CtrnetConfig
from ltx_trainer.ctrnet.fcp import estimate_fcp_filter, fcp_weight
from ltx_trainer.ctrnet.layout import LIMITATIONS
from ltx_trainer.ctrnet.losses import g_loss, overlap_sampling_weight, speaker_activity_loss
from ltx_trainer.ctrnet.mock import evaluation_smoke
from ltx_trainer.ctrnet.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table_hyperparameters,
    table_ii_ctrnet_close_talk,
    table_iii_pulss_farfield,
    table_iv_oracle_diarization,
    table_v_estimated_diarization,
)
from ltx_trainer.ctrnet.pulss import enumerate_sync_delay, pseudo_label_at_reference

__all__ = [
    "LIMITATIONS",
    "Chime6Split",
    "CtrnetConfig",
    "benchmarks_bundle",
    "enumerate_sync_delay",
    "estimate_fcp_filter",
    "evaluation_demo",
    "evaluation_smoke",
    "fcp_weight",
    "framework_card",
    "g_loss",
    "headline_results",
    "overlap_sampling_weight",
    "pseudo_label_at_reference",
    "speaker_activity_loss",
    "table_hyperparameters",
    "table_ii_ctrnet_close_talk",
    "table_iii_pulss_farfield",
    "table_iv_oracle_diarization",
    "table_v_estimated_diarization",
]
