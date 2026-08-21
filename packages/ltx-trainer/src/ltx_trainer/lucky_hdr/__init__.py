"""LuckyHDR — handheld bracket align-and-merge (Li et al. arXiv:2604.19976)."""

from ltx_trainer.lucky_hdr.benchmarks import TABLE1_SYNTHETIC_3FRAME, benchmarks_bundle
from ltx_trainer.lucky_hdr.capture_ae import ae_loss, bracket_evs
from ltx_trainer.lucky_hdr.features import align_features, merge_features
from ltx_trainer.lucky_hdr.layout import LIMITATIONS
from ltx_trainer.lucky_hdr.losses import LuckyHdrLoss
from ltx_trainer.lucky_hdr.merge_net import MergeStage
from ltx_trainer.lucky_hdr.model import LuckyHdr, LuckyHdrConfig
from ltx_trainer.lucky_hdr.pipeline import (
    load_lucky_hdr_checkpoint,
    merge_bracket_paths,
    merge_bracket_stack,
    pipeline_merge_smoke,
)
from ltx_trainer.lucky_hdr.paper import evaluation_demo, framework_card
from ltx_trainer.lucky_hdr.shift_net import ShiftStage
from ltx_trainer.lucky_hdr.synthetic import synthesize_bracket_burst
from ltx_trainer.lucky_hdr.tonemap import normalize_exposure, tone_map_mu
from ltx_trainer.lucky_hdr.warp import warp_image

__all__ = [
    "LIMITATIONS",
    "LuckyHdr",
    "LuckyHdrConfig",
    "LuckyHdrLoss",
    "MergeStage",
    "ShiftStage",
    "TABLE1_SYNTHETIC_3FRAME",
    "ae_loss",
    "align_features",
    "benchmarks_bundle",
    "bracket_evs",
    "evaluation_demo",
    "framework_card",
    "load_lucky_hdr_checkpoint",
    "merge_bracket_paths",
    "merge_bracket_stack",
    "merge_features",
    "normalize_exposure",
    "pipeline_merge_smoke",
    "synthesize_bracket_burst",
    "tone_map_mu",
    "warp_image",
]
