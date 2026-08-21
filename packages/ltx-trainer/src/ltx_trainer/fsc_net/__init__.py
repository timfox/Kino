"""FSC-Net — FFC + progressive learning for speech BWE (arXiv:2606.06962)."""

from ltx_trainer.fsc_net.config import FscNetConfig
from ltx_trainer.fsc_net.eval import eval_smoke, pipeline_demo
from ltx_trainer.fsc_net.ffc import channel_wise_subband, fast_fourier_conv2d, ffc_demo
from ltx_trainer.fsc_net.fold import annotate_audio_save_data
from ltx_trainer.fsc_net.losses import (
    log_spectral_distance,
    losses_demo,
    mr_stft_loss,
    stage_loss,
)
from ltx_trainer.fsc_net.mock import evaluation_smoke
from ltx_trainer.fsc_net.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_vctk,
    table2_ears,
    table3_ablation,
)
from ltx_trainer.fsc_net.progressive import progressive_demo, progressive_targets, sliding_window_average

__all__ = [
    "FscNetConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "channel_wise_subband",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "fast_fourier_conv2d",
    "ffc_demo",
    "framework_card",
    "headline_results",
    "log_spectral_distance",
    "losses_demo",
    "mr_stft_loss",
    "pipeline_demo",
    "progressive_demo",
    "progressive_targets",
    "sliding_window_average",
    "stage_loss",
    "table1_vctk",
    "table2_ears",
    "table3_ablation",
]
