"""P2PSynCodec — plain-to-pseudo synergistic VQ codec (arXiv:2606.05876)."""

from ltx_trainer.p2psyncodec.config import P2PSynCodecConfig
from ltx_trainer.p2psyncodec.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.p2psyncodec.fold import annotate_audio_save_data
from ltx_trainer.p2psyncodec.mock import evaluation_smoke
from ltx_trainer.p2psyncodec.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table1_objective,
    table2_pseudo_vq_ablation,
)
from ltx_trainer.p2psyncodec.quantizer import (
    bitrate_bps,
    bitrate_kbps,
    plain_vq_token,
    pseudo_vq_ce_loss,
    synergistic_quantized_vector,
    teacher_forcing_train_step,
)

__all__ = [
    "P2PSynCodecConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "bitrate_bps",
    "bitrate_kbps",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "pipeline_demo_export",
    "plain_vq_token",
    "pseudo_vq_ce_loss",
    "synergistic_quantized_vector",
    "table1_objective",
    "table2_pseudo_vq_ablation",
    "teacher_forcing_train_step",
]
