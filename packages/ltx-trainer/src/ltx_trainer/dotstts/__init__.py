"""dots.tts — continuous AR TTS (arXiv:2606.07080)."""

from ltx_trainer.dotstts.ar_fm import ar_fm_demo, block_causal_mask, flow_matching_loss
from ltx_trainer.dotstts.audiovae import audiovae_demo, audiovae_loss_stage1, audiovae_loss_stage2
from ltx_trainer.dotstts.config import DotsttsConfig
from ltx_trainer.dotstts.eval import eval_smoke, pipeline_demo
from ltx_trainer.dotstts.fold import annotate_audio_save_data
from ltx_trainer.dotstts.meanflow import meanflow_demo, meanflow_loss
from ltx_trainer.dotstts.mock import evaluation_smoke
from ltx_trainer.dotstts.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_audiovae_reconstruction,
    table2_seed_tts_eval,
    table3_minimax_multilingual_average,
    table5_emergent_tts_selected,
    table_efficiency,
    training_stages,
)
from ltx_trainer.dotstts.soar import soar_demo, soar_loss
from ltx_trainer.dotstts.upstream import (
    build_gradio_argv,
    build_infer_argv,
    build_prepare_data_argv,
    build_train_argv,
    hf_checkpoints,
    install_plan,
    upstream_knowledge,
    upstream_root,
    upstream_status,
)

__all__ = [
    "DotsttsConfig",
    "annotate_audio_save_data",
    "ar_fm_demo",
    "audiovae_demo",
    "audiovae_loss_stage1",
    "audiovae_loss_stage2",
    "benchmarks_bundle",
    "block_causal_mask",
    "build_gradio_argv",
    "build_infer_argv",
    "build_prepare_data_argv",
    "build_train_argv",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "flow_matching_loss",
    "framework_card",
    "headline_results",
    "hf_checkpoints",
    "install_plan",
    "meanflow_demo",
    "meanflow_loss",
    "pipeline_demo",
    "soar_demo",
    "soar_loss",
    "table1_audiovae_reconstruction",
    "table2_seed_tts_eval",
    "table3_minimax_multilingual_average",
    "table5_emergent_tts_selected",
    "table_efficiency",
    "training_stages",
    "upstream_knowledge",
    "upstream_root",
    "upstream_status",
]
