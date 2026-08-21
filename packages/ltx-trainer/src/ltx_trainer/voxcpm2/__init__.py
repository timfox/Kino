"""VoxCPM2 — hierarchical diffusion-autoregressive TTS (arXiv:2606.06928)."""

from ltx_trainer.voxcpm2.audiovae import audiovae_demo, audiovae_v2_params, table10_reconstruction
from ltx_trainer.voxcpm2.backbone import backbone_demo, table1_family_config
from ltx_trainer.voxcpm2.config import GenerationMode, Voxcpm2Config
from ltx_trainer.voxcpm2.eval import eval_smoke, pipeline_demo
from ltx_trainer.voxcpm2.fold import annotate_audio_save_data
from ltx_trainer.voxcpm2.mock import evaluation_smoke
from ltx_trainer.voxcpm2.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table3_seed_tts_eval,
    table4_inference_recipes,
    table7_minimax_sim_summary,
    table9_instruct_tts_eval,
    table11_inference_efficiency,
    training_protocol,
)
from ltx_trainer.voxcpm2.sequence import format_voice_design_text, sequence_demo, sequence_table
from ltx_trainer.voxcpm2.upstream import (
    build_gradio_argv,
    build_infer_argv,
    install_plan,
    upstream_knowledge,
    upstream_root,
    upstream_status,
)

__all__ = [
    "GenerationMode",
    "Voxcpm2Config",
    "annotate_audio_save_data",
    "audiovae_demo",
    "audiovae_v2_params",
    "backbone_demo",
    "benchmarks_bundle",
    "build_gradio_argv",
    "build_infer_argv",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "format_voice_design_text",
    "framework_card",
    "headline_results",
    "install_plan",
    "pipeline_demo",
    "sequence_demo",
    "sequence_table",
    "table1_family_config",
    "table3_seed_tts_eval",
    "table4_inference_recipes",
    "table7_minimax_sim_summary",
    "table9_instruct_tts_eval",
    "table10_reconstruction",
    "table11_inference_efficiency",
    "training_protocol",
    "upstream_knowledge",
    "upstream_root",
    "upstream_status",
]
