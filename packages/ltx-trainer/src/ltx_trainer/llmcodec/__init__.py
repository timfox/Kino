"""LLMCodec — video codec LLM weight compression (arXiv:2606.05861)."""

from ltx_trainer.llmcodec.affine import apply_affine, learn_affine_stub, output_reconstruction_error
from ltx_trainer.llmcodec.codec import CodecResult, compress_yuv420_stub, effective_bitwidth, psnr, rd_curve_stub
from ltx_trainer.llmcodec.compress import compress_weight_matrix
from ltx_trainer.llmcodec.config import CodecType, CodingProfile, LlmCodecConfig, YuvLayout
from ltx_trainer.llmcodec.layout import LIMITATIONS
from ltx_trainer.llmcodec.mapping import pack_yuv420, rt_dequantize, rt_quantize, weight_matrix_to_y_plane, weight_to_yuv420, y_plane_to_weights
from ltx_trainer.llmcodec.mock import evaluation_smoke, run_affine_smoke, run_codec_ablation_smoke, run_compress_layer_smoke, run_rd_curve_smoke
from ltx_trainer.llmcodec.pipeline import (
    benchmarks_bundle,
    codec_psnr_at_2bit,
    evaluation_demo,
    fig1_llama3_2bit_anchors,
    framework_card,
    profile_psnr_at_qp12,
    table_i_compression,
    table_ii_ablation,
)
from ltx_trainer.llmcodec.infer import llmcodec_enabled, llmcodec_qp
from ltx_trainer.llmcodec.vllm_plan import gopex_env_snippet, vllm_integration_plan

__all__ = [
    "CodecResult",
    "CodecType",
    "CodingProfile",
    "LIMITATIONS",
    "LlmCodecConfig",
    "YuvLayout",
    "apply_affine",
    "benchmarks_bundle",
    "codec_psnr_at_2bit",
    "compress_weight_matrix",
    "compress_yuv420_stub",
    "effective_bitwidth",
    "evaluation_demo",
    "evaluation_smoke",
    "fig1_llama3_2bit_anchors",
    "framework_card",
    "gopex_env_snippet",
    "learn_affine_stub",
    "llmcodec_enabled",
    "llmcodec_qp",
    "output_reconstruction_error",
    "pack_yuv420",
    "profile_psnr_at_qp12",
    "psnr",
    "rd_curve_stub",
    "rt_dequantize",
    "rt_quantize",
    "run_affine_smoke",
    "run_codec_ablation_smoke",
    "run_compress_layer_smoke",
    "run_rd_curve_smoke",
    "table_i_compression",
    "table_ii_ablation",
    "vllm_integration_plan",
    "weight_matrix_to_y_plane",
    "weight_to_yuv420",
    "y_plane_to_weights",
]
