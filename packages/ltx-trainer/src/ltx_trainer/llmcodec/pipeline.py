"""LLMCodec framework card, paper tables, and demos (arXiv:2606.05861)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.llmcodec.config import CodecType, CodingProfile, LlmCodecConfig
from ltx_trainer.llmcodec.layout import LIMITATIONS
from ltx_trainer.llmcodec.mock import (
    evaluation_smoke,
    run_affine_smoke,
    run_codec_ablation_smoke,
    run_compress_layer_smoke,
    run_rd_curve_smoke,
)


def framework_card(cfg: LlmCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or LlmCodecConfig()
    return {
        "name": "LLMCodec — video codec-based LLM weight compression",
        "paper": c.paper_arxiv,
        "authors": "Rui Wang, Yan Zhao, Li Song, Zhengxue Cheng (Shanghai Jiao Tong University)",
        "task": "Post-training LLM weight compression via affine outlier removal + RTN + VVC/VVenC",
        "idea": (
            "Map 2D weight tensors to YUV420 (Y-only data, constant U/V), compress with All-Intra VVC, "
            "decompress to INT8 and dequantize for inference — no fine-tuning or calibration data required."
        ),
        "components": [
            "Learnable affine transformation (FlatQuant-style outlier mitigation)",
            "RTN FP32→INT8 quantization",
            "Weight-to-YUV420 mapping",
            "VVenC / VVC All-Intra encoding (QP-controlled rate)",
        ],
        "codecs_compared": [x.value for x in CodecType],
        "profiles_compared": [x.value for x in CodingProfile],
        "models": list(c.models),
        "baselines": list(c.baselines),
        "defaults": c.__dict__,
        "limitations": LIMITATIONS,
    }


def table_i_compression() -> list[dict[str, str | float | None]]:
    """Table I — LLMCodec vs GPTQ vs FlatQuant at 3-bit and 2-bit."""
    rows = [
        ("LLaMA-3-8B", 16, "FP16", 6.14, 9.45, 72.42),
        ("LLaMA-3-8B", 3, "GPTQ", 9.79, 15.79, 63.97),
        ("LLaMA-3-8B", 3, "FlatQuant", 7.37, 12.54, 67.79),
        ("LLaMA-3-8B", 3, "LLMCodec", 7.61, 12.23, 69.90),
        ("LLaMA-3-8B", 2, "GPTQ", 465.23, 1263.24, 30.25),
        ("LLaMA-3-8B", 2, "FlatQuant", 41.15, 104.93, 33.98),
        ("LLaMA-3-8B", 2, "LLMCodec", 26.53, 41.86, 52.17),
        ("Qwen-2.5-7B-Instruct", 16, "FP16", 8.35, 14.38, 69.68),
        ("Qwen-2.5-7B-Instruct", 3, "GPTQ", 8.77, 15.40, 64.59),
        ("Qwen-2.5-7B-Instruct", 3, "FlatQuant", 8.68, 15.24, 68.31),
        ("Qwen-2.5-7B-Instruct", 3, "LLMCodec", 8.82, 14.94, 68.65),
        ("Qwen-2.5-7B-Instruct", 2, "GPTQ", 116.35, 1196.20, 30.78),
        ("Qwen-2.5-7B-Instruct", 2, "FlatQuant", 14.76, 36.17, 44.55),
        ("Qwen-2.5-7B-Instruct", 2, "LLMCodec", 14.01, 24.20, 60.53),
        ("LLaMA-2-7B", 16, "FP16", 5.47, 7.26, 68.86),
        ("LLaMA-2-7B", 3, "GPTQ", 6.34, 9.07, 64.41),
        ("LLaMA-2-7B", 3, "FlatQuant", 6.05, 7.86, 65.88),
        ("LLaMA-2-7B", 3, "LLMCodec", 6.33, 8.63, 66.09),
        ("LLaMA-2-7B", 2, "GPTQ", 40.51, 150.15, 31.44),
        ("LLaMA-2-7B", 2, "FlatQuant", 15.08, 38.04, 38.94),
        ("LLaMA-2-7B", 2, "LLMCodec", 50.29, 72.69, 48.90),
    ]
    cols = ("model", "avg_bitwidth", "method", "ppl_wikitext2", "ppl_c4", "avg_accuracy")
    return [dict(zip(cols, r, strict=True)) for r in rows]


def table_ii_ablation() -> list[dict[str, str | float | bool]]:
    """Table II — Outlier removal + quantizer ablation at 2-bit (LLaMA-3-8B)."""
    return [
        {
            "method": "w/o affine + GPTQ",
            "outlier_elimination": False,
            "quantizer": "GPTQ",
            "ppl_wikitext2": 1880.53,
            "avg_accuracy": 32.12,
        },
        {
            "method": "affine + GPTQ",
            "outlier_elimination": True,
            "quantizer": "GPTQ",
            "ppl_wikitext2": 26.60,
            "avg_accuracy": 53.58,
        },
        {
            "method": "LLMCodec (affine + RTN)",
            "outlier_elimination": True,
            "quantizer": "RTN",
            "ppl_wikitext2": 26.53,
            "avg_accuracy": 52.17,
        },
    ]


def fig1_llama3_2bit_anchors() -> dict[str, float]:
    """Fig. 1 rate-distortion anchors at ~2-bit for LLaMA-3-8B."""
    return {
        "LLMCodec_ppl_wikitext2": 26.53,
        "FlatQuant_ppl_wikitext2": 41.15,
        "GPTQ_ppl_wikitext2": 465.23,
        "LLMCodec_avg_accuracy": 52.17,
        "FlatQuant_avg_accuracy": 33.98,
        "GPTQ_avg_accuracy": 30.25,
    }


def codec_psnr_at_2bit() -> list[dict[str, str | float]]:
    """Fig. 4c codec comparison at ~2-bit average (representative PSNR)."""
    return [
        {"codec": "VVC/H.266", "psnr_db": 38.5, "avg_bitwidth": 2.0},
        {"codec": "HEVC/H.265", "psnr_db": 36.2, "avg_bitwidth": 2.0},
        {"codec": "WebP", "psnr_db": 32.8, "avg_bitwidth": 2.0},
        {"codec": "JPEG", "psnr_db": 31.1, "avg_bitwidth": 2.0},
    ]


def profile_psnr_at_qp12() -> list[dict[str, str | float]]:
    """Fig. 4b — All-Intra beats inter profiles at QP=12."""
    return [
        {"profile": "All-Intra", "psnr_y_db": 58.2},
        {"profile": "Low-Delay", "psnr_y_db": 55.1},
        {"profile": "Random-Access", "psnr_y_db": 54.3},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i": table_i_compression(),
        "table_ii": table_ii_ablation(),
        "fig1_llama3_2bit": fig1_llama3_2bit_anchors(),
        "codec_compare": codec_psnr_at_2bit(),
        "profile_compare": profile_psnr_at_qp12(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "smoke": evaluation_smoke(seed=seed),
        "benchmarks": benchmarks_bundle(),
    }
