"""LLMCodec video-codec LLM weight compression config (arXiv:2606.05861)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CodecType(str, Enum):
    VVC = "VVC/H.266"
    HEVC = "HEVC/H.265"
    JPEG = "JPEG"
    WEBP = "WebP"


class CodingProfile(str, Enum):
    ALL_INTRA = "All-Intra"
    LOW_DELAY = "Low-Delay"
    RANDOM_ACCESS = "Random-Access"


class YuvLayout(str, Enum):
    YUV420 = "YUV420"
    YUV444 = "YUV444"


@dataclass
class LlmCodecConfig:
    paper_arxiv: str = "arXiv:2606.05861"
    encoder: str = "VVenC"
    default_codec: CodecType = CodecType.VVC
    default_profile: CodingProfile = CodingProfile.ALL_INTRA
    yuv_layout: YuvLayout = YuvLayout.YUV420
    chroma_fill: int = 128
    int_bits: int = 8
    qp_default: int = 12
    qp_range: tuple[int, int] = (0, 20)
    models: tuple[str, ...] = ("LLaMA-3-8B", "LLaMA-2-7B", "Qwen-2.5-7B-Instruct")
    baselines: tuple[str, ...] = ("GPTQ", "FlatQuant")
    benchmarks_wikitext2: str = "WikiText2"
    benchmarks_c4: str = "C4"
    downstream_tasks: tuple[str, ...] = (
        "ARC-Challenge",
        "HellaSwag",
        "LAMBADA",
        "PIQA",
        "WinoGrande",
    )
    affine_source: str = "FlatQuant-style learnable per-layer T"
    weight_quantizer: str = "RTN"
    gpu_note: str = "2× NVIDIA RTX 4090 (paper)"
