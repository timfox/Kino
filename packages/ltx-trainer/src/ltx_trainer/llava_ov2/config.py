"""LLaVA-OneVision-2 codec-aligned MLLM (Liang et al., arXiv:2605.25979)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLaVAOV2Config:
    """Defaults from paper Sec. 2–5."""

    backbone_encoder: str = "OneVision-Encoder"
    llm: str = "Qwen3-8B"
    native_resolution: int = 392
    patch_size: int = 14
    merge_size: int = 2
    tokens_per_frame: int = 196

    # Codec-stream (Sec. 2.2)
    bin_duration_s: float = 0.5
    gop_min_s: float = 2.0
    gop_max_s: float = 8.0
    target_gop_count: int = 13
    block_patch_size: int = 16
    same_frame_attenuation: float = 0.5

    # Training recipe (Sec. 4–5)
    frame_budgets: tuple[int, ...] = (30, 90, 384, 768)
    mixed_batch_codec: float = 0.50
    mixed_batch_uniform_video: float = 0.375
    mixed_batch_image: float = 0.125

    video_caption_clips_m: float = 8.0
    spatial_corpus_m: float = 4.0
    jumpscore_clips: int = 189

    hidden_dim: int = 3584
