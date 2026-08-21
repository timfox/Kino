"""Mirage / latent spatial memory configuration (arXiv:2606.09828)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MirageConfig:
    name: str = "Mirage"
    title: str = "Latent Spatial Memory for Video World Models"
    paper_arxiv: str = "2606.09828"
    website: str = "https://aka.ms/latent-spatial-memory"
    backbone: str = "Wan2.2-TI2V-5B"
    vae_stride: int = 16
    vae_temporal_stride: int = 4
    latent_channels: int = 48
    rgb_height: int = 704
    rgb_width: int = 1280
    latent_height: int = 44
    latent_width: int = 80
    chunk_latent_frames: int = 9
    chunk_rgb_frames: int = 33
    chunk_overlap_latent_frames: int = 1
    hidden_dim: int = 3072
    ff_dim: int = 14336
    num_heads: int = 24
    num_blocks: int = 30
    text_context_tokens: int = 512
    flow_steps: int = 40
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    text_dropout: float = 0.2
    adamw_beta1: float = 0.0
    adamw_beta2: float = 0.999
    weight_decay: float = 1e-3
    control_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28)
    control_input_channels: int = 48
    segmenter: str = "SAM3"
    entity_extractor: str = "Qwen3-VL-2B"
    stage1_lr: float = 1e-5
    stage2_lr: float = 1e-4
    depth_source: str = "depth_anything_3"
    depth_downsample: str = "bilinear"
    scheduler: str = "unipc"
    guidance_scale: float = 0.0
    speedup_vs_rgb: float = 10.57
    memory_reduction_vs_rgb: float = 55.0
    per_frame_readout_s: float = 0.25
    cache_mib_per_chunk: float = 0.45
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def latent_grid(self) -> tuple[int, int]:
        return self.latent_height, self.latent_width

    @property
    def rgb_grid(self) -> tuple[int, int]:
        return self.rgb_height, self.rgb_width
