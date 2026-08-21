"""Hyperparameters for Echo-Infinity (Bian et al., arXiv:2606.04527)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchoInfinityConfig:
    """Wan2.1-T2V-1.3B defaults from paper §4.1 and §B."""

    paper_arxiv: str = "arXiv:2606.04527"
    project_page: str = "https://echo-team-joy-future-academy-jd.github.io/Echo-Infinity/"
    code_url: str = "https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Infinity"

    base_model: str = "Wan2.1-T2V-1.3B"
    resolution: tuple[int, int] = (832, 480)
    fps: int = 16
    clip_seconds: float = 5.0

    # Three-tier KV (§3.2)
    num_sink_frames: int = 3  # NS
    local_window_frames: int = 9  # NW
    chunk_size_frames: int = 3  # B
    num_memory_query_frames: int = 3  # NQ
    tokens_per_frame: int = 1560  # S (latent tokens per frame)
    fmax: int = 20  # pretrained max temporal RoPE id

    # Memory encoder (§B)
    memory_encoder_layers: int = 2  # Lenc
    hidden_dim: int = 1536
    num_heads: int = 12
    head_dim: int = 128
    gate_bias: float = 2.0  # σ(g) ≈ 0.88 at step 0

    # Training (§B)
    stage1_iterations: int = 400
    stage2_iterations: int = 3000
    lr_fake: float = 1e-5
    lr_student: float = 2e-6
    lr_memory_multiplier: float = 5.0
    ema_decay: float = 0.99
    ema_start_step: int = 200
    global_batch_size: int = 64

    # Throughput anchor (Tab. 1)
    throughput_fps: float = 18.5
    throughput_overhead_pct: float = 10.6

    # Inference flags
    enable_memory_update: bool = True
    detach_memory_across_subclips: bool = True  # stage-2: detach Q across 5s boundaries
