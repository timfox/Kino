"""DEMON: Diffusion Engine for Musical Orchestrated Noise (arXiv:2605.28657)."""

from ltx_trainer.demon.config import DemonConfig, PropagationClass
from ltx_trainer.demon.layout import LIMITATIONS
from ltx_trainer.demon.metrics import (
    table_12_propagation,
    table_13_ablation,
    table_14_depth_tradeoff,
    table_2_per_frame_curves,
    table_3_system_comparison,
)
from ltx_trainer.demon.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
)
from ltx_trainer.demon.ring_buffer import ablation_completion_rates, propagation_taxonomy, ring_smoke
from ltx_trainer.demon.sde import (
    PER_FRAME_CURVES,
    ramp_curve,
    sde_renoise_blend,
    sde_smoke,
    segment_gradient,
    segment_source_similarities,
)
from ltx_trainer.demon.stream_sim import ablation_from_simulation, run_stream, stream_smoke
from ltx_trainer.demon.vae_window import vae_smoke, window_bounds, windowed_decode_latency_ms

__all__ = [
    "DemonConfig",
    "LIMITATIONS",
    "PER_FRAME_CURVES",
    "PropagationClass",
    "ablation_completion_rates",
    "ablation_from_simulation",
    "benchmarks_bundle",
    "evaluation_demo",
    "framework_card",
    "headline_results",
    "pipeline_demo",
    "propagation_taxonomy",
    "ramp_curve",
    "ring_smoke",
    "run_stream",
    "sde_renoise_blend",
    "sde_smoke",
    "segment_gradient",
    "segment_source_similarities",
    "stream_smoke",
    "table_12_propagation",
    "table_13_ablation",
    "table_14_depth_tradeoff",
    "table_2_per_frame_curves",
    "table_3_system_comparison",
    "vae_smoke",
    "window_bounds",
    "windowed_decode_latency_ms",
]
