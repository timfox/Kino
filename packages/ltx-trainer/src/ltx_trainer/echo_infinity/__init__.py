"""Echo-Infinity: learnable evolving memory for real-time infinite video generation (arXiv:2606.04527).

Reference: Memory Queries + Unified Relative RoPE on causal Wan-class DiT with DMD training.
Upstream code: https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Infinity
"""

from ltx_trainer.echo_infinity.config import EchoInfinityConfig
from ltx_trainer.echo_infinity.inference import CausalChunkDiT, echo_chunk_step, rollout_chunks
from ltx_trainer.echo_infinity.kv_cache import constant_memory_budget_tokens
from ltx_trainer.echo_infinity.ltx_plan import ltx_integration_plan
from ltx_trainer.echo_infinity.memory import (
    CrossAttentionMemoryEncoder,
    GatedMemoryResidual,
    MemoryQueryStack,
    memory_token_count,
    update_memory_queries,
)
from ltx_trainer.echo_infinity.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_240s,
    table_ablation_extended,
    table_interactive_60s,
    table_long_vbench_30_240,
    table_short_vbench_5s,
    training_step_demo,
)
from ltx_trainer.echo_infinity.relative_rope import (
    RoPELayout,
    compute_rope_layout,
    layout_for_step,
    verify_layout_in_range,
)

__all__ = [
    "CausalChunkDiT",
    "CrossAttentionMemoryEncoder",
    "EchoInfinityConfig",
    "GatedMemoryResidual",
    "MemoryQueryStack",
    "RoPELayout",
    "compute_rope_layout",
    "constant_memory_budget_tokens",
    "echo_chunk_step",
    "evaluation_demo",
    "framework_card",
    "layout_for_step",
    "ltx_integration_plan",
    "memory_token_count",
    "rollout_chunks",
    "table_ablation_240s",
    "table_ablation_extended",
    "table_interactive_60s",
    "table_long_vbench_30_240",
    "table_short_vbench_5s",
    "training_step_demo",
    "update_memory_queries",
    "verify_layout_in_range",
]
