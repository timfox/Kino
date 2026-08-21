"""Green AI / LLM energy models (Sec. 7)."""

from __future__ import annotations

from ltx_trainer.energy_aware.metrics import joules_per_token, tokens_per_watt


def estimate_training_cluster_mw(num_gpus: int, tdp_w: float, *, utilization: float = 0.85) -> float:
    """Sustained MW for GPU cluster (paper: 100k GPUs → 100–150 MW)."""
    return num_gpus * tdp_w * utilization / 1e6


def estimate_llm_inference_joules(
    *,
    prefill_w: float,
    prefill_s: float,
    decode_w: float,
    decode_s_per_token: float,
    num_tokens: int,
) -> dict[str, float]:
    """Prefill (compute-bound) + decode (memory-bound) phases (Sec. 7.2)."""
    prefill_j = prefill_w * prefill_s
    decode_j = decode_w * decode_s_per_token * max(num_tokens, 0)
    total = prefill_j + decode_j
    return {
        "prefill_j": prefill_j,
        "decode_j": decode_j,
        "total_j": total,
        "joules_per_token": joules_per_token(total, max(num_tokens, 1)),
    }


def carbon_elastic_workers(
    base_workers: int,
    carbon_intensity_ratio: float,
    *,
    min_workers: int = 1,
) -> int:
    """
    Carbon elasticity: scale active workers with grid carbon (Sec. 7.1).

    ``carbon_intensity_ratio`` = current / baseline (1.0 = normal).
    """
    if carbon_intensity_ratio <= 0:
        return base_workers
    scaled = int(round(base_workers / carbon_intensity_ratio))
    return max(min_workers, scaled)


def rag_kv_cache_retention_bonus(
    cache_hit: bool,
    recompute_joules: float,
    *,
    retention_factor: float = 0.9,
) -> float:
    """Carbon-aware prompt caching avoids recompute (Sec. 7.2)."""
    if cache_hit:
        return recompute_joules * retention_factor
    return 0.0


def training_tokens_per_watt(tokens_per_second: float, cluster_watts: float) -> float:
    return tokens_per_watt(tokens_per_second, cluster_watts)
