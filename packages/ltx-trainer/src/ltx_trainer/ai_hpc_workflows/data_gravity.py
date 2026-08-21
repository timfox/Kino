"""Data gravity and I/O planning (Tips 2, 11)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.constants import IO_FORMATS


def data_gravity_assessment(
    *,
    dataset_gb: float,
    transfers_per_epoch: int,
    compute_hours_per_epoch: float,
    scratch_gb: float,
) -> dict[str, Any]:
    """Heuristic: if data moves more than it is computed on, redesign (Tip 2)."""
    transfer_volume = dataset_gb * transfers_per_epoch
    inefficient = transfers_per_epoch > 1 and transfer_volume > compute_hours_per_epoch * dataset_gb * 0.1
    recommendations: list[str] = []
    if transfers_per_epoch > 2:
        recommendations.append("stage dataset on node-local scratch once per run")
    if inefficient:
        recommendations.append("bring compute to data; reduce cross-step transfers")
    if scratch_gb < dataset_gb:
        recommendations.append("increase scratch allocation or shard dataset locally")
    return {
        "transfer_volume_gb_equiv": round(transfer_volume, 2),
        "compute_hours_per_epoch": compute_hours_per_epoch,
        "inefficient_by_heuristic": inefficient,
        "recommendations": recommendations,
    }


def io_mitigation_plan(
    *,
    small_file_count: int,
    checkpoint_count: int,
    concurrent_writers: int,
) -> dict[str, Any]:
    """Tip 11 mitigation strategies for parallel filesystem contention."""
    actions: list[str] = []
    if small_file_count > 1000:
        actions.append("aggregate logs/checkpoints into tar or HDF5 bundles")
    if checkpoint_count > 10:
        actions.append("retain rolling checkpoint window; delete stale shards on scratch")
    if concurrent_writers > 4:
        actions.append("stagger simulation output and AI ingestion via workflow engine")
    if not actions:
        actions.append("current I/O pattern acceptable; monitor metadata ops")
    return {
        "small_file_count": small_file_count,
        "checkpoint_count": checkpoint_count,
        "concurrent_writers": concurrent_writers,
        "formats": list(IO_FORMATS),
        "actions": actions,
    }
