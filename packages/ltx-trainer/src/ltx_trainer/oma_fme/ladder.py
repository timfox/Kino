"""Bitrate ladder and encode job planning (Sec. 3.2–3.5)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.oma_fme.config import QPS, RESOLUTIONS
from ltx_trainer.oma_fme.variants import SharingStrategy


@dataclass(frozen=True)
class EncodeJob:
    resolution: str
    qp: int
    is_anchor: bool
    reuse_from: str | None  # prior job key or None for full RDO


def _job_key(resolution: str, qp: int) -> str:
    return f"{resolution}:qp{qp}"


def crc_encode_plan(
    *,
    anchor_qp: int = 22,
    strategy_resolutions: tuple[str, ...] = ("HD", "4K", "8K"),
) -> list[EncodeJob]:
    """CRC: one anchor per resolution seeds dependents and next resolution."""
    jobs: list[EncodeJob] = []
    prev_anchor_key: str | None = None
    for res in strategy_resolutions:
        anchor_key = _job_key(res, anchor_qp)
        jobs.append(EncodeJob(res, anchor_qp, True, prev_anchor_key if prev_anchor_key else None))
        for qp in QPS:
            if qp == anchor_qp:
                continue
            jobs.append(EncodeJob(res, qp, False, anchor_key))
        prev_anchor_key = anchor_key
    return jobs


def pra_encode_plan(
    *,
    anchor_qp: int = 22,
    strategy_resolutions: tuple[str, ...] = ("HD", "4K", "8K"),
) -> list[EncodeJob]:
    """PRA: independent anchor per resolution; dependents reuse same-res anchor only."""
    jobs: list[EncodeJob] = []
    for res in strategy_resolutions:
        anchor_key = _job_key(res, anchor_qp)
        jobs.append(EncodeJob(res, anchor_qp, True, None))
        for qp in QPS:
            if qp == anchor_qp:
                continue
            jobs.append(EncodeJob(res, qp, False, anchor_key))
    return jobs


def plan_for_strategy(strategy: SharingStrategy, anchor_qp: int = 22) -> list[EncodeJob]:
    if strategy == SharingStrategy.CRC:
        return crc_encode_plan(anchor_qp=anchor_qp)
    return pra_encode_plan(anchor_qp=anchor_qp)


def ladder_representation_count(
    *,
    projection_tiles: int = 1,
    strategy: SharingStrategy = SharingStrategy.CRC,
) -> int:
    """Total x265 encodes = tiles × jobs in plan."""
    n_jobs = len(plan_for_strategy(strategy))
    return projection_tiles * n_jobs


def resolution_pixels(name: str) -> int:
    h, w = RESOLUTIONS[name]
    return h * w
