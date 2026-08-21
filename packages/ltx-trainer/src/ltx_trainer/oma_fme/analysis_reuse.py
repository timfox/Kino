"""x265 analysis-reuse encode time model (Sec. 4.1, stub)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.oma_fme.config import RESOLUTIONS
from ltx_trainer.oma_fme.ladder import EncodeJob, plan_for_strategy, resolution_pixels
from ltx_trainer.oma_fme.variants import ProjectionFormat, SharingStrategy


@dataclass
class EncodeTimeModel:
    """Heuristic full-search seconds per megapixel at 8K reference."""

    base_sec_per_mpx: float = 0.012
    anchor_multiplier: float = 1.0
    dependent_multiplier: float = 0.35
    cross_res_multiplier: float = 0.55
    cmp_parallel_divisor: float = 6.0

    def job_seconds(self, job: EncodeJob, *, ref_res: str = "8K") -> float:
        mpx = resolution_pixels(job.resolution) / 1e6
        ref_mpx = resolution_pixels(ref_res) / 1e6
        scale = mpx / ref_mpx
        t = self.base_sec_per_mpx * mpx * 1e6 * scale**0.85
        if job.is_anchor:
            if job.reuse_from is not None:
                t *= self.cross_res_multiplier
            else:
                t *= self.anchor_multiplier
        else:
            t *= self.dependent_multiplier
        return max(t, 0.1)


def simulate_ladder_times(
    strategy: SharingStrategy,
    projection: ProjectionFormat,
    *,
    model: EncodeTimeModel | None = None,
) -> dict[str, float]:
    """Return serial total seconds and parallel max per-tile for one sequence."""
    model = model or EncodeTimeModel()
    tiles = 6 if projection == ProjectionFormat.CMP else 1
    jobs = plan_for_strategy(strategy)
    per_job = [model.job_seconds(j) for j in jobs]
    serial = sum(per_job) * tiles
    # Parallel runtime: slowest job × tiles (faces encode concurrently in CMP).
    parallel_wall = max(per_job) if projection == ProjectionFormat.CMP else max(per_job)
    return {"serial_s": serial, "parallel_max_s": parallel_wall}


def simulate_default_erp_times(model: EncodeTimeModel | None = None) -> dict[str, float]:
    """ERP-Default: full search every QP × resolution, no reuse."""
    model = model or EncodeTimeModel()
    total = 0.0
    peak = 0.0
    from ltx_trainer.oma_fme.config import QPS

    for res in ("HD", "4K", "8K"):
        for qp in QPS:
            job = EncodeJob(res, qp, True, None)
            t = model.job_seconds(job) / model.anchor_multiplier  # full RDO cost
            total += t
            peak = max(peak, t)
    return {"serial_s": total, "parallel_max_s": peak}
