"""Toy SET dispatch loop demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.benchmarks import summary_anchors, table_1_averages
from ltx_trainer.set_cuda_graph.config import SetCudaGraphConfig
from ltx_trainer.set_cuda_graph.overhead import decompose_measured
from ltx_trainer.set_cuda_graph.scheduling import (
    GraphJob,
    callback_return_worker,
    find_job,
    submit_job,
)


def run_demo(cfg: SetCudaGraphConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SetCudaGraphConfig()
    b = cfg.num_workers
    queues: list[list[GraphJob]] = [[] for _ in range(b)]
    free_pool = list(range(b))
    done = [0]

    # Seed one job per worker queue.
    for i in range(b):
        submit_job(queues, job_id=i, target_worker=i, queue_capacity=cfg.queue_capacity)

    dispatches = 0
    while free_pool and dispatches < b * 2:
        wid = free_pool.pop(0)
        job = find_job(wid, queues)
        if job is None:
            free_pool.append(wid)
            break
        if job.is_stolen:
            job.worker_id = wid
        callback_return_worker(free_pool, wid, done)
        dispatches += 1
        submit_job(queues, job_id=100 + dispatches, target_worker=wid, queue_capacity=cfg.queue_capacity)

    overhead = decompose_measured(
        batch_size=cfg.batch_size,
        tin=0.05,
        tk=1.0,
        tout=0.03,
        tin_in=0.01,
        tin_k=0.02,
        delta_tk=0.05,
        tk_out=0.01,
        t_inter=0.08 if cfg.batch_size >= 64 else 0.02,
    )
    avg = table_1_averages()
    return {
        "config": {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "gpu": cfg.gpu,
        },
        "dispatch_demo": {
            "jobs_completed": done[0],
            "dispatches": dispatches,
            "free_pool_len": len(free_pool),
        },
        "overhead_stub": overhead,
        "set_beats_batching": avg["Batching"]["combined"] >= 1.15,
        "summary": summary_anchors(),
    }
