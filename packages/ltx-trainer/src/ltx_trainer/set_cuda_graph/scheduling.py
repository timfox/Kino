"""SET runtime: job submitter, dispatcher, callback (§4, Algorithms 1–3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphJob:
    job_id: int
    worker_id: int | None = None
    is_stolen: bool = False


@dataclass
class WorkerState:
    worker_id: int
    stream_id: int
    queue: list[GraphJob] = field(default_factory=list)


def find_job(
    worker_id: int,
    queues: list[list[GraphJob]],
) -> GraphJob | None:
    """Algorithm 2 FindJob: local pop then round-robin steal."""
    if worker_id < 0 or worker_id >= len(queues):
        raise IndexError("worker_id out of range")
    local = queues[worker_id]
    if local:
        job = local.pop(0)
        job.is_stolen = False
        return job
    b = len(queues)
    for k in range(1, b):
        victim_id = (worker_id + k) % b
        victim = queues[victim_id]
        if victim:
            job = victim.pop(0)
            job.is_stolen = True
            return job
    return None


def submit_job(
    queues: list[list[GraphJob]],
    *,
    job_id: int,
    target_worker: int,
    queue_capacity: int,
) -> bool:
    """Algorithm 1: enqueue if slot available."""
    if target_worker < 0 or target_worker >= len(queues):
        raise IndexError("target_worker out of range")
    if len(queues[target_worker]) >= queue_capacity:
        return False
    queues[target_worker].append(GraphJob(job_id=job_id, worker_id=target_worker))
    return True


def callback_return_worker(
    free_pool: list[int],
    worker_id: int,
    done_count: list[int],
) -> None:
    """Algorithm 3: increment cdone, return worker to W_pool."""
    done_count[0] += 1
    free_pool.append(worker_id)


def runtime_components_card() -> dict[str, Any]:
    return {
        "workers": "b workers: stream S_i, graph executable G_i, buffers M_i",
        "per_worker_queues": "Q_i store prepared graph executables (not bare indices)",
        "free_worker_pool": "W_pool updated on completion only (no polling)",
        "submitter": "producer — monitors slots, UpdateGraphParams, push to Q_i",
        "dispatcher": "consumer — pop W_pool, FindJob, LaunchJob + stream callback",
        "work_stealing": "round-robin peer steal with JIT buffer retarget on stolen jobs",
        "sync_overhead": "O(1) shared resources via callback events vs O(b) queue queries",
    }


def algorithms_summary() -> list[dict[str, str]]:
    return [
        {"id": "Alg. 1", "name": "Job Submitter", "trigger": "queue slot free"},
        {"id": "Alg. 2", "name": "Job Dispatch & Launch", "trigger": "worker in W_pool"},
        {"id": "Alg. 3", "name": "Async Resource Return", "trigger": "stream callback on completion"},
    ]
