"""Controlled prototypes: SAS, SAS-contextflow, MAS."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.mas_ib.config import MasIbConfig
from ltx_trainer.mas_ib.ib import evaluate_relay


@dataclass
class WorkerTrace:
    worker_id: int
    subtask: str
    local_context_len: int
    relay_out: str | None = None


@dataclass
class PrototypeRun:
    name: str
    shared_context: bool
    compressed_relays: bool
    workers: list[WorkerTrace] = field(default_factory=list)
    final_score: float = 0.0
    relay_stats: dict | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shared_context": self.shared_context,
            "compressed_relays": self.compressed_relays,
            "n_workers": len(self.workers),
            "final_score": round(self.final_score, 4),
            "relay_stats": self.relay_stats,
            "workers": [
                {
                    "id": w.worker_id,
                    "subtask": w.subtask,
                    "local_context_len": w.local_context_len,
                    "relay_out": w.relay_out,
                }
                for w in self.workers
            ],
        }


def run_sas(subtasks: list[str], base_score: float = 0.45) -> PrototypeRun:
    """Single shared context accumulates all subtasks."""
    ctx = 0
    workers = []
    for i, st in enumerate(subtasks):
        ctx += 10 + len(st)
        workers.append(WorkerTrace(i, st, ctx, relay_out=None))
    # Shared context can hurt weak models (interference) — slightly lower than MAS when δ≈0
    return PrototypeRun("SAS", shared_context=True, compressed_relays=False, workers=workers, final_score=base_score)


def run_sas_contextflow(subtasks: list[str], base_score: float = 0.40) -> PrototypeRun:
    """Same decomposition as MAS but full upstream context visible."""
    ctx = 0
    workers = []
    for i, st in enumerate(subtasks):
        ctx += 10 + len(st)
        workers.append(WorkerTrace(i, st, ctx, relay_out=f"full_ctx_{i}"))
    return PrototypeRun(
        "SAS-contextflow",
        shared_context=True,
        compressed_relays=False,
        workers=workers,
        final_score=base_score,
    )


def run_mas(
    subtasks: list[str],
    config: MasIbConfig | None = None,
    *,
    contextflow_score: float = 0.40,
) -> PrototypeRun:
    """Isolated workers + compressed relays; score shifts by MAS gain sign."""
    cfg = config or MasIbConfig()
    stats = evaluate_relay(cfg)
    workers = []
    relay = ""
    for i, st in enumerate(subtasks):
        local = 8 + len(st) + len(relay)
        out = f"m{i}:{st[:12]}" if cfg.enable_compression else f"full:{st}"
        workers.append(WorkerTrace(i, st, local, relay_out=out))
        relay = out
    # Map continuous gain into score delta vs contextflow
    delta_score = 0.05 * (1.0 if stats.mas_gain > 0 else -1.0) * min(1.0, abs(stats.mas_gain) / 20.0)
    # Stronger β shrinks positive gains
    if stats.mas_gain > 0:
        delta_score *= max(0.2, 1.0 / (1.0 + 0.3 * cfg.beta))
    score = contextflow_score + delta_score
    return PrototypeRun(
        "MAS",
        shared_context=False,
        compressed_relays=cfg.enable_compression,
        workers=workers,
        final_score=max(0.0, min(1.0, score)),
        relay_stats=stats.as_dict(),
    )


def compare_prototypes(
    subtasks: list[str] | None = None,
    config: MasIbConfig | None = None,
) -> dict:
    tasks = subtasks or ["locate", "act", "verify"]
    cfg = config or MasIbConfig()
    sas = run_sas(tasks)
    flow = run_sas_contextflow(tasks)
    mas = run_mas(tasks, cfg, contextflow_score=flow.final_score)
    gain = mas.final_score - flow.final_score
    return {
        "sas": sas.as_dict(),
        "sas_contextflow": flow.as_dict(),
        "mas": mas.as_dict(),
        "mas_minus_contextflow": round(gain, 4),
        "mas_helps": gain > 0,
        "config": {"beta": cfg.beta, "delta_loss": cfg.delta_loss, "relay_bits": cfg.relay_bits},
    }
