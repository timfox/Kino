"""End-to-end FDTD+CPML multi-GPU communication demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.benchmarks import summary_anchors, table_10_single_gpu, table_i_positioning
from ltx_trainer.fdtd_cpml_multigpu.communication import compare_exchange, enlarged_ghost_sweep, estimate_runtime_stub
from ltx_trainer.fdtd_cpml_multigpu.config import FdtdCpmlConfig
from ltx_trainer.fdtd_cpml_multigpu.cpml import cpml_card
from ltx_trainer.fdtd_cpml_multigpu.decomposition import per_gpu_memory_gib, select_best_decomposition
from ltx_trainer.fdtd_cpml_multigpu.stencil import throughput_mpoints_per_sec

TABLE13_KEYS = {320, 480, 544, 640, 800}


def run_demo(cfg: FdtdCpmlConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FdtdCpmlConfig()
    n = cfg.nx
    rt = estimate_runtime_stub(
        cfg.nx,
        cfg.ny,
        cfg.nz,
        cfg.n_steps,
        n_gpus=cfg.n_gpus,
        exchange=cfg.exchange,
        comm_interval=cfg.comm_interval,
    )
    return {
        "decomposition": select_best_decomposition(n if n in (320, 544, 800) else 800),
        "cpml": cpml_card(),
        "exchange": compare_exchange(n if n in TABLE13_KEYS else 800),
        "enlarged_ghost": enlarged_ghost_sweep(n if n in (800,) else 800),
        "runtime_stub": {
            "runtime_s": round(rt, 3),
            "mpoints_s": round(throughput_mpoints_per_sec(cfg.nx, cfg.ny, cfg.nz, cfg.n_steps, rt), 2),
            "exchange": cfg.exchange,
            "comm_interval_s": cfg.comm_interval,
        },
        "memory_per_gpu_gib": round(per_gpu_memory_gib(cfg.nx, cfg.ny, cfg.nz, cfg.n_gpus), 2),
        "single_gpu_baseline": table_10_single_gpu(),
        "positioning": table_i_positioning(),
        "summary": summary_anchors(),
    }
