"""Paper benchmark anchors (Table 3–5, Fig. 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gpu_nmpc.config import ResolveMode
from ltx_trainer.gpu_nmpc.nmpc import NmpcRunSummary, run_nmpc_loop


def table3_total_times() -> list[dict[str, Any]]:
    """Total NMPC simulation times (s) — Table 3 anchors."""
    rows: list[dict[str, Any]] = []

    def add(config: str, bench: str, base: float, ws: float, pu: float, ws_pu: float) -> None:
        for mode, val in (
            ("base", base),
            ("warmstart", ws),
            ("param_update", pu),
            ("warmstart_param_update", ws_pu),
        ):
            rows.append(
                {
                    "configuration": config,
                    "benchmark": bench,
                    "mode": mode,
                    "total_time_s": val,
                }
            )

    # InfiniteExaModels + MadNLP + cuDSS (GPU)
    add("InfiniteExaModels+MadNLP+cuDSS", "distillation", 21.91, 37.76, 11.77, 3.00)
    add("InfiniteExaModels+MadNLP+cuDSS", "pde_plate", 37.30, 54.75, 31.63, 3.28)
    # JuMP reverse-mode (CPU) for comparison
    add("JuMP+ReverseAD+Ipopt", "distillation", 58.72, 51.62, 24.52, 19.06)
    add("JuMP+ReverseAD+Ipopt", "pde_plate", 125.41, 88.20, 119.11, 77.33)
    return rows


def per_iteration_solve_ms(config: str, benchmark: str, *, mode: str = "warmstart_param_update") -> float:
    """Table 4/5 per-iteration solve anchors (distillation / plate)."""
    key = (config, benchmark, mode)
    anchors = {
        ("InfiniteExaModels+MadNLP+cuDSS", "distillation", "base"): 130.0,
        ("InfiniteExaModels+MadNLP+cuDSS", "distillation", "warmstart_param_update"): 20.0,
        ("InfiniteExaModels+MadNLP+cuDSS", "pde_plate", "base"): 960.0,
        ("InfiniteExaModels+MadNLP+cuDSS", "pde_plate", "warmstart_param_update"): 60.0,
    }
    return anchors.get(key, 50.0)


def speedup_vs_base(summary: NmpcRunSummary, baseline: NmpcRunSummary) -> float:
    if summary.total_ms <= 0:
        return 1.0
    return baseline.total_ms / summary.total_ms


def stub_timing_ratio(cfg_mode: ResolveMode, benchmark: str) -> dict[str, float]:
    """Run stub NMPC and compare WS+PU to base for the requested benchmark."""
    from ltx_trainer.gpu_nmpc.config import GpuNmpcConfig

    cfg = GpuNmpcConfig()
    base = run_nmpc_loop(benchmark=benchmark, cfg=cfg, mode="base", steps=cfg.demo_nmpc_steps)
    opt = run_nmpc_loop(benchmark=benchmark, cfg=cfg, mode="warmstart_param_update", steps=cfg.demo_nmpc_steps)
    return {
        "base_total_ms": base.total_ms,
        "ws_pu_total_ms": opt.total_ms,
        "speedup": speedup_vs_base(opt, base),
        "solve_fraction_base": base.total_solve_ms / max(base.total_ms, 1e-9),
        "solve_fraction_ws_pu": opt.total_solve_ms / max(opt.total_ms, 1e-9),
    }
