"""Framework card, demos, and benchmark manifest (arXiv:2606.04725)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gpu_nmpc.config import GpuNmpcConfig
from ltx_trainer.gpu_nmpc.metrics import stub_timing_ratio, table3_total_times
from ltx_trainer.gpu_nmpc.nmpc import run_nmpc_loop


def framework_card(cfg: GpuNmpcConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GpuNmpcConfig()
    return {
        "name": "GPU-NMPC",
        "paper": cfg.paper_arxiv,
        "task": "Parametric interior-point NMPC with symbolic Cholesky reuse",
        "packages": list(cfg.packages),
        "components": [
            "InfiniteOpt direct transcription → InfiniteSIMD-NLP",
            "Lifted-KKT condensed interior-point (MadNLP + cuDSS)",
            "SymCholesky once; NumCholesky each IP iteration",
            "warmstart_backend_start_values + set_parameter_value re-solves",
        ],
        "benchmarks": ["distillation_column", "pde_heated_plate"],
        "distillation": cfg.distillation.__dict__,
        "plate": cfg.plate.__dict__,
        "upstream_repo": "https://github.com/infiniteopt/GPU-NMPC-paper",
    }


def paper_limitations() -> list[str]:
    return [
        "Julia InfiniteOpt / MadNLP / cuDSS stack not bundled — CPU numpy stub stands in for GPU kernels.",
        "Distillation and PDE dynamics are reduced-order proxies, not full tray/2D PDE discretizations.",
        "Timing model uses unit costs aligned to Table 4/5 ratios, not measured wall clock on RTX 6000 Ada.",
        "MPCGPU and OptimalControl.jl baselines are reference numbers only in metrics.py.",
        "HyKKT linear solver variant documented but not exercised in this stub.",
    ]


def evaluation_demo(cfg: GpuNmpcConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GpuNmpcConfig()
    dist = run_nmpc_loop(benchmark="distillation", cfg=cfg, mode="warmstart_param_update")
    plate = run_nmpc_loop(benchmark="pde_plate", cfg=cfg, mode="warmstart_param_update")
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "table3_rows": len(table3_total_times()),
        "distillation_run": dist.__dict__,
        "pde_plate_run": plate.__dict__,
        "stub_speedups": {
            "distillation": stub_timing_ratio("warmstart_param_update", "distillation"),
            "pde_plate": stub_timing_ratio("warmstart_param_update", "pde_plate"),
        },
    }


def evaluation_smoke(cfg: GpuNmpcConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    dist = demo["distillation_run"]
    assert dist["reused_symbolic_all_but_first"]
    assert dist["total_ms"] > 0
    assert demo["stub_speedups"]["distillation"]["speedup"] > 1.5
    return {"status": "ok", "paper": (cfg or GpuNmpcConfig()).paper_arxiv, "demo_keys": list(demo.keys())}
