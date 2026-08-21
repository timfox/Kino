"""Framework card, evaluation demo, and smoke entry points."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.benchmarks import benchmarks_bundle, table2_webgpu_rows
from ltx_trainer.torch_webgpu.config import TorchWebGPUConfig
from ltx_trainer.torch_webgpu.crossover import crossover_demo
from ltx_trainer.torch_webgpu.dispatch import dispatch_demo
from ltx_trainer.torch_webgpu.fx_graph import fx_graph_demo
from ltx_trainer.torch_webgpu.fusion import fusion_demo
from ltx_trainer.torch_webgpu.integration import integration_bundle
from ltx_trainer.torch_webgpu.overhead import overhead_demo


def framework_card(cfg: TorchWebGPUConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TorchWebGPUConfig()
    return {
        "paper": benchmarks_bundle()["paper"],
        "method": {
            "backend": "torch-webgpu PrivateUse1 + Dawn/WGSL",
            "measurement": "sequential-dispatch profiler (not single-op sync)",
            "distinction": "per-dispatch (~24–36 μs) vs per-operation (~95 μs incl. Python)",
            "optimization": "kernel fusion 876→564 dispatches (+53% tok/s Vulkan)",
            "models": "Qwen2.5-0.5B / 1.5B autoregressive batch=1",
        },
        "config": cfg.__dict__,
        "integration": integration_bundle(),
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    webgpu = table2_webgpu_rows()
    return {
        "dispatch": dispatch_demo(seed=seed),
        "overhead": overhead_demo(),
        "fusion": fusion_demo(),
        "fx_graph": fx_graph_demo(),
        "crossover": crossover_demo(),
        "ref_05b_tok_s": next(r["tok_s"] for r in webgpu if "0.5B" in r["model"] and "fused" in r["backend"]),
        "ref_per_operation_us": 95.0,
        "ref_fusion_gain_pct": 53,
    }


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "package": "torch_webgpu",
        "paper": "torch_webgpu",
        "arxiv": "2604.02344",
        "ref_05b_tok_s": 21.0,
        "ref_per_operation_us": demo["ref_per_operation_us"],
        "sequential_dispatch_us": demo["dispatch"]["sequential_per_dispatch_us"],
        "fusion_gain_pct": demo["ref_fusion_gain_pct"],
        "gopex_stub_count": len(integration_bundle()["gopex_stubs"]),
    }
