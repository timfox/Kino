"""End-to-end TME / Ozaki II demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.arch import table_2_architectures
from ltx_trainer.fp8_ozaki.benchmarks import (
    summary_anchors,
    table_3_speedups,
    table_4_h100_baseline,
    table_5_substrate_comparison,
    tme_figure_1_samples,
)
from ltx_trainer.fp8_ozaki.config import Fp8OzakiConfig
from ltx_trainer.fp8_ozaki.constants import WORKLOAD_OI
from ltx_trainer.fp8_ozaki.kernels import (
    batched_gemv_projection,
    kernel_coverage_audit,
    register_fusion_card,
    spmv_blocked_ell_projection,
    stencil_7pt_projection,
)
from ltx_trainer.fp8_ozaki.ozaki_ii import demo_dot_product_fp64, ozaki_ii_cost_card
from ltx_trainer.fp8_ozaki.tme import ozaki_speedup, roofline_curve


def run_demo(cfg: Fp8OzakiConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Fp8OzakiConfig()
    gpu = cfg.gpu  # type: ignore[assignment]
    r = cfg.moduli_count
    workload_speedups = {
        w: round(ozaki_speedup(WORKLOAD_OI[w], gpu, moduli_count=r, beta=cfg.bandwidth_multiplier), 2)  # type: ignore[arg-type]
        for w in cfg.workloads
        if w in WORKLOAD_OI
    }
    return {
        "gpu": gpu,
        "moduli_count": r,
        "bandwidth_multiplier": cfg.bandwidth_multiplier,
        "substrate": cfg.substrate,
        "architectures": table_2_architectures(),
        "ozaki_ii": ozaki_ii_cost_card(moduli_count=r, substrate=cfg.substrate),
        "register_fusion": register_fusion_card(),
        "kernels": {
            "batched_gemv": batched_gemv_projection(gpu, batch=8, moduli_count=r),  # type: ignore[arg-type]
            "stencil_7pt": stencil_7pt_projection(gpu, moduli_count=r),  # type: ignore[arg-type]
            "spmv": spmv_blocked_ell_projection(gpu, moduli_count=r),  # type: ignore[arg-type]
        },
        "workload_speedups": workload_speedups,
        "roofline": roofline_curve(
            gpu,  # type: ignore[arg-type]
            cfg.roofline_oi_samples,
            moduli_count=r,
            beta=cfg.bandwidth_multiplier,
            substrate=cfg.substrate,
        ),
        "figure_1": tme_figure_1_samples(),
        "tables": {
            "table_3": table_3_speedups(moduli_count=r),
            "table_4": table_4_h100_baseline(moduli_count=r),
            "table_5": table_5_substrate_comparison(moduli_count=r),
        },
        "kernel_audit": kernel_coverage_audit(),
        "dot_product_demo": demo_dot_product_fp64([1.1, 2.2, 3.3], [0.5, 1.5, 2.5], moduli_count=4),
        "summary": summary_anchors(),
    }
