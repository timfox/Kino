"""Paper tables 1–5 and summary anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.arch import ARCHITECTURES, table_2_architectures
from ltx_trainer.fp8_ozaki.constants import (
    B300_NATIVE_GEMM_H100_RATIO,
    TABLE3_SPEEDUP_ANCHORS,
    TABLE4_H100_RELATIVE_GEMM,
    TABLE5_FP8_ADVANTAGE,
    WORKLOAD_OI,
)
from ltx_trainer.fp8_ozaki.tme import (
    emulated_throughput_tflops,
    native_throughput_tflops,
    ozaki_i_slice_count,
    ozaki_speedup,
)
from ltx_trainer.fp8_ozaki.tme import EmulationParams


def table_1_ozaki_i_slices() -> list[dict[str, Any]]:
    """Table 1: Ozaki I slice counts from accumulator bound (3)."""
    rows: list[dict[str, Any]] = []
    for k in (256, 1024, 4096, 16384):
        for substrate in ("fp16", "int8", "fp8"):
            rows.append(
                {
                    "substrate": substrate,
                    "k": k,
                    "slice_count": ozaki_i_slice_count(substrate, k),
                }
            )
    return rows


def table_3_speedups(*, moduli_count: int = 10) -> list[dict[str, Any]]:
    """Table 3: Ozaki II/fp8 speedup vs native fp64 (model + anchors)."""
    rows: list[dict[str, Any]] = []
    for workload, oi in WORKLOAD_OI.items():
        row: dict[str, Any] = {"workload": workload, "oi": oi}
        for gpu in ("H100", "B200", "B300", "R200"):
            model = round(ozaki_speedup(oi, gpu, moduli_count=moduli_count), 2)  # type: ignore[arg-type]
            anchor = TABLE3_SPEEDUP_ANCHORS.get(workload, {}).get(gpu)
            row[f"{gpu}_model"] = model
            if anchor is not None:
                row[f"{gpu}_anchor"] = anchor
        rows.append(row)
    return rows


def table_4_h100_baseline(*, moduli_count: int = 10) -> list[dict[str, Any]]:
    """Table 4: absolute fp64-equivalent TFLOPS relative to H100."""
    h100 = ARCHITECTURES["H100"]
    rows: list[dict[str, Any]] = []
    for workload, oi in WORKLOAD_OI.items():
        h100_nat = native_throughput_tflops(oi, h100)
        for gpu in ("H100", "B200", "B300", "R200"):
            arch = ARCHITECTURES[gpu]  # type: ignore[index]
            params = EmulationParams(alpha=moduli_count, beta=1.0)
            nat = native_throughput_tflops(oi, arch)
            emu = emulated_throughput_tflops(oi, arch, params)
            rows.append(
                {
                    "workload": workload,
                    "gpu": gpu,
                    "path": "ozaki_ii",
                    "native_tflops": round(nat, 2),
                    "ozaki_tflops": round(emu, 2),
                    "vs_h100_native": round(emu / h100_nat, 2) if h100_nat else None,
                }
            )
    return rows


def table_5_substrate_comparison(*, moduli_count: int = 10) -> list[dict[str, Any]]:
    """Table 5: INT8 vs FP8 Ozaki II ceilings at r=10."""
    rows: list[dict[str, Any]] = []
    for gpu in ("H100", "B200", "B300", "R200"):
        arch = ARCHITECTURES[gpu]  # type: ignore[index]
        int8_ceiling = arch.int8_tensor_tops / moduli_count
        fp8_ceiling = arch.fp8_tensor_tflops / moduli_count
        advantage = fp8_ceiling / int8_ceiling if int8_ceiling > 0 else float("inf")
        rows.append(
            {
                "gpu": gpu,
                "int8_ceiling_tflops": round(int8_ceiling, 1),
                "fp8_ceiling_tflops": round(fp8_ceiling, 1),
                "fp8_advantage_model": round(advantage, 1),
                "fp8_advantage_anchor": TABLE5_FP8_ADVANTAGE[gpu],
            }
        )
    return rows


def summary_anchors() -> dict[str, Any]:
    b300 = ARCHITECTURES["B300"]
    params = EmulationParams(alpha=10, beta=1.0)
    return {
        "thesis": "fp8 + Ozaki II restores fp64 across HPC kernel spectrum on fp64-starved GPUs",
        "tme_parameters": {"alpha": "moduli count r", "beta": "bandwidth multiplier", "gamma": "Garner latency"},
        "b300_native_fp64_tflops": b300.fp64_native_tflops,
        "b300_ozaki_ceiling_tflops": round(b300.fp8_tensor_tflops / 10, 0),
        "b300_memory_ridge_oi": round(b300.memory_ridge_oi, 2),
        "b300_stencil_speedup_anchor": 3.1,
        "b300_gemm_speedup_anchor": 380.0,
        "b300_native_gemm_vs_h100": B300_NATIVE_GEMM_H100_RATIO,
        "table4_gemm_h100_relative": TABLE4_H100_RELATIVE_GEMM,
        "rubin_emulated_dgemm_published_tflops": 200.0,
        "register_fusion_required": "β → 1 for memory-bound kernels",
        "companion_fft": "Kulisch INT32 Phase B (Part 2)",
        "cublas_ozaki_integration": "October 2025 [26]",
    }


def tme_figure_1_samples() -> dict[str, Any]:
    """Roofline samples for Figure 1 (B300 + R200)."""
    from ltx_trainer.fp8_ozaki.tme import roofline_curve

    oi = (0.1, 0.2, 0.5, 1.0, 1.5, 4.0, 10.0, 50.0, 100.0)
    return {
        "B300": roofline_curve("B300", oi),
        "R200": roofline_curve("R200", oi),
        "B200_native_reference_tflops": ARCHITECTURES["B200"].fp64_native_tflops,
    }
