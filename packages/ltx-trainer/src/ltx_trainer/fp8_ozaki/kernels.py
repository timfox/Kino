"""Memory-bound kernel strategies (§5): GEMV, stencil, SpMV."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.arch import ARCHITECTURES, GpuId
from ltx_trainer.fp8_ozaki.tme import EmulationParams, emulated_throughput_tflops, native_throughput_tflops


def register_fusion_card() -> dict[str, Any]:
    """β → 1 discipline: residue planes stay register-resident."""
    return {
        "technique": "register_level_fusion",
        "beta_target": 1.0,
        "steps": [
            "load fp64 tile to shared memory",
            "decompose residues in registers per modulus",
            "issue r wmma/tcgen05 MMAs per plane",
            "Garner reconstruct before HBM store",
        ],
        "beta_unfused": "r (each residue plane materialised in HBM)",
        "gamma_amortised": "O(r²/k) negligible for k ≳ 100",
    }


def batched_gemv_projection(
    gpu: GpuId,
    batch: int = 8,
    *,
    moduli_count: int = 10,
    beta: float = 1.0,
) -> dict[str, Any]:
    """§5.2 fused Ozaki-II batched GEMV (Algorithm 1)."""
    oi = batch / 2.0  # ≈ B/2 FLOPs/Byte per paper
    arch = ARCHITECTURES[gpu]
    params = EmulationParams(alpha=moduli_count, beta=beta)
    return {
        "kernel": "batched_gemv",
        "algorithm": 1,
        "batch": batch,
        "operational_intensity": oi,
        "native_tflops": round(native_throughput_tflops(oi, arch), 2),
        "ozaki_tflops": round(emulated_throughput_tflops(oi, arch, params), 2),
        "register_budget_note": "B limited to 4–8 before spilling raises β",
        "tensor_core_shape": "16×16×16 MMA on n-dimension",
    }


def stencil_7pt_projection(
    gpu: GpuId,
    *,
    moduli_count: int = 10,
    beta: float = 1.0,
) -> dict[str, Any]:
    """§5.3 im2col 7-point stencil (Algorithm 2)."""
    oi = 0.5
    arch = ARCHITECTURES[gpu]
    params = EmulationParams(alpha=moduli_count, beta=beta)
    return {
        "kernel": "stencil_7pt",
        "algorithm": 2,
        "operational_intensity": oi,
        "mapping": "im2col → c · U_im2col (7 × N_tile)",
        "coefficients": "pre-decomposed {c(i)} in constant memory",
        "native_tflops": round(native_throughput_tflops(oi, arch), 2),
        "ozaki_tflops": round(emulated_throughput_tflops(oi, arch, params), 2),
        "hbm_bytes_per_output": 24,
        "paper_speedup_b300": 3.1,
    }


def spmv_blocked_ell_projection(
    gpu: GpuId,
    *,
    moduli_count: int = 10,
    padding_ratio: float = 1.0,
) -> dict[str, Any]:
    """§5.4 Blocked-ELL SpMV (Algorithm 3); β inherits padding."""
    oi = 0.2 / padding_ratio
    beta = padding_ratio
    arch = ARCHITECTURES[gpu]
    params = EmulationParams(alpha=moduli_count, beta=beta)
    return {
        "kernel": "spmv_blocked_ell",
        "algorithm": 3,
        "operational_intensity_structural": 0.2,
        "effective_oi": round(oi, 3),
        "padding_ratio": padding_ratio,
        "beta": beta,
        "native_tflops": round(native_throughput_tflops(0.2, arch), 2),
        "ozaki_tflops": round(emulated_throughput_tflops(oi, arch, params), 2),
        "block_width": "bw ∈ {16, 32} for tensor-core k-dim",
        "hybrid_note": "CSR-ELL/HYB when row density heterogeneous",
    }


def kernel_coverage_audit() -> list[dict[str, str]]:
    """§7.1 residual-kernel audit summary."""
    return [
        {"category": "dense_gemm", "path": "ozaki_ii_fp8", "binding_on_b300": False},
        {"category": "batched_gemv", "path": "ozaki_ii_fp8", "binding_on_b300": False},
        {"category": "stencil", "path": "ozaki_ii_fp8", "binding_on_b300": False},
        {"category": "spmv", "path": "ozaki_ii_fp8", "binding_on_b300": False},
        {"category": "fft_3d", "path": "kulisch_int32_phase_b", "binding_on_b300": "conditional"},
        {"category": "blas1_reductions", "path": "fp32_kahan", "binding_on_b300": False},
        {"category": "triangular_panel", "path": "ozaki_outer_dgemm", "binding_on_b300": False},
        {"category": "small_inner_kernel", "path": "latency_bound", "binding_on_b300": "gpu_agnostic"},
    ]
