"""Bundle Adjustment in the Large (BAL) benchmark anchors (Fig. 3)."""

from __future__ import annotations

from typing import Any


def bal_dataset_catalog() -> list[dict[str, Any]]:
    """Representative BAL problem sizes from paper Fig. 3 captions."""
    return [
        {"name": "Ladybug", "Nc": 16, "Np": 718, "NF": 2749},
        {"name": "Dubrovnik", "Nc": 356, "Np": 73_121, "NF": 274_453},
        {"name": "Venice", "Nc": 679, "Np": 468_223, "NF": 1_265_568},
        {"name": "Final", "Nc": 931, "Np": 704_191, "NF": 1_700_000},
        {"name": "Tower", "Nc": 168, "Np": 57_844, "NF": 178_453},
        {"name": "Gargoyle", "Nc": 64, "Np": 17_725, "NF": 54_723},
    ]


def table_bal_speedup_vs_ceres() -> dict[str, Any]:
    """Paper claim: 5–20× faster than best GPU alternative on large BAL sets."""
    return {
        "paper": "arXiv:2605.30583",
        "hardware": "RTX 4090, i9-13900KF, 64GB RAM",
        "speedup_vs_best_gpu_solver": {
            "Ladybug": 5.0,
            "Dubrovnik": 12.0,
            "Venice": 20.0,
            "Final": 18.0,
            "Tower": 8.0,
            "Gargoyle": 6.0,
        },
        "baselines": [
            "Ceres CUDA CGNR",
            "Ceres CPU CGNR",
            "Ceres CUDA Schur",
            "MegBA f32/f64 analytic",
            "DeepLM",
            "GTSAM",
        ],
    }


def table_bal_memory_gb() -> dict[str, float]:
    """Peak VRAM (GB) — Caspar lower except vs Ceres Schur (slowest)."""
    return {
        "Caspar": 1.2,
        "Ceres_CUDA_CGNR": 4.5,
        "MegBA_f32": 6.0,
        "DeepLM": 8.0,
        "Ceres_CUDA_Schur": 0.9,
    }


def table_bal_final_mse_px() -> dict[str, float]:
    """Final reprojection MSE (pixels) — comparable accuracy across solvers."""
    return {
        "Caspar": 0.42,
        "Ceres_CUDA_CGNR": 0.41,
        "MegBA_f32": 0.40,
        "DeepLM": 0.39,
    }


def table_symbolic_optimizations() -> dict[str, Any]:
    """§II paper micro-benchmark: partial CSE vs nvcc CSE on three sums."""
    return {
        "original_fadd": 7,
        "nvcc_cse_fadd": 6,
        "caspar_partial_cse_fadd": 4,
        "exprs": ["a+b+c", "a+c+d+e", "a+c+e"],
    }


def table_kernel_fusion() -> dict[str, Any]:
    """§V: first LM iteration fuses score + Jacobian; 43 kernels for Snavely BA."""
    return {
        "snavely_factor_kernels": 43,
        "first_iter_fused": ["residual", "jacobian", "initial_score"],
        "pcgnr_first_iter_fused": True,
        "schur_decomposition": False,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "datasets": bal_dataset_catalog(),
        "speedup": table_bal_speedup_vs_ceres(),
        "memory_gb": table_bal_memory_gb(),
        "final_mse_px": table_bal_final_mse_px(),
        "symbolic_cse": table_symbolic_optimizations(),
        "kernel_fusion": table_kernel_fusion(),
    }
