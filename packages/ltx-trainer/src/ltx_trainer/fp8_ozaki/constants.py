"""Paper anchors and architectural constants (Matsuoka, arXiv:2606.06510)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06510"
PAPER_TITLE = (
    "FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_AUTHOR = "Satoshi Matsuoka"
PAPER_AFFILIATION = "RIKEN Center for Computational Science (R-CCS)"

# Default Ozaki II FP8 parameter set (optimistic; paper §2.4 notes r≥11 empirically).
DEFAULT_MODULI_COUNT = 10
DEFAULT_BANDWIDTH_MULTIPLIER = 1.0  # register-fused β → 1

# Table 3 workload operational intensities (FLOPs / Byte).
WORKLOAD_OI: dict[str, float] = {
    "dense_gemm": 50.0,
    "batched_gemv_b8": 4.0,
    "batched_gemv_b2": 1.5,
    "stencil_7pt": 0.5,
    "spmv": 0.2,
}

# Paper Table 3 speedup anchors (Ozaki II/fp8 vs native fp64 on same GPU).
TABLE3_SPEEDUP_ANCHORS: dict[str, dict[str, float]] = {
    "dense_gemm": {"H100": 1.0, "B200": 10.0, "B300": 380.0, "R200_native": 12.0, "R200_emul": 2.0},
    "batched_gemv_b8": {"H100": 1.0, "B200": 1.6, "B300": 24.0, "R200_native": 2.7},
    "batched_gemv_b2": {"H100": 1.0, "B200": 1.0, "B300": 9.2, "R200_native": 1.0},
    "stencil_7pt": {"H100": 1.0, "B200": 1.0, "B300": 3.1, "R200_native": 1.0},
    "spmv": {"H100": 1.0, "B200": 1.0, "B300": 1.2, "R200_native": 1.0},
}

# Table 4 H100-relative Ozaki throughput anchors (dense GEMM).
TABLE4_H100_RELATIVE_GEMM = {
    "H100": 2.96,
    "B200": 6.72,
    "B300": 7.46,
    "R200": 5.97,
}

# B300 native fp64 regression vs H100 (dense GEMM).
B300_NATIVE_GEMM_H100_RATIO = 0.02

# FP8 advantage over INT8 substrate at r=10 (Table 5).
TABLE5_FP8_ADVANTAGE = {"H100": 1.0, "B200": 29.0, "B300": 30.0, "R200": 16.0}

COMPANION_FFT_ARXIV = "2606.06510"  # Part 2 companion (in preparation per paper)
