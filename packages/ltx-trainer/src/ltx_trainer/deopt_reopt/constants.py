"""Paper anchors — LLM C++→CUDA Deopt-Reopt (Mukunoki et al., arXiv:2606.06063)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06063"
PAPER_TITLE = (
    "LLM-Based Porting of Optimized C++ to CUDA Through Deoptimization and Reoptimization"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
UPSTREAM_REPO = "https://github.com/mukunoki/deopt_reopt"

MODEL_O120 = "gpt-oss-120b"
MODEL_Q235 = "qwen-3-235b-a22b-instruct-2507"

TRIALS_PER_MODE = 50
SINGLE_SHOT_LLM_CALLS = {"direct": 2, "deopt_reopt": 3, "direct_3": 3}
ITERATIVE_GENERATIONS = 3
ITERATIVE_PG_PER_GEN = 3

HARDWARE = "GH200 (72-core Neoverse-V2 + NVIDIA H100)"
CUDA_ARCH = "sm_90"
COMPILER_CPU = "nvc++ -O3 -fopenmp -tp=neoverse-v2 -std=c++17"
COMPILER_CUDA = "nvcc -O3 -arch=sm_90 -std=c++17"

KERNEL_GROUPS = ("divergent", "dominated", "shared")

# Table I — problem sizes, units, tolerances (abbreviated for stub lookups).
TABLE1_KERNELS: tuple[dict[str, str | int | float], ...] = (
    {
        "name": "conv2d",
        "group": "divergent",
        "problem": "N=16,C=32,K=64,H=W=96",
        "unit": "GFlops",
        "input_perf": 2328.0,
        "ref_cpu": 474.9,
        "ref_gpu": 24647.0,
        "tol": 1e-3,
        "loc": 142,
    },
    {
        "name": "bfft",
        "group": "divergent",
        "problem": "B=8192,N=1024",
        "unit": "GFlops",
        "input_perf": 598.9,
        "ref_cpu": 596.4,
        "ref_gpu": 4907.0,
        "tol": 1e-8,
        "loc": 123,
    },
    {
        "name": "softmax",
        "group": "divergent",
        "problem": "M=1024,N=32768",
        "unit": "GB/s",
        "input_perf": 185.6,
        "ref_cpu": 210.3,
        "ref_gpu": 1235.0,
        "tol": 1e-8,
        "loc": 67,
    },
    {
        "name": "bgemm",
        "group": "divergent",
        "problem": "B=1024,M=N=K=128",
        "unit": "GFlops",
        "input_perf": 1935.0,
        "ref_cpu": 159.4,
        "ref_gpu": 32848.0,
        "tol": 1e-10,
        "loc": 210,
    },
    {
        "name": "dfspmm",
        "group": "dominated",
        "problem": "CSR M=K=24576,N=128,32nnz/row",
        "unit": "GFlops",
        "input_perf": 17.3,
        "ref_cpu": 735.3,
        "ref_gpu": 2181.0,
        "tol": 1e-7,
        "loc": 148,
    },
    {
        "name": "fft",
        "group": "dominated",
        "problem": "N=4194304",
        "unit": "GFlops",
        "input_perf": 209.8,
        "ref_cpu": 170.0,
        "ref_gpu": 3154.0,
        "tol": 1e-8,
        "loc": 519,
    },
    {
        "name": "btdma",
        "group": "dominated",
        "problem": "B=32768,N=256",
        "unit": "GB/s",
        "input_perf": 295.7,
        "ref_cpu": 159.4,
        "ref_gpu": 141.1,
        "tol": 1e-8,
        "loc": 92,
    },
    {
        "name": "ddgemm",
        "group": "dominated",
        "problem": "DD M=N=K=256",
        "unit": "GFlops",
        "input_perf": 44.1,
        "ref_cpu": 4.3,
        "ref_gpu": 569.9,
        "tol": 1e-22,
        "loc": 86,
    },
    {
        "name": "stencil",
        "group": "shared",
        "problem": "N=3072,T=100",
        "unit": "GB/s",
        "input_perf": 1404.0,
        "ref_cpu": 970.8,
        "ref_gpu": 2789.0,
        "tol": 1e-8,
        "loc": 116,
    },
    {
        "name": "spmm",
        "group": "shared",
        "problem": "CSR M=98304,K=524288,N=128",
        "unit": "GB/s",
        "input_perf": 414.6,
        "ref_cpu": 203.9,
        "ref_gpu": 752.7,
        "tol": 1e-10,
        "loc": 98,
    },
    {
        "name": "gemm",
        "group": "shared",
        "problem": "M=N=K=1024",
        "unit": "GFlops",
        "input_perf": 305.5,
        "ref_cpu": 2653.0,
        "ref_gpu": 40895.0,
        "tol": 1e-8,
        "loc": 259,
    },
    {
        "name": "spmv",
        "group": "shared",
        "problem": "BSR M=K=524288,BS=4",
        "unit": "GB/s",
        "input_perf": 1508.0,
        "ref_cpu": 685.6,
        "ref_gpu": 2527.0,
        "tol": 1e-10,
        "loc": 63,
    },
)

# Table II — success (%) and median perf over successful trials (paper values).
TABLE2_SINGLE_SHOT: tuple[dict[str, object], ...] = (
    {"kernel": "conv2d", "model": "O120", "d_succ": 64, "dr_succ": 70, "d_med": 1824, "dr_med": 2704, "sig": "dr_bh"},
    {"kernel": "conv2d", "model": "Q235", "d_succ": 12, "dr_succ": 96, "d_med": 106, "dr_med": 4366, "sig": "dr_bh"},
    {"kernel": "bfft", "model": "O120", "d_succ": 48, "dr_succ": 52, "d_med": 228, "dr_med": 164, "sig": None},
    {"kernel": "bfft", "model": "Q235", "d_succ": 12, "dr_succ": 66, "d_med": 309, "dr_med": 46, "sig": "d_unadj"},
    {"kernel": "softmax", "model": "O120", "d_succ": 94, "dr_succ": 90, "d_med": 1315, "dr_med": 1317, "sig": None},
    {"kernel": "softmax", "model": "Q235", "d_succ": 2, "dr_succ": 52, "d_med": 20, "dr_med": 18, "sig": None},
    {"kernel": "bgemm", "model": "O120", "d_succ": 88, "dr_succ": 90, "d_med": 552, "dr_med": 4546, "sig": None},
    {"kernel": "bgemm", "model": "Q235", "d_succ": 0, "dr_succ": 94, "d_med": None, "dr_med": 354, "sig": None},
    {"kernel": "dfspmm", "model": "O120", "d_succ": 86, "dr_succ": 82, "d_med": 801, "dr_med": 797, "sig": None},
    {"kernel": "dfspmm", "model": "Q235", "d_succ": 4, "dr_succ": 88, "d_med": 463, "dr_med": 2346, "sig": None},
    {"kernel": "fft", "model": "O120", "d_succ": 42, "dr_succ": 56, "d_med": 288, "dr_med": 424, "sig": None},
    {"kernel": "fft", "model": "Q235", "d_succ": 2, "dr_succ": 20, "d_med": 0.909, "dr_med": 0.958, "sig": None},
    {"kernel": "btdma", "model": "O120", "d_succ": 92, "dr_succ": 44, "d_med": 923, "dr_med": 380, "sig": "d_bh"},
    {"kernel": "btdma", "model": "Q235", "d_succ": 8, "dr_succ": 10, "d_med": 927, "dr_med": 404, "sig": None},
    {"kernel": "ddgemm", "model": "O120", "d_succ": 66, "dr_succ": 82, "d_med": 714, "dr_med": 564, "sig": "d_bh"},
    {"kernel": "ddgemm", "model": "Q235", "d_succ": 42, "dr_succ": 80, "d_med": 89, "dr_med": 510, "sig": "dr_bh"},
    {"kernel": "stencil", "model": "O120", "d_succ": 82, "dr_succ": 58, "d_med": 2670, "dr_med": 2570, "sig": None},
    {"kernel": "stencil", "model": "Q235", "d_succ": 36, "dr_succ": 62, "d_med": 2708, "dr_med": 2799, "sig": None},
    {"kernel": "spmm", "model": "O120", "d_succ": 66, "dr_succ": 54, "d_med": 51, "dr_med": 685, "sig": "dr_bh"},
    {"kernel": "spmm", "model": "Q235", "d_succ": 0, "dr_succ": 0, "d_med": None, "dr_med": None, "sig": None},
    {"kernel": "gemm", "model": "O120", "d_succ": 84, "dr_succ": 66, "d_med": 4416, "dr_med": 4731, "sig": "dr_bh"},
    {"kernel": "gemm", "model": "Q235", "d_succ": 0, "dr_succ": 0, "d_med": None, "dr_med": None, "sig": None},
    {"kernel": "spmv", "model": "O120", "d_succ": 68, "dr_succ": 60, "d_med": 1719, "dr_med": 1714, "sig": None},
    {"kernel": "spmv", "model": "Q235", "d_succ": 96, "dr_succ": 72, "d_med": 1687, "dr_med": 1686, "sig": None},
)

TABLE2_ITERATIVE: tuple[dict[str, object], ...] = (
    {"kernel": "conv2d", "model": "O120", "d_succ": 100, "dr_succ": 100, "d_med": 2512, "dr_med": 2908, "sig": "dr_bh"},
    {"kernel": "conv2d", "model": "Q235", "d_succ": 58, "dr_succ": 90, "d_med": 545, "dr_med": 4389, "sig": "dr_bh"},
    {"kernel": "bfft", "model": "O120", "d_succ": 100, "dr_succ": 100, "d_med": 370, "dr_med": 467, "sig": "dr_bh"},
    {"kernel": "bfft", "model": "Q235", "d_succ": 84, "dr_succ": 78, "d_med": 319, "dr_med": 273, "sig": None},
    {"kernel": "softmax", "model": "O120", "d_succ": 100, "dr_succ": 94, "d_med": 1322, "dr_med": 1385, "sig": "dr_bh"},
    {"kernel": "softmax", "model": "Q235", "d_succ": 86, "dr_succ": 44, "d_med": 35, "dr_med": 34, "sig": None},
    {"kernel": "bgemm", "model": "O120", "d_succ": 100, "dr_succ": 100, "d_med": 3666, "dr_med": 4485, "sig": None},
    {"kernel": "bgemm", "model": "Q235", "d_succ": 14, "dr_succ": 96, "d_med": 1.79, "dr_med": 4474, "sig": "dr_bh"},
    {"kernel": "ddgemm", "model": "Q235", "d_succ": 100, "dr_succ": 98, "d_med": 119, "dr_med": 848, "sig": "dr_bh"},
    {"kernel": "stencil", "model": "O120", "d_succ": 98, "dr_succ": 98, "d_med": 2849, "dr_med": 2788, "sig": "d_bh"},
    {"kernel": "stencil", "model": "Q235", "d_succ": 90, "dr_succ": 94, "d_med": 2919, "dr_med": 2840, "sig": "dr_bh"},
)

# BH-FDR significant Single-shot wins (after correction): 5 DR, 3 D
SINGLE_SHOT_DR_WINS_BH = (
    ("Q235", "conv2d"),
    ("Q235", "ddgemm"),
    ("O120", "conv2d"),
    ("O120", "spmm"),
    ("O120", "gemm"),
)
SINGLE_SHOT_D_WINS_BH = (
    ("O120", "ddgemm"),
    ("O120", "btdma"),
    ("Q235", "bfft"),
)

# Direct-3 control: DR significantly above D3 (unadjusted) in 4 cases
DIRECT3_DR_AHEAD = (
    ("O120", "conv2d", 1.3),
    ("O120", "fft", 1.3),
    ("Q235", "conv2d", 18.0),
    ("Q235", "ddgemm", 5.4),
)
