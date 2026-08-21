"""Paper anchors — GPU multi-precision integer division (Marchioro et al., arXiv:2606.06386)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06386"
PAPER_TITLE = "On GPU Implementation for Multi-Precision Integer Division"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
UPSTREAM_REPO = "https://github.com/aske0778/midint-arithmetic-division"

# Midsize regime: one division instance per CUDA block (A100 evaluation).
BIT_EXP_MIN = 13
BIT_EXP_MAX = 18
DEFAULT_WORD_BITS = 64
DEFAULT_SEQUENTIALIZATION_Q = 4
SHARED_MEMORY_LIMIT_KB = 100
REGISTER_SPILL_BYTES_2P18 = 392

# Cost model (classical multiplication): full division uses 5–7 full mults (§2.3).
FULL_MULT_COST_MIN = 5
FULL_MULT_COST_MAX = 7

# Theorem 2 quotient correction δ ∈ {-1, 0, 1}.
QUOTIENT_DELTA_RANGE = (-1, 0, 1)

# Table 1 anchors (A100, uint64, Q=4).
TABLE1_ROWS: tuple[dict[str, float | int | str], ...] = (
    {"bits_exp": 18, "insts_exp": 14, "our_mul_ms": 271.205, "cgbn_mul_ratio": 0.037, "div_mul_ratio": 5.17, "cgbn_div_ratio": None, "gmp_speedup": 11.3},
    {"bits_exp": 17, "insts_exp": 15, "our_mul_ms": 130.994, "cgbn_mul_ratio": 0.268, "div_mul_ratio": 5.38, "cgbn_div_ratio": None, "gmp_speedup": 14.0},
    {"bits_exp": 16, "insts_exp": 16, "our_mul_ms": 69.059, "cgbn_mul_ratio": 1.041, "div_mul_ratio": 5.73, "cgbn_div_ratio": None, "gmp_speedup": 18.9},
    {"bits_exp": 15, "insts_exp": 17, "our_mul_ms": 38.048, "cgbn_mul_ratio": 1.124, "div_mul_ratio": 7.83, "cgbn_div_ratio": 3.63, "gmp_speedup": 12.6},
    {"bits_exp": 14, "insts_exp": 18, "our_mul_ms": 24.619, "cgbn_mul_ratio": 1.512, "div_mul_ratio": 8.95, "cgbn_div_ratio": 4.98, "gmp_speedup": 7.4},
    {"bits_exp": 13, "insts_exp": 19, "our_mul_ms": 19.198, "cgbn_mul_ratio": 2.112, "div_mul_ratio": 9.96, "cgbn_div_ratio": 7.36, "gmp_speedup": 11.1},
)

CGBN_DIV_MAX_BITS_EXP = 15  # cgbn_div_rem not supported above 2^15 in paper experiments

PAPER_EXAMPLES: tuple[dict[str, int], ...] = (
    {"u": 314_159_265_358_979, "v": 27_183, "q": 11_557_196_238, "B": 10},
    {"u": 726_319_138_718_412, "v": 27_183, "q": 26_719_609_267, "B": 10},
)
