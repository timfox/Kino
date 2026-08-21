"""PCCL: Process Group-Aware Collective Algorithm Synthesizer (arXiv:2606.07019)."""

from __future__ import annotations

PAPER_ARXIV = "2606.07019"
PAPER_TITLE = "PCCL: Process Group-Aware Scalable and Generic Collective Algorithm Synthesizer"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"

COLLECTIVE_PATTERNS = (
    "broadcast",
    "scatter",
    "gather",
    "reduce",
    "reduce_scatter",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
)

SYNTHESIZER_COMPARISON: tuple[tuple[str, bool, bool, bool, bool], ...] = (
    ("SCCL", True, False, False, False),
    ("TACCL", True, True, False, False),
    ("Blink", False, True, False, False),  # △ scalability
    ("MultiTree", False, False, False, False),
    ("ForestColl", False, True, False, False),
    ("TACOS", True, True, False, False),
    ("TE-CCL", False, True, True, False),
    ("PCCL", True, True, True, True),
)

# Paper anchor metrics
SYNTHESIS_512_NPU_MINUTES = 11.68
SYNTHESIS_1000_NPU_HOURS = 2.01
TE_CCL_SPEEDUP_36NPU = 4404
PROCESS_GROUP_SPEEDUP_AVG = 2.68
ALL_TO_ALL_COMPLEXITY = "O(n^3)"
