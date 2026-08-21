"""Paper cards and knowledge bundle (Matsuoka, arXiv:2606.06510)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.benchmarks import (
    summary_anchors,
    table_3_speedups,
    table_4_h100_baseline,
    table_5_substrate_comparison,
)
from ltx_trainer.fp8_ozaki.constants import (
    PAPER_ARXIV,
    PAPER_AUTHOR,
    PAPER_TITLE,
    PAPER_URL,
)
from ltx_trainer.fp8_ozaki.references import reference_anchors


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "author": PAPER_AUTHOR,
        "findings": [
            "TME + Ozaki II: fp8 emulation restores fp64 across HPC kernel spectrum",
            "B300 dense GEMM Ozaki speedup ~380× vs native fp64 on same GPU",
            "register-fused β→1 required for memory-bound kernels (stencil, SpMV)",
            "Rubin column adds published emulated DGEMM; Genesis path for DOE",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "table_3": table_3_speedups(),
        "table_4": table_4_h100_baseline(),
        "table_5": table_5_substrate_comparison(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
