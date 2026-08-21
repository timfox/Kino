"""Paper metadata for LF-Diff."""

from __future__ import annotations

from ltx_trainer.lf_diff.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict:
    b = benchmarks_bundle()
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "task": "Multi-exposure HDR via LPR + linearized diffusion (tonemap T + LPR loss)",
        "tables": list(b.keys()),
    }
