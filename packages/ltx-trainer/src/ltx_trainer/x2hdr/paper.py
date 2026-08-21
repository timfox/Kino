"""Paper metadata for X2HDR."""

from __future__ import annotations

from ltx_trainer.x2hdr.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle


def framework_card() -> dict:
    b = benchmarks_bundle()
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "task": "SDR-to-HDR video via PU21-aligned frozen VAE + latent diffusion",
        "encoding": "pu21",
        "tables": list(b.keys()),
    }
