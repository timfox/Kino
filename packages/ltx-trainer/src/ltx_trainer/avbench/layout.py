"""Scope and limitations for AVBench reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Benchmark focuses on short clips (about 5–12 s), aligned with current T2AV model defaults; long-form extension is future work.",
    "Full automated evaluation requires released AVBench weights, SyncNet/LatentSync, Whisper, DF_Arena, NISQAv2, DOVER++, and Audiobox checkpoints — not bundled here.",
    "470 curated prompts and 300K SFT pairs are paper-scale; this package only encodes tables, formulas, and smoke helpers.",
    "Human 2AFC and Pearson correlations are reported constants in the paper; reproducing full human studies is external.",
)
