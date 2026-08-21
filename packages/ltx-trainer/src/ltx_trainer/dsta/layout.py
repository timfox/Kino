"""Scope notes for DSTA reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No VideoMAE-B / ActionFormer / AdaTAD training: adapter math and paper tables only.",
    "Fine-Badminton full 31-match release is partial on Zenodo (10-video subset).",
    "mAP values are quoted from Table II–V, not computed on local video features.",
    "PyTorch/MMAction2/MMCV training stack is external to this package.",
)
