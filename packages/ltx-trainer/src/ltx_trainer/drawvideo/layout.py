"""Scope notes for DrawVideo reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No FLUX, Kontext, or Wan2.2 weights: pipeline stages and paper metrics only.",
    "SketchLongVideo raw videos are not redistributed; only construction metadata is modeled.",
    "PySceneDetect, FFmpeg, and ComfyUI workflows are external to this package.",
    "Evaluation metrics (LPIPS, CLIP, Edge-F1) are quoted from Table 1, not computed here.",
)
