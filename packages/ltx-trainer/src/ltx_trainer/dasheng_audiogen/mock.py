"""Dasheng AudioGen evaluation smoke (arXiv:2605.27838)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.pipeline import pipeline_demo


def evaluation_smoke(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    c = cfg or DashengAudioGenConfig()
    demo = pipeline_demo(c, seed=42)
    ev = demo["eval"]
    return {
        "paper": c.paper_arxiv,
        "latent_dim": c.latent_dim,
        "audio_scene_capable": True,
        "sma_fad_ours_beats_expert": ev["sma_fad_ours_beats_expert"],
        "structured_captions_help": ev["structured_beats_unstructured_wer"],
        "musiccaps_fad": c.musiccaps_fad,
    }
