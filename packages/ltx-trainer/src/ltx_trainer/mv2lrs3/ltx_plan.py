"""LTX integration plan for AVSR generalisability metadata."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mv2lrs3.config import Mv2Lrs3Config


def ltx_integration_plan(cfg: Mv2Lrs3Config | None = None) -> dict[str, Any]:
    cfg = cfg or Mv2Lrs3Config()
    return {
        "paper": cfg.paper_arxiv,
        "use_case": "Attach MV2LRS3 covariate vectors to AV training shards for generalisation audits",
        "preprocess": [
            "Extract 8-D matching features per utterance",
            "Tag easy/hard subsets from multi-model WER consensus",
        ],
        "training": "No DiT gradient — AVSR benchmark metadata only",
    }
