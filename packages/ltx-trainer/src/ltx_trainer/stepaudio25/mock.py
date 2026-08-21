"""StepAudio 2.5 MTP loss smoke (arXiv:2605.23463)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stepaudio25.config import Stepaudio25Config
from ltx_trainer.stepaudio25.mtp import mtp_branch_weights, mtp_combined_loss


def evaluation_smoke(cfg: Stepaudio25Config | None = None) -> dict[str, Any]:
    c = cfg or Stepaudio25Config()
    w = mtp_branch_weights(h=5, alpha=0.9)
    loss = mtp_combined_loss(0.5, [0.4, 0.3, 0.2, 0.1, 0.05])
    return {
        "paper": c.paper_arxiv,
        "branch_weights_sum": round(float(w.sum()), 4),
        "mtp_loss": round(loss, 4),
        "asr_rtf": c.asr_rtf,
    }
