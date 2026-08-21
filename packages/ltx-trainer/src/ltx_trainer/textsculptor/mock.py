"""TextSculptor metrics smoke (arXiv:2605.21090)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.textsculptor.config import TextSculptorConfig
from ltx_trainer.textsculptor.metrics import text_accuracy


def evaluation_smoke(cfg: TextSculptorConfig | None = None) -> dict[str, Any]:
    c = cfg or TextSculptorConfig()
    acc = text_accuracy("OPEN", "OPEN", n_edit_words=1)
    return {
        "paper": "arXiv:2605.21090",
        "text_accuracy": round(acc, 4),
    }
