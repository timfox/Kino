"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "uwd_lex": {
            "task": "unsupervised_word_discovery_lexicon_eval",
            "paper": "arXiv:2606.06183",
            "metrics": ["WNES", "iWNES", "F1-WNES", "d-PAcc"],
        },
    }
