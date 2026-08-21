"""LiveBrowseComp evaluation smoke (arXiv:2605.28721)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livebrowsecomp.pipeline import evaluation_smoke as _evaluation_smoke


def evaluation_smoke() -> dict[str, Any]:
    return _evaluation_smoke()
