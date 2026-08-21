"""Sharp Picture Fourier PAC-Bayes smoke (arXiv:2605.20988)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sharp_picture.config import SharpPictureConfig
from ltx_trainer.sharp_picture.fourier import fourier_expansion, parity_characteristic


def evaluation_smoke(cfg: SharpPictureConfig | None = None) -> dict[str, Any]:
    c = cfg or SharpPictureConfig()
    x = [0, 1, 0, 1]
    chi = parity_characteristic({0, 2}, x)
    val = fourier_expansion({frozenset({0}): 0.5, frozenset(): 0.1}, x)
    return {
        "paper": "arXiv:2605.20988",
        "parity_char": round(chi, 3),
        "fourier_val": round(val, 3),
    }
