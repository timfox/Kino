"""MixFake TKEO smoke."""

from __future__ import annotations

from typing import Any


def toy_feature_sequence() -> list[float]:
    return [0.1, 0.3, -0.2, 0.5, 0.0]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.mixfake.signals import feature_flux, teager_kaiser_sequence

    seq = toy_feature_sequence()
    tkeo = teager_kaiser_sequence(seq)
    return {"feature_len": len(seq), "tkeo_len": len(tkeo), "flux": round(feature_flux(seq), 4)}
