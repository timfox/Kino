"""AV-fold sidecar for MV2LRS3 metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("mv2lrs3", {}))
    sidecar.setdefault("benchmark", "MV2LRS3")
    sidecar.setdefault("reference_distribution", "LRS3_test")
    sidecar.setdefault("matching_factors", 7)
    data = dict(data)
    data["mv2lrs3"] = sidecar
    return data
