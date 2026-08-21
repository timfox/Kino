"""AV-fold sidecar for BiEAR binaural metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("biear", {}))
    sidecar.setdefault("framework", "BiEAR")
    sidecar.setdefault("adaptive_q_filterbank", True)
    sidecar.setdefault("n_sectors", 8)
    sidecar.setdefault("tasks", ["detect", "azimuth", "distance"])
    data = dict(data)
    data["biear"] = sidecar
    return data
