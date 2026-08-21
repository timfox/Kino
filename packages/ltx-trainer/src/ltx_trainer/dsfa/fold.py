"""AV-fold sidecar for DSFA CodecFake metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("dsfa", {}))
    sidecar.setdefault("framework", "DSFA")
    sidecar.setdefault("task", "codecfake_detection")
    sidecar.setdefault("proxy_data", "CoRS")
    sidecar.setdefault("eval_sets", ["CoSG Eval", "CoSG ExtEval"])
    data = dict(data)
    data["dsfa"] = sidecar
    return data
