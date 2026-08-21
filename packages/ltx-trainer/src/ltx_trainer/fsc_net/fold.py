"""AV-fold sidecar for FSC-Net BWE metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("fsc_net", {}))
    sidecar.setdefault("framework", "FSC-Net")
    sidecar.setdefault("bwe_scenarios", ["4khz_to_48khz", "16khz_to_48khz"])
    sidecar.setdefault("progressive_windows", [257, 65, 17, 5, 1])
    data = dict(data)
    data["fsc_net"] = sidecar
    return data
