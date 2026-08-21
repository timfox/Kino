"""Smoke evaluation (auto-generated; extend with domain-specific toy calls)."""

from __future__ import annotations

from typing import Any


def evaluation_smoke() -> dict[str, Any]:
    from pathlib import Path

    import numpy as np

    from ltx_trainer.trisplat.cameras import TriSplatCamera

    cam = TriSplatCamera(
        image_path=Path("view0.png"),
        width=640,
        height=480,
        K=np.eye(3),
        R_cw=np.eye(3),
        t_cw=np.zeros(3),
    )
    return _slim_result({"width": cam.width, "height": cam.height, "view_id": cam.view_id})


def _slim_result(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        out: dict[str, Any] = {}
        for k, v in result.items():
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, (int, str, bool)) or v is None:
                out[k] = v
        return out or {"ok": True}
    if isinstance(result, (int, float)):
        return {"value": round(float(result), 4)}
    if isinstance(result, (list, tuple)) and len(result) <= 8:
        return {"len": len(result)}
    return {"ok": True, "repr": str(result)[:120]}
