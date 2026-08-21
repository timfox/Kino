"""Smoke evaluation (auto-generated; extend with domain-specific toy calls)."""

from __future__ import annotations

from typing import Any


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.airbnb_nls.pipeline import generation_step_demo

    result = generation_step_demo()
    return _slim_result(result)


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
