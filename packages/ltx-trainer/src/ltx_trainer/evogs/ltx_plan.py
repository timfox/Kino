"""Optional LTX progressive streaming sidecar plan."""

from __future__ import annotations

from typing import Any


def ltx_integration_plan() -> dict[str, Any]:
    return {
        "summary": "EvoGS evolution-tree LOD sidecars for progressive LTX delivery / Sphere360 streaming",
        "fold_keys": ["evogs_lod", "evogs_psi_energy", "evogs_ghost_ratio"],
        "sidecar_fields": {
            "lod_level": "0–3 quality tier for clip shard",
            "psi_energy_norm": "||ψ|| for transmit prioritization",
            "ghost_ratio_proxy": "fraction opacity < 0.005 in leaf set",
            "storage_mb_proxy": "cumulative refinement payload",
        },
        "inference_hooks": [
            "GOPEX_EVOGS_STREAM=1 — prefer lower-LOD prefix until bandwidth allows refinement",
            "Mixed foveal/peripheral LOD traversal (Sec. 3.3)",
        ],
        "env": {
            "GOPEX_EVOGS_ENABLE": "1",
            "GOPEX_EVOGS_STREAM": "1",
            "GOPEX_EVOGS_LEVELS": "4",
        },
    }


def gopex_env_snippet() -> str:
    plan = ltx_integration_plan()
    return "\n".join(f'export {k}="{v}"' for k, v in plan["env"].items())
