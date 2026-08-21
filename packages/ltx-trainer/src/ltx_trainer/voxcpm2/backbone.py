"""VoxCPM family backbone configuration (Table 1)."""

from __future__ import annotations

from typing import Any


def table1_family_config() -> list[dict[str, Any]]:
    """Table 1 — VoxCPM / VoxCPM1.5 / VoxCPM2."""
    return [
        {
            "model": "VoxCPM",
            "params_b": 0.6,
            "locenc": "4L, H=1024",
            "tslm": "MiniCPM-4-0.5B",
            "fsq_dim": 256,
            "ralm": "6L, H=1024",
            "locdit": "4L, H=1024",
            "patch_p": 2,
            "lm_hz": 12.5,
            "max_seq": 4096,
            "in_hz": 16000,
            "out_hz": 16000,
        },
        {
            "model": "VoxCPM1.5",
            "params_b": 0.8,
            "locenc": "8L, H=1024",
            "tslm": "MiniCPM-4-0.5B",
            "fsq_dim": 256,
            "ralm": "8L, H=1024",
            "locdit": "8L, H=1024",
            "patch_p": 4,
            "lm_hz": 6.25,
            "max_seq": 4096,
            "in_hz": 44100,
            "out_hz": 44100,
        },
        {
            "model": "VoxCPM2",
            "params_b": 2.0,
            "locenc": "12L, H=1024",
            "tslm": "MiniCPM-4-1B",
            "fsq_dim": 512,
            "ralm": "8L, H=2048",
            "locdit": "12L, H=1024",
            "patch_p": 4,
            "lm_hz": 6.25,
            "max_seq": 8192,
            "in_hz": 16000,
            "out_hz": 48000,
        },
    ]


def backbone_demo() -> dict[str, Any]:
    rows = table1_family_config()
    v2 = next(r for r in rows if r["model"] == "VoxCPM2")
    return {
        "voxcpm2_params_b": v2["params_b"],
        "fsq_dim": v2["fsq_dim"],
        "lm_token_rate_hz": v2["lm_hz"],
        "output_hz": v2["out_hz"],
    }
