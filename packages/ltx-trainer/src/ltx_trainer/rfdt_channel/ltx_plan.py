"""LTX / GOPEX hooks for RF digital twin indoor scenes."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig


def ltx_integration_plan(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Indoor teleplay sets: mesh from video + depth → 28 GHz link budget",
            "CONTROL ROOM / office scenes: semantic materials for CFR conditioning",
            "Compare All-Concrete vs multi-material sidecars for LTX audio-video sync",
        ],
        "hooks": {
            "capture": "GMSL RGB + LiDAR PCD via ROS 2 export",
            "reconstruct": "COLMAP poses → 3DGS → SuGaR mesh",
            "regularize": "LiDAR scale + wall solidify before simulation",
            "simulate": f"Sionna RT {cfg.carrier_ghz} GHz CIR/CFR/RadioMap",
        },
        "outputs": ["CIR", "CFR", "Radio_Map"],
    }
