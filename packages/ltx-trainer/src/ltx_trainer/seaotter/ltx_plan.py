"""Gopex hooks for cloud-robotics video ingest and perception sidecars."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seaotter.config import SeaotterConfig


def ltx_integration_plan(cfg: SeaotterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeaotterConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_case": (
            "Extreme on-device compression for robot/wearable cameras with "
            "one-time cloud transcode to JPEG for ML training and VLM/VLA consumers."
        ),
        "gopex_workflow": [
            "Capture RGB @ full resolution on edge (FRAPPE G_A encode)",
            "Uplink int8 latent over BLE/5G/Wi-Fi budget",
            "Cloud: G_S decode + learned JPEG sandwich transcode (once per frame)",
            "Store standard JPEG with custom Q tables in metadata",
            "Downstream: dataloaders use JPEG decode; optional F⁻¹ for sRGB backbones",
            "Attach body-type / scene tags from parallel perception pipelines",
        ],
        "env_knobs": {
            "GOPEX_SEAOTTER_FRAPPE_N": str(cfg.matched_frappe_n),
            "GOPEX_SEAOTTER_VARIANT": "ZS | FT",
            "GOPEX_SEAOTTER_ARTIFACT": "Path to bundled (F, F⁻¹, Q, α) from UT-SysML/seaotter",
            "GOPEX_SEAOTTER_SKIP_INVERSE": "1 to consume learned color space directly",
        },
        "bandwidth_targets": {
            "wifi_1080p30_25mbps_cr": 60,
            "5g_720p30_5mbps_cr": 133,
            "ble_480p30_1mbps_cr": 288,
        },
        "related": [cfg.repo_url, cfg.frappe_repo],
    }
