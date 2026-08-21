"""WS/OS heterogeneous accelerator latency profiles (Section III)."""

from __future__ import annotations

from typing import Any


def accelerator_card(dataflow: str) -> dict[str, Any]:
    if dataflow == "ws":
        return {
            "dataflow": "weight_stationary",
            "efficient_for": "layers with many channels",
            "reference": "NVDLA-style",
        }
    return {
        "dataflow": "output_stationary",
        "efficient_for": "layers with large output feature maps",
        "reference": "ShiDianNao-style",
    }


def vgg11_layer_latencies_us() -> list[dict[str, float]]:
    """VGG11 per-layer latency profile (Fig. 3 style): WS preferred, OS 2–8× slower late layers."""
    base_ws = [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
    rows = []
    for i, ws in enumerate(base_ws):
        ratio = 1.0 if i < 4 else min(8.0, 1.5 + 0.6 * (i - 3))
        rows.append({"layer": i + 1, "ws_us": float(ws), "os_us": float(ws * ratio)})
    return rows


def preferred_accelerator(layer_idx: int, *, late_ws_bias: bool = True) -> str:
    """Most VGG11 layers in paper are WS-preferred; variants target OS."""
    if late_ws_bias and layer_idx >= 4:
        return "ws"
    return "ws" if layer_idx % 2 == 0 else "os"
