"""Evaluation setup (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "eval_protocol": "20 outdoor prompt sets, Matterport3D-style",
        "backbones": ["SANA", "LTX-Video", "FLUX", "HunyuanVideo"],
        "baselines": ["360 LoRA", "Text2Light", "PanFusion", "DynamicScaler", "360DVD"],
    }
