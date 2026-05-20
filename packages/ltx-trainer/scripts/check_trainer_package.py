#!/usr/bin/env python3
"""Verify the ltx-trainer package has scripts and modules required for training."""

from __future__ import annotations

import sys
from pathlib import Path

_REQUIRED_SCRIPTS = (
    "train.py",
    "process_captions.py",
    "process_dataset.py",
    "process_videos.py",
    "inference.py",
    "fold_flat_dim_bridge.py",
)

_REQUIRED_SRC = (
    "src/ltx_trainer/trainer.py",
    "src/ltx_trainer/config.py",
    "src/ltx_trainer/validation_sampler.py",
    "src/ltx_trainer/datasets.py",
    "src/ltx_trainer/model_loader.py",
)

_REQUIRED_CONFIGS = (
    "configs/ltx2_av_lora_gemma4_bridge_connectors.yaml",
    "configs/ltx2_text_stack_gemma4_bridge.yaml",
)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    missing: list[str] = []
    for name in _REQUIRED_SCRIPTS:
        if not (root / "scripts" / name).is_file():
            missing.append(f"scripts/{name}")
    for rel in _REQUIRED_SRC:
        if not (root / rel).is_file():
            missing.append(rel)
    for rel in _REQUIRED_CONFIGS:
        if not (root / rel).is_file():
            missing.append(rel)
    if missing:
        print("ltx-trainer package incomplete:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print(f"OK: ltx-trainer package at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
