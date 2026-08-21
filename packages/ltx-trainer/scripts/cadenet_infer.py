#!/usr/bin/env python3
"""CADENet inference / benchmark demo (Khairy & Elias arXiv:2605.19837)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.cadenet.metrics import dawn_table_ii, detection_f1  # noqa: E402
from ltx_trainer.cadenet.pipeline import CADENet  # noqa: E402
from ltx_trainer.cadenet.schema import WeatherCondition  # noqa: E402
from ltx_trainer.cadenet.synthetic import synthesize_scene  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="CADENet adverse-weather perception")
    p.add_argument("--weather", choices=[w.value for w in WeatherCondition], default="rain")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark", action="store_true", help="Print DAWN Table II stats")
    args = p.parse_args()

    if args.benchmark:
        print(json.dumps({"dawn_table_ii": dawn_table_ii()}, indent=2))
        return

    weather = WeatherCondition(args.weather)
    degraded, _, gt = synthesize_scene(weather=weather, size=args.size, seed=args.seed)
    net = CADENet()
    result = net.process_frame(degraded)
    c1 = detection_f1(result.detections_s, gt)
    c2 = detection_f1(result.detections_fused, gt)
    out = {
        "weather": result.weather.value,
        "severity": result.severity,
        "c1_f1": round(c1, 4),
        "c2_f1": round(c2, 4),
        "delta_f1": round(c2 - c1, 4),
        "num_tracks": len(result.tracks),
        "num_det_s": len(result.detections_s),
        "num_det_fused": len(result.detections_fused),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
