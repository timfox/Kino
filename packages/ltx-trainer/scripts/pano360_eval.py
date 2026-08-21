#!/usr/bin/env python3
"""Pano360 equirectangular photogrammetry geometry CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.pano360.geometry import is_equirectangular_size  # noqa: E402
from ltx_trainer.pano360.synthetic import synthesize_equirect_rgb  # noqa: E402
from ltx_trainer.pano360.views import default_ring_views  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "name": "pano360",
                "task": "Equirectangular 360° video → perspective views → SfM / photogrammetry",
                "modules": ["geometry", "views", "ingest", "pipeline"],
                "default_views": 8,
            },
            indent=2,
        )
    )
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.pano360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    rgb = synthesize_equirect_rgb(args.height, args.width, seed=args.seed)
    print(
        json.dumps(
            {
                "shape": list(rgb.shape),
                "is_equirectangular": is_equirectangular_size(args.width, args.height),
                "mean": round(float(rgb.mean()), 2),
            },
            indent=2,
        )
    )
    return 0


def _cmd_views(args: argparse.Namespace) -> int:
    views = default_ring_views(args.count, extra_pitches=tuple(args.pitch))
    print(
        json.dumps(
            {
                "count": len(views),
                "views": [
                    {"yaw_deg": v.yaw_deg, "pitch_deg": v.pitch_deg, "fov_deg": v.fov_deg, "label": v.label}
                    for v in views
                ],
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Pano360 ERP photogrammetry stub")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=_cmd_knowledge)

    sp = sub.add_parser("smoke")
    sp.set_defaults(func=_cmd_smoke)

    sp = sub.add_parser("synthetic")
    sp.add_argument("--height", type=int, default=128)
    sp.add_argument("--width", type=int, default=256)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_synthetic)

    sp = sub.add_parser("views")
    sp.add_argument("--count", type=int, default=8)
    sp.add_argument("--pitch", type=float, nargs="*", default=[-25.0, 25.0])
    sp.set_defaults(func=_cmd_views)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
