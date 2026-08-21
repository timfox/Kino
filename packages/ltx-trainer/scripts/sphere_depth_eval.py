#!/usr/bin/env python3
"""Sphere-Depth pose-aware ERP depth benchmark CLI (arXiv:2604.23432)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.sphere_depth.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.sphere_depth.config import PAPER_URL, SphereDepthConfig  # noqa: E402
from ltx_trainer.sphere_depth.paper import framework_card  # noqa: E402
from ltx_trainer.sphere_depth.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.sphere_depth.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = SphereDepthConfig(
        erp_height=args.erp_height,
        erp_width=args.erp_width,
        cubemap_face_size=args.face_size,
    )
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Sphere-Depth ERP depth benchmark (arXiv:2604.23432)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--erp-height", type=int, default=64)
    demo.add_argument("--erp-width", type=int, default=128)
    demo.add_argument("--face-size", type=int, default=32)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
