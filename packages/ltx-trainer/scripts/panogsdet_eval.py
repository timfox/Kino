#!/usr/bin/env python3
"""PanoGSDet panoramic 3D detection CLI (Ning et al. arXiv:2605.14601)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.panogsdet.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.panogsdet.config import PAPER_URL, PanoGSDetConfig  # noqa: E402
from ltx_trainer.panogsdet.paper import framework_card  # noqa: E402
from ltx_trainer.panogsdet.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.panogsdet.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = PanoGSDetConfig(
        height=args.height,
        width=args.width,
        feature_dim=args.feature_dim,
        opt_blocks=args.opt_blocks,
    )
    out = evaluation_demo_run(
        cfg,
        device=args.device,
        face_size=args.face_size,
        max_gaussians=args.max_gaussians,
    )
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PanoGSDet ERP 3D detection (arXiv:2605.14601)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.add_argument("--feature-dim", type=int, default=16)
    demo.add_argument("--opt-blocks", type=int, default=1)
    demo.add_argument("--face-size", type=int, default=24)
    demo.add_argument("--max-gaussians", type=int, default=256)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
