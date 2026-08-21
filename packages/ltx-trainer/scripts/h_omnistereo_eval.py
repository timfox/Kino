#!/usr/bin/env python3
"""H-OmniStereo omnidirectional stereo CLI (Jiang et al. arXiv:2605.14963)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.h_omnistereo.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.h_omnistereo.config import PAPER_URL, HOmniStereoConfig  # noqa: E402
from ltx_trainer.h_omnistereo.paper import framework_card  # noqa: E402
from ltx_trainer.h_omnistereo.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.h_omnistereo.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = HOmniStereoConfig(
        train_crop_w=args.width,
        train_crop_h=args.height,
        feature_dim=args.feature_dim,
        refine_iters=args.refine_iters,
    )
    out = evaluation_demo_run(cfg, device=args.device, refine_iters=args.refine_iters)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="H-OmniStereo ERP stereo (arXiv:2605.14963)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--width", type=int, default=64)
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--feature-dim", type=int, default=32)
    demo.add_argument("--refine-iters", type=int, default=2)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
