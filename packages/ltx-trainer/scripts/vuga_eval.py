#!/usr/bin/env python3
"""VUGA viewport-unaware BOIQA / BIQA CLI (Yan et al. arXiv:2604.23953)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.vuga.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.vuga.config import PAPER_URL, VUGAConfig  # noqa: E402
from ltx_trainer.vuga.paper import framework_card  # noqa: E402
from ltx_trainer.vuga.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.vuga.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = VUGAConfig(
        input_size=args.input_size,
        stage_dims=tuple(int(x) for x in args.stage_dims.split(",")),
        cmp_dim=args.cmp_dim,
    )
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="VUGA BOIQA/BIQA (arXiv:2604.23953)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--input-size", type=int, default=64)
    demo.add_argument("--stage-dims", default="32,64,128,256")
    demo.add_argument("--cmp-dim", type=int, default=128)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
