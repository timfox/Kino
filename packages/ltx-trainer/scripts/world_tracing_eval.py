#!/usr/bin/env python3
"""World Tracing pixel-aligned multilayer geometry CLI (arXiv:2606.13652)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.world_tracing.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.world_tracing.config import PAPER_URL, WTConfig  # noqa: E402
from ltx_trainer.world_tracing.paper import framework_card  # noqa: E402
from ltx_trainer.world_tracing.pipeline import evaluation_demo_run  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.world_tracing.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = WTConfig(
        height=args.height,
        width=args.width,
        patch_size=args.patch_size,
        width_dim=args.width_dim,
        num_blocks=args.num_blocks,
        ode_steps=args.ode_steps,
    )
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="World Tracing WT-DiT (arXiv:2606.13652)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--height", type=int, default=28)
    demo.add_argument("--width", type=int, default=28)
    demo.add_argument("--patch-size", type=int, default=14)
    demo.add_argument("--width-dim", type=int, default=64)
    demo.add_argument("--num-blocks", type=int, default=1)
    demo.add_argument("--ode-steps", type=int, default=2)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
