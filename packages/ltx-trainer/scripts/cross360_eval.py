#!/usr/bin/env python3
"""Cross360 CLI (Huang et al., arXiv:2601.17271)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.cross360.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.cross360.config import CODE_URL, Cross360Config, PAPER_URL  # noqa: E402
from ltx_trainer.cross360.datasets import all_datasets_card, dataset_card  # noqa: E402
from ltx_trainer.cross360.paper import framework_card  # noqa: E402
from ltx_trainer.cross360.pipeline import evaluation_demo_run  # noqa: E402
from ltx_trainer.cross360.tangent import tp_sampling_layout  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.name:
        print(json.dumps(dataset_card(args.name), indent=2))
    else:
        print(json.dumps(all_datasets_card(), indent=2))
    return 0


def _cmd_tp_layout(args: argparse.Namespace) -> int:
    print(json.dumps(tp_sampling_layout(incomplete_fov=args.incomplete), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.cross360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = Cross360Config(
        height=args.height,
        width=args.width,
        incomplete_fov=args.incomplete,
    )
    out = evaluation_demo_run(cfg, device=args.device)
    print(json.dumps({"paper": PAPER_URL, "code": CODE_URL, **out}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Cross360 360° depth via cross projections")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    ds = sub.add_parser("dataset")
    ds.add_argument("--name", default="", help="Matterport3D, Structured3D, …")
    ds.set_defaults(func=_cmd_dataset)
    tpl = sub.add_parser("tp-layout")
    tpl.add_argument("--incomplete", action="store_true")
    tpl.set_defaults(func=_cmd_tp_layout)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    demo = sub.add_parser("demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.add_argument("--incomplete", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
