#!/usr/bin/env python3
"""PanoWorld-X CLI (Yin et al., arXiv:2509.24997)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.panoworld_x.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.panoworld_x.config import PAPER_URL, PanoWorldXConfig  # noqa: E402
from ltx_trainer.panoworld_x.datasets import datasets_card  # noqa: E402
from ltx_trainer.panoworld_x.paper import framework_card  # noqa: E402
from ltx_trainer.panoworld_x.pipeline import evaluation_demo_run, train_step  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.panoworld_x.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    from ltx_trainer.panoworld_x.benchmarks import TABLE2_ABLATION
    from ltx_trainer.panoworld_x.pipeline import ablation_branches

    cfg = PanoWorldXConfig(height=args.height, width=args.width, patch_size=16)
    print(
        json.dumps(
            {"table2_paper": TABLE2_ABLATION, "stub_forward": ablation_branches(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_sphere(_: argparse.Namespace) -> int:
    from ltx_trainer.panoworld_x.pipeline import sphere_edge_connectivity

    print(json.dumps(sphere_edge_connectivity(PanoWorldXConfig()), indent=2))
    return 0


def _cmd_route_pipeline(_: argparse.Namespace) -> int:
    from ltx_trainer.panoworld_x.route_sampling import pipeline_report

    print(json.dumps(pipeline_report(), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = PanoWorldXConfig(height=args.height, width=args.width, patch_size=16)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = PanoWorldXConfig(height=args.height, width=args.width, patch_size=16)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PanoWorld-X sphere-aware panoramic video")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("sphere").set_defaults(func=_cmd_sphere)
    sub.add_parser("route-pipeline").set_defaults(func=_cmd_route_pipeline)
    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=96)
    ab.add_argument("--width", type=int, default=192)
    ab.set_defaults(func=_cmd_ablation)
    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=64)
    tr.add_argument("--width", type=int, default=128)
    tr.set_defaults(func=_cmd_train_stub)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=96)
    demo.add_argument("--width", type=int, default=192)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
