#!/usr/bin/env python3
"""One Flight Over the Gap survey CLI (Lin et al., arXiv:2509.04444)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.pano_flight.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.pano_flight.config import PAPER_URL, PROJECT_URL, PanoFlightConfig  # noqa: E402
from ltx_trainer.pano_flight.future import future_card  # noqa: E402
from ltx_trainer.pano_flight.gaps import gaps_card  # noqa: E402
from ltx_trainer.pano_flight.paper import framework_card  # noqa: E402
from ltx_trainer.pano_flight.pipeline import (  # noqa: E402
    cross_method_demo,
    cross_task_demo,
    evaluation_demo_run,
    stitching_pipeline_card,
)
from ltx_trainer.pano_flight.projections import projection_catalogue  # noqa: E402
from ltx_trainer.pano_flight.strategies import classify_method, strategies_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_gaps(_: argparse.Namespace) -> int:
    print(json.dumps(gaps_card(), indent=2))
    return 0


def _cmd_strategies(_: argparse.Namespace) -> int:
    print(json.dumps(strategies_card(), indent=2))
    return 0


def _cmd_projections(_: argparse.Namespace) -> int:
    print(json.dumps(projection_catalogue(), indent=2))
    return 0


def _cmd_taxonomy(_: argparse.Namespace) -> int:
    print(json.dumps(cross_task_demo(), indent=2))
    return 0


def _cmd_future(_: argparse.Namespace) -> int:
    print(json.dumps(future_card(), indent=2))
    return 0


def _cmd_stitching(_: argparse.Namespace) -> int:
    print(json.dumps(stitching_pipeline_card(), indent=2))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"method": args.name, "strategy": classify_method(args.name).value},
            indent=2,
        )
    )
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.pano_flight.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = PanoFlightConfig(height=args.height, width=args.width)
    print(
        json.dumps(
            {
                "paper": PAPER_URL,
                "project": PROJECT_URL,
                "demo": evaluation_demo_run(cfg),
                "cross_method": cross_method_demo(),
                "cross_task": cross_task_demo(),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Perspective-to-panoramic vision survey")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("gaps").set_defaults(func=_cmd_gaps)
    sub.add_parser("strategies").set_defaults(func=_cmd_strategies)
    sub.add_parser("projections").set_defaults(func=_cmd_projections)
    sub.add_parser("taxonomy").set_defaults(func=_cmd_taxonomy)
    sub.add_parser("future").set_defaults(func=_cmd_future)
    sub.add_parser("stitching").set_defaults(func=_cmd_stitching)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    clf = sub.add_parser("classify")
    clf.add_argument("name", help="Method name to map to mitigation strategy")
    clf.set_defaults(func=_cmd_classify)

    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=256)
    demo.add_argument("--width", type=int, default=512)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
