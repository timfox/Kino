#!/usr/bin/env python3
"""Terastal heterogeneous multi-DNN scheduling CLI (Wu et al., arXiv:2606.06818)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.terastal.benchmarks import (  # noqa: E402
    miss_rate_comparison_anchors,
    summary_anchors,
    table_i_hardware,
    table_ii_workloads,
)
from ltx_trainer.terastal.budget import model_virtual_budgets  # noqa: E402
from ltx_trainer.terastal.accelerators import vgg11_layer_latencies_us  # noqa: E402
from ltx_trainer.terastal.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.terastal.pipeline import run_demo  # noqa: E402
from ltx_trainer.terastal.variants import variant_transform_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "table_i": table_i_hardware(),
                "table_ii": table_ii_workloads(),
                "miss_rate": miss_rate_comparison_anchors(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_variant(_: argparse.Namespace) -> int:
    print(json.dumps(variant_transform_card(args.gamma), indent=2))
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    profile = vgg11_layer_latencies_us()
    layer_lat = [[row["ws_us"], row["os_us"]] for row in profile]
    print(json.dumps(model_virtual_budgets(layer_lat, args.deadline), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(run_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.terastal.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Terastal layer-variant scheduling stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    vp = sub.add_parser("variant")
    vp.add_argument("--gamma", type=int, default=2)
    vp.set_defaults(func=_cmd_variant)

    bp = sub.add_parser("budget")
    bp.add_argument("--deadline", type=float, default=1.0 / 15.0)
    bp.set_defaults(func=_cmd_budget)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
