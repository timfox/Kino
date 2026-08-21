#!/usr/bin/env python3
"""(k, δ)-truss evaluation CLI (Hu et al., arXiv:2606.11582)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.kd_truss.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.kd_truss.config import PAPER_URL, KdTrussConfig  # noqa: E402
from ltx_trainer.kd_truss.integration import integration_bundle  # noqa: E402
from ltx_trainer.kd_truss.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.kd_truss.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from ltx_trainer.kd_truss.running_example import FIG1_TIMESTAMPS
    from ltx_trainer.kd_truss.truss import online_kd_truss_query

    edges = FIG1_TIMESTAMPS.keys()
    result = sorted(online_kd_truss_query(edges, k=args.k, delta=args.delta))
    print(json.dumps({"k": args.k, "delta": args.delta, "edges": [list(e) for e in result]}, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo(seed=args.seed)
    out["config"] = KdTrussConfig().__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="(k, δ)-truss evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    q = sub.add_parser("query")
    q.add_argument("--k", type=int, default=4)
    q.add_argument("--delta", type=int, default=1)

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "query": _cmd_query,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
