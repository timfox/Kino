#!/usr/bin/env python3
"""URNG / UG evaluation CLI (Liang et al., arXiv:2606.11789)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.urng.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.urng.config import PAPER_URL, UrngConfig  # noqa: E402
from ltx_trainer.urng.integration import integration_bundle  # noqa: E402
from ltx_trainer.urng.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.urng.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from ltx_trainer.urng.intervals import QueryType
    from ltx_trainer.urng.prune import build_ug
    from ltx_trainer.urng.query import brute_force_query, interval_aware_beam_search
    from ltx_trainer.urng.running_example import FIG2_NODES

    ids = sorted(FIG2_NODES)
    vectors = {k: FIG2_NODES[k]["vector"] for k in ids}
    intervals = {k: FIG2_NODES[k]["interval"] for k in ids}
    qtype = QueryType[args.type.upper()]
    qvec = vectors[args.node]
    qint = intervals[args.node]
    graph = build_ug(
        ids,
        vectors,
        intervals,
        ef_spatial=len(ids),
        ef_attribute=32,
        max_edges_if=64,
        max_edges_is=64,
        iterations=3,
    )
    bf = brute_force_query(vectors, intervals, qvec, qint, qtype, k=args.k)
    ug = interval_aware_beam_search(
        graph,
        vectors,
        intervals,
        qvec,
        qint,
        qtype,
        k=args.k,
        ef_search=args.ef_search,
    )
    print(
        json.dumps(
            {
                "type": qtype.name,
                "query_node": args.node,
                "query_interval": [qint.left, qint.right],
                "k": args.k,
                "brute_force": bf,
                "ug": ug,
                "match": bf == ug,
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo(seed=args.seed)
    out["config"] = UrngConfig().__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="URNG / UG evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    q = sub.add_parser("query")
    q.add_argument("--type", default="IFANN", help="IFANN|ISANN|RFANN|RSANN")
    q.add_argument("--node", default="G", help="Fig.2 node id for query vector/interval")
    q.add_argument("--k", type=int, default=3)
    q.add_argument("--ef-search", type=int, default=32, dest="ef_search")

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
