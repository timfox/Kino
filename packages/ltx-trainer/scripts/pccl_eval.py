#!/usr/bin/env python3
"""PCCL collective synthesizer CLI (Won et al., arXiv:2606.07019)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.pccl.benchmarks import (  # noqa: E402
    parallelism_collectives_table,
    process_group_speedup_anchors,
    scalability_anchors,
    table_i_synthesizer_comparison,
    table_ii_collective_support,
)
from ltx_trainer.pccl.config import PcclConfig  # noqa: E402
from ltx_trainer.pccl.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.pccl.pipeline import run_demo  # noqa: E402
from ltx_trainer.pccl.process_group import speedup_vs_direct  # noqa: E402
from ltx_trainer.pccl.synthesis import synthesize  # noqa: E402


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
                "table_i": table_i_synthesizer_comparison(),
                "table_ii": table_ii_collective_support(),
                "table_iii_parallelism": parallelism_collectives_table(),
                "scalability": scalability_anchors(),
                "process_group": process_group_speedup_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    group = [int(x) for x in args.process_group.split(",")] if args.process_group else None
    cfg = PcclConfig(
        n_npus=args.n_npus,
        mesh_width=args.mesh_width,
        topology=args.topology,
        collective=args.collective,
        process_group=group,
    )
    print(json.dumps(synthesize(cfg), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(run_demo(), indent=2))
    return 0


def _cmd_speedup(args: argparse.Namespace) -> int:
    print(json.dumps(speedup_vs_direct(n_process_groups=args.groups), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.pccl.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="PCCL collective synthesizer stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    sp = sub.add_parser("synthesize")
    sp.add_argument("--n-npus", type=int, default=16)
    sp.add_argument("--mesh-width", type=int, default=4)
    sp.add_argument("--topology", default="2d_mesh", choices=("2d_mesh", "ring", "hypercube"))
    sp.add_argument("--collective", default="all_gather")
    sp.add_argument("--process-group", default="1,2,3")
    sp.set_defaults(func=_cmd_synthesize)

    su = sub.add_parser("speedup")
    su.add_argument("--groups", type=int, default=2)
    su.set_defaults(func=_cmd_speedup)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
