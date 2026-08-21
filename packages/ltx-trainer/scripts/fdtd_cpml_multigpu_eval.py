#!/usr/bin/env python3
"""FDTD+CPML multi-GPU communication CLI (Obieke, arXiv:2606.06910)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.fdtd_cpml_multigpu.benchmarks import (  # noqa: E402
    summary_anchors,
    table_10_single_gpu,
    table_11_strong_scaling_rtx8000,
    table_i_positioning,
)
from ltx_trainer.fdtd_cpml_multigpu.communication import (  # noqa: E402
    compare_exchange,
    enlarged_ghost_sweep,
)
from ltx_trainer.fdtd_cpml_multigpu.config import FdtdCpmlConfig  # noqa: E402
from ltx_trainer.fdtd_cpml_multigpu.cpml import cpml_card  # noqa: E402
from ltx_trainer.fdtd_cpml_multigpu.decomposition import select_best_decomposition  # noqa: E402
from ltx_trainer.fdtd_cpml_multigpu.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.fdtd_cpml_multigpu.pipeline import run_demo  # noqa: E402


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
                "table_i": table_i_positioning(),
                "table_10": table_10_single_gpu(),
                "table_11": table_11_strong_scaling_rtx8000(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_decomposition(args: argparse.Namespace) -> int:
    print(json.dumps(select_best_decomposition(args.grid), indent=2))
    return 0


def _cmd_exchange(args: argparse.Namespace) -> int:
    print(json.dumps(compare_exchange(args.grid), indent=2))
    return 0


def _cmd_ghost(args: argparse.Namespace) -> int:
    print(json.dumps(enlarged_ghost_sweep(args.grid), indent=2))
    return 0


def _cmd_cpml(_: argparse.Namespace) -> int:
    print(json.dumps(cpml_card(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(run_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.fdtd_cpml_multigpu.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FDTD+CPML multi-GPU communication stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("cpml").set_defaults(func=_cmd_cpml)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    dp = sub.add_parser("decomposition")
    dp.add_argument("--grid", type=int, default=800)
    dp.set_defaults(func=_cmd_decomposition)

    ex = sub.add_parser("exchange")
    ex.add_argument("--grid", type=int, default=800)
    ex.set_defaults(func=_cmd_exchange)

    gh = sub.add_parser("ghost")
    gh.add_argument("--grid", type=int, default=800)
    gh.set_defaults(func=_cmd_ghost)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
