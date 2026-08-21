#!/usr/bin/env python3
"""GPU midsize multi-precision division CLI (Marchioro et al., arXiv:2606.06386)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.midint_division.benchmarks import (  # noqa: E402
    positioning_vs_cgbn,
    summary_anchors,
    table_1_cgbn_comparison,
)
from ltx_trainer.midint_division.config import MidintDivisionConfig  # noqa: E402
from ltx_trainer.midint_division.cost_model import full_mult_bounds  # noqa: E402
from ltx_trainer.midint_division.division import divide, verify_example  # noqa: E402
from ltx_trainer.midint_division.gpu_mapping import block_layout_card  # noqa: E402
from ltx_trainer.midint_division.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.midint_division.pipeline import run_demo  # noqa: E402


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
                "table_1": table_1_cgbn_comparison(),
                "positioning": positioning_vs_cgbn(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_cost(_: argparse.Namespace) -> int:
    print(json.dumps(full_mult_bounds(), indent=2))
    return 0


def _cmd_layout(args: argparse.Namespace) -> int:
    print(json.dumps(block_layout_card(args.bits_exp), indent=2))
    return 0


def _cmd_divide(args: argparse.Namespace) -> int:
    q, r, meta = divide(args.u, args.v, base=args.base)
    print(json.dumps({"q": q, "r": r, "meta": meta}, indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(run_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.midint_division.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_examples(_: argparse.Namespace) -> int:
    from ltx_trainer.midint_division.constants import PAPER_EXAMPLES

    out = [verify_example(ex["u"], ex["v"], ex["q"], base=ex["B"]) for ex in PAPER_EXAMPLES]
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Midsize GPU integer division stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("cost").set_defaults(func=_cmd_cost)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("examples").set_defaults(func=_cmd_examples)

    ly = sub.add_parser("layout")
    ly.add_argument("--bits-exp", type=int, default=18)
    ly.set_defaults(func=_cmd_layout)

    dv = sub.add_parser("divide")
    dv.add_argument("--u", type=int, required=True)
    dv.add_argument("--v", type=int, required=True)
    dv.add_argument("--base", type=int, default=10)
    dv.set_defaults(func=_cmd_divide)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
