#!/usr/bin/env python3
"""LLM C++→CUDA Deopt-Reopt CLI (Mukunoki et al., arXiv:2606.06063)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.deopt_reopt.benchmarks import (  # noqa: E402
    conv2d_highlight,
    positioning_vs_direct,
    summary_anchors,
    table_1_kernels,
    table_2_comparison,
)
from ltx_trainer.deopt_reopt.config import DeoptReoptConfig  # noqa: E402
from ltx_trainer.deopt_reopt.kernels import kernel_by_name, kernel_catalog  # noqa: E402
from ltx_trainer.deopt_reopt.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.deopt_reopt.pipeline import run_demo  # noqa: E402
from ltx_trainer.deopt_reopt.statistics import (  # noqa: E402
    iterative_significance_summary,
    single_shot_significance_summary,
)
from ltx_trainer.deopt_reopt.workflows import single_shot_phases, workflow_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(args: argparse.Namespace) -> int:
    wf = args.workflow
    print(
        json.dumps(
            {
                "table_1": table_1_kernels(),
                "table_2": table_2_comparison(wf),
                "positioning": positioning_vs_direct(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_workflows(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "workflows": workflow_card(),
                "single_shot_phases": {
                    m: single_shot_phases(m) for m in ("direct", "deopt_reopt", "direct_3")
                },
            },
            indent=2,
        )
    )
    return 0


def _cmd_stats(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "single_shot": single_shot_significance_summary(),
                "iterative": iterative_significance_summary(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_kernel(args: argparse.Namespace) -> int:
    print(json.dumps(kernel_by_name(args.name), indent=2))
    return 0


def _cmd_conv2d(_: argparse.Namespace) -> int:
    print(json.dumps(conv2d_highlight(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(run_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.deopt_reopt.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_kernels(_: argparse.Namespace) -> int:
    print(json.dumps(kernel_catalog(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deopt-Reopt LLM C++→CUDA porting stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    t = sub.add_parser("tables")
    t.add_argument("--workflow", choices=("single_shot", "iterative"), default="single_shot")
    t.set_defaults(func=_cmd_tables)
    sub.add_parser("workflows").set_defaults(func=_cmd_workflows)
    sub.add_parser("stats").set_defaults(func=_cmd_stats)
    k = sub.add_parser("kernel")
    k.add_argument("name")
    k.set_defaults(func=_cmd_kernel)
    sub.add_parser("conv2d").set_defaults(func=_cmd_conv2d)
    sub.add_parser("kernels").set_defaults(func=_cmd_kernels)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
