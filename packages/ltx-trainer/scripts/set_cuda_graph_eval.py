#!/usr/bin/env python3
"""SET CUDA Graph scheduling CLI (Li et al., arXiv:2606.05495)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.set_cuda_graph.benchmarks import (  # noqa: E402
    fig6_overhead_anchors,
    summary_anchors,
    table_1_averages,
    table_1_speedups,
    table_2_overhead,
)
from ltx_trainer.set_cuda_graph.config import SetCudaGraphConfig  # noqa: E402
from ltx_trainer.set_cuda_graph.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.set_cuda_graph.overhead import overhead_model_card  # noqa: E402
from ltx_trainer.set_cuda_graph.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.set_cuda_graph.pipeline import run_demo  # noqa: E402
from ltx_trainer.set_cuda_graph.scheduling import algorithms_summary, runtime_components_card
from ltx_trainer.set_cuda_graph.workloads import workload_catalog


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
                "table_1": table_1_speedups(),
                "table_1_averages": table_1_averages(),
                "table_2": table_2_overhead(),
                "fig_6": fig6_overhead_anchors(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "overhead_model": overhead_model_card(),
                "runtime": runtime_components_card(),
                "algorithms": algorithms_summary(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_workloads(_: argparse.Namespace) -> int:
    print(json.dumps(workload_catalog(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = SetCudaGraphConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
        gpu=args.gpu,
    )
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SET CUDA graph scheduling stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("workloads", _cmd_workloads),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--batch-size", type=int, default=8)
    demo.add_argument("--workers", type=int, default=8)
    demo.add_argument("--gpu", choices=("rtx_3090", "rtx_5090"), default="rtx_3090")
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
