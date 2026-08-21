#!/usr/bin/env python3
"""Demystifying NVSHMEM CLI (Ma et al., arXiv:2606.05951)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.nvshmem_demystify.benchmarks import (  # noqa: E402
    figure4_rma,
    figure5_allreduce,
    positioning_vs_nccl,
    summary_anchors,
)
from ltx_trainer.nvshmem_demystify.collectives import collective_table, psync_card
from ltx_trainer.nvshmem_demystify.config import NvshmemDemystifyConfig  # noqa: E402
from ltx_trainer.nvshmem_demystify.deepep import deepep_overview, ht_path_card, ll_path_card
from ltx_trainer.nvshmem_demystify.memory import heap_growth_steps, symmetric_heap_card
from ltx_trainer.nvshmem_demystify.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.nvshmem_demystify.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.nvshmem_demystify.pipeline import run_demo  # noqa: E402
from ltx_trainer.nvshmem_demystify.rma import rma_performance_summary, transport_backends


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
                "table_ii_collectives": collective_table(),
                "figure_4": figure4_rma(),
                "figure_5": figure5_allreduce(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_memory(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "symmetric_heap": symmetric_heap_card(),
                "growth_steps": heap_growth_steps(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_rma(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "performance": rma_performance_summary(),
                "transports": transport_backends(),
                "positioning": positioning_vs_nccl(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_collectives(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "table_ii": collective_table(),
                "psync": psync_card(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_deepep(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "overview": deepep_overview(),
                "ht": ht_path_card(),
                "ll": ll_path_card(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = NvshmemDemystifyConfig(
        pe_count=args.pe_count,
        include_deepep=not args.no_deepep,
        ibgda_tuned=not args.no_ibgda_tune,
    )
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NVSHMEM demystify study stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("memory", _cmd_memory),
        ("rma", _cmd_rma),
        ("collectives", _cmd_collectives),
        ("deepep", _cmd_deepep),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--pe-count", type=int, default=8)
    demo.add_argument("--no-deepep", action="store_true")
    demo.add_argument("--no-ibgda-tune", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
