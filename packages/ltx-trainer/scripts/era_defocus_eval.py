#!/usr/bin/env python3
"""ErA defocus deblurring CLI (Vo & Park, arXiv:2606.06540)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.era_defocus.benchmarks import dataset_card, summary_anchors, table_1_results  # noqa: E402
from ltx_trainer.era_defocus.config import EraDefocusConfig  # noqa: E402
from ltx_trainer.era_defocus.losses import loss_components_card  # noqa: E402
from ltx_trainer.era_defocus.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.era_defocus.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.era_defocus.pipeline import run_demo  # noqa: E402
from ltx_trainer.era_defocus.unrolling import unrolling_block_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps({"dataset": dataset_card(), "table_1": table_1_results(), "summary": summary_anchors()}, indent=2))
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(json.dumps({"unrolling": unrolling_block_card(), "losses": loss_components_card()}, indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    cfg = EraDefocusConfig(unrolling_depth=args.depth, omega_recon=args.omega)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ErA error-aware defocus deblurring stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--depth", type=int, default=10)
    demo.add_argument("--omega", type=float, default=0.5)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
