#!/usr/bin/env python3
"""OMAF fast multirate 360° encoding CLI (Premkumar & Herglotz, arXiv:2601.17568)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.oma_fme.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.oma_fme.config import OmaFmeConfig, PAPER_URL  # noqa: E402
from ltx_trainer.oma_fme.datasets import sjtu_dataset_card  # noqa: E402
from ltx_trainer.oma_fme.paper import paper_knowledge  # noqa: E402
from ltx_trainer.oma_fme.pipeline import run_pipeline  # noqa: E402
from ltx_trainer.oma_fme.synthetic import synthetic_variant_comparison  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(paper_knowledge(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(sjtu_dataset_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.oma_fme.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    cfg = OmaFmeConfig(variant=args.variant, anchor_quality=args.anchor)
    out = run_pipeline(cfg)
    print(json.dumps({"paper": PAPER_URL, **out}, indent=2))
    return 0


def _cmd_synthetic(_: argparse.Namespace) -> int:
    print(json.dumps(synthetic_variant_comparison(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fast multirate 360° encoding for OMAF (arXiv:2601.17568)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("synthetic").set_defaults(func=_cmd_synthetic)
    plan = sub.add_parser("plan")
    plan.add_argument("--variant", default="ERP-CRC", help="ERP-CRC, ERP-PRA, CMP-CRC, CMP-PRA")
    plan.add_argument("--anchor", default="HQ", choices=("LQ", "MQ", "HQ"))
    plan.set_defaults(func=_cmd_plan)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
