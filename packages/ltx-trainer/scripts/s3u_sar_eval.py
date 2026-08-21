#!/usr/bin/env python3
"""S³U-SAR CLI (Yin et al., arXiv:2606.06847)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.s3u_sar.benchmarks import (  # noqa: E402
    dataset_card,
    summary_anchors,
    table_1_baselines,
    table_1_s3u_metrics,
    table_2_ablation,
    table_3_cross_category,
    table_4_orientation,
)
from ltx_trainer.s3u_sar.config import S3USarConfig  # noqa: E402
from ltx_trainer.s3u_sar.losses import loss_components_card  # noqa: E402
from ltx_trainer.s3u_sar.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.s3u_sar.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.s3u_sar.pipeline import run_demo  # noqa: E402
from ltx_trainer.s3u_sar.structure import structure_representation_card  # noqa: E402


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
                "dataset": dataset_card(),
                "table_1": table_1_baselines(),
                "table_1_s3u": table_1_s3u_metrics(),
                "table_2": table_2_ablation(),
                "table_3": table_3_cross_category(),
                "table_4": table_4_orientation(),
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
                "structure": structure_representation_card(),
                "losses": loss_components_card(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    cfg = S3USarConfig(temperature=args.temperature)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="S³U-SAR semantic scattering structure stub")
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
    demo.add_argument("--temperature", type=float, default=1.0)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
