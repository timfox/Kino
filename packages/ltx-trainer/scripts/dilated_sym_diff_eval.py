#!/usr/bin/env python3
"""Dilated symmetric difference CLI (Urieli, arXiv:2606.06512)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.dilated_sym_diff.benchmarks import fig3_iou_curve, fig4_iou_curve, summary_anchors  # noqa: E402
from ltx_trainer.dilated_sym_diff.config import DilatedSymDiffConfig  # noqa: E402
from ltx_trainer.dilated_sym_diff.layer import layer_card  # noqa: E402
from ltx_trainer.dilated_sym_diff.metrics import metrics_card  # noqa: E402
from ltx_trainer.dilated_sym_diff.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.dilated_sym_diff.morphology import morphology_card  # noqa: E402
from ltx_trainer.dilated_sym_diff.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.dilated_sym_diff.pipeline import run_demo  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"fig3_iou": fig3_iou_curve(), "fig4_iou": fig4_iou_curve(), "summary": summary_anchors()},
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(json.dumps({"morphology": morphology_card(), "metrics": metrics_card(), "layer": layer_card()}, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = DilatedSymDiffConfig(default_radius=args.radius)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dilated symmetric difference binary image comparison stub")
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
    demo.add_argument("--radius", type=int, default=8)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
