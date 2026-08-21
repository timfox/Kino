#!/usr/bin/env python3
"""HypeVPR CLI (Woo et al., arXiv:2506.04764)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.hypevpr.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.hypevpr.config import PAPER_URL, HypeVPRConfig  # noqa: E402
from ltx_trainer.hypevpr.datasets import datasets_card  # noqa: E402
from ltx_trainer.hypevpr.paper import framework_card  # noqa: E402
from ltx_trainer.hypevpr.pipeline import (  # noqa: E402
    ablation_tables,
    evaluation_demo_run,
    norm_hierarchy_demo,
    retrieval_demo,
    train_step,
)
from ltx_trainer.hypevpr.poincare import expmap0, poincare_distance  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.hypevpr.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_poincare(_: argparse.Namespace) -> int:
    import torch

    c = 1.0
    a = expmap0(torch.tensor([[0.1, 0.2, 0.0]]), c=c)
    b = expmap0(torch.tensor([[0.2, 0.1, 0.0]]), c=c)
    d = float(poincare_distance(a, b, c=c).item())
    print(json.dumps({"distance": d, "norm_a": float(a.norm())}, indent=2))
    return 0


def _cmd_retrieval(args: argparse.Namespace) -> int:
    cfg = HypeVPRConfig(query_size=args.size)
    print(json.dumps(retrieval_demo(cfg), indent=2))
    return 0


def _cmd_norms(args: argparse.Namespace) -> int:
    cfg = HypeVPRConfig(query_size=args.size, hierarchy_levels=args.levels)
    print(json.dumps(norm_hierarchy_demo(cfg), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = HypeVPRConfig(query_size=args.size)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = HypeVPRConfig(query_size=args.size)
    print(json.dumps(evaluation_demo_run(cfg), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_tables(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"HypeVPR — {PAPER_URL}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("poincare").set_defaults(func=_cmd_poincare)
    ab = sub.add_parser("ablation")
    ab.set_defaults(func=_cmd_ablation)
    for name, fn in (
        ("retrieval", _cmd_retrieval),
        ("norms", _cmd_norms),
        ("train-stub", _cmd_train_stub),
        ("demo", _cmd_demo),
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--size", type=int, default=64)
        sp.add_argument("--levels", type=int, default=4)
        sp.set_defaults(func=fn)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
