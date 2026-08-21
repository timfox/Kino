#!/usr/bin/env python3
"""OB3D CLI (Ito et al., arXiv:2505.20126)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.ob3d.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.ob3d.config import PAPER_URL, OB3DConfig, eval_indices, train_indices  # noqa: E402
from ltx_trainer.ob3d.datasets import datasets_card  # noqa: E402
from ltx_trainer.ob3d.paper import framework_card  # noqa: E402
from ltx_trainer.ob3d.pipeline import (  # noqa: E402
    cpe_demo,
    evaluation_demo_run,
    nvs_demo,
    recon_demo,
    train_step,
    trajectory_demo,
)


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
    from ltx_trainer.ob3d.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_splits(_: argparse.Namespace) -> int:
    print(json.dumps({"train": train_indices(), "eval": eval_indices()}, indent=2))
    return 0


def _cmd_cpe(_: argparse.Namespace) -> int:
    print(json.dumps(cpe_demo(), indent=2))
    return 0


def _cmd_nvs(_: argparse.Namespace) -> int:
    print(json.dumps(nvs_demo(OB3DConfig()), indent=2))
    return 0


def _cmd_recon(_: argparse.Namespace) -> int:
    print(json.dumps(recon_demo(OB3DConfig()), indent=2))
    return 0


def _cmd_trajectory(_: argparse.Namespace) -> int:
    print(json.dumps(trajectory_demo(), indent=2))
    return 0


def _cmd_train_stub(_: argparse.Namespace) -> int:
    print(json.dumps(train_step(OB3DConfig()), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(OB3DConfig()), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"OB3D — {PAPER_URL}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("splits").set_defaults(func=_cmd_splits)
    sub.add_parser("cpe").set_defaults(func=_cmd_cpe)
    sub.add_parser("nvs").set_defaults(func=_cmd_nvs)
    sub.add_parser("recon").set_defaults(func=_cmd_recon)
    sub.add_parser("trajectory").set_defaults(func=_cmd_trajectory)
    sub.add_parser("train-stub").set_defaults(func=_cmd_train_stub)
    sub.add_parser("demo").set_defaults(func=_cmd_demo)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
