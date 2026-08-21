#!/usr/bin/env python3
"""EDM CLI (Jung et al., arXiv:2502.20685)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.edm.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.edm.config import PAPER_URL, EdmConfig  # noqa: E402
from ltx_trainer.edm.datasets import datasets_card  # noqa: E402
from ltx_trainer.edm.paper import framework_card  # noqa: E402
from ltx_trainer.edm.pipeline import ablation_table_check, evaluation_demo_run, train_step  # noqa: E402


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
    from ltx_trainer.edm.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(EdmConfig()), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_table_check(), indent=2))
    return 0


def _cmd_train_stub(_: argparse.Namespace) -> int:
    print(json.dumps(train_step(EdmConfig()), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="EDM dense ERP matching stub")
    p.add_argument("command", choices=("knowledge", "tables", "dataset", "smoke", "demo", "ablation", "train-stub"))
    args = p.parse_args()
    print(f"# {PAPER_URL}", file=sys.stderr)
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "ablation": _cmd_ablation,
        "train-stub": _cmd_train_stub,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
