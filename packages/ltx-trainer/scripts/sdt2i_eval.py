#!/usr/bin/env python3
"""SDT2I CLI (Winter et al., arXiv:2502.12691)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.sdt2i.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.sdt2i.config import PAPER_URL, Sdt2iConfig  # noqa: E402
from ltx_trainer.sdt2i.datasets import datasets_card  # noqa: E402
from ltx_trainer.sdt2i.paper import framework_card  # noqa: E402
from ltx_trainer.sdt2i.pipeline import ablation_table_check, evaluation_demo_run, train_step  # noqa: E402


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
    from ltx_trainer.sdt2i.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    variant = getattr(args, "variant", "mstd") or "mstd"
    print(json.dumps(evaluation_demo_run(variant), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_table_check(), indent=2))
    return 0


def _cmd_train_stub(_: argparse.Namespace) -> int:
    print(json.dumps(train_step(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="SDT2I spherical dense T2I stub")
    p.add_argument(
        "command",
        choices=("knowledge", "tables", "dataset", "smoke", "demo", "ablation", "train-stub"),
    )
    p.add_argument("--variant", choices=("mstd", "mpf"), default="mstd", help="demo variant")
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
