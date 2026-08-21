#!/usr/bin/env python3
"""ErpGS CLI (Ito et al., arXiv:2505.19883)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.erpgs.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.erpgs.config import PAPER_URL, ErpGSConfig  # noqa: E402
from ltx_trainer.erpgs.datasets import datasets_card  # noqa: E402
from ltx_trainer.erpgs.paper import framework_card  # noqa: E402
from ltx_trainer.erpgs.pipeline import (  # noqa: E402
    ablation_table_check,
    evaluation_demo_run,
    nvs_metrics_demo,
    train_step,
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
    from ltx_trainer.erpgs.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(ErpGSConfig()), indent=2))
    return 0


def _cmd_nvs(_: argparse.Namespace) -> int:
    print(json.dumps(nvs_metrics_demo(ErpGSConfig()), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_table_check(), indent=2))
    return 0


def _cmd_train_stub(_: argparse.Namespace) -> int:
    print(json.dumps(train_step(ErpGSConfig()), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ErpGS omnidirectional 3DGS stub")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("dataset", _cmd_dataset),
        ("smoke", _cmd_smoke),
        ("demo", _cmd_demo),
        ("nvs", _cmd_nvs),
        ("ablation", _cmd_ablation),
        ("train-stub", _cmd_train_stub),
    ]:
        sub.add_parser(name).set_defaults(func=fn)
    args = p.parse_args()
    print(f"# ErpGS {PAPER_URL}", file=sys.stderr)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
