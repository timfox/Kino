#!/usr/bin/env python3
"""VU-BOIQA CLI (Yan et al., arXiv:2503.06129)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.vuboiqa.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.vuboiqa.config import PAPER_URL, VuBoiqaConfig  # noqa: E402
from ltx_trainer.vuboiqa.datasets import datasets_card  # noqa: E402
from ltx_trainer.vuboiqa.paper import framework_card  # noqa: E402
from ltx_trainer.vuboiqa.pipeline import (  # noqa: E402
    ablation_table_check,
    correlation_demo,
    evaluation_demo_run,
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
    from ltx_trainer.vuboiqa.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(VuBoiqaConfig()), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_table_check(), indent=2))
    return 0


def _cmd_correlation(_: argparse.Namespace) -> int:
    print(json.dumps(correlation_demo(), indent=2))
    return 0


def _cmd_train_stub(_: argparse.Namespace) -> int:
    print(json.dumps(train_step(VuBoiqaConfig(num_patches=3, patch_size=48)), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="VU-BOIQA viewport-unaware OIQA stub")
    p.add_argument(
        "command",
        choices=("knowledge", "tables", "dataset", "smoke", "demo", "ablation", "correlation", "train-stub"),
    )
    args = p.parse_args()
    print(f"# {PAPER_URL}", file=sys.stderr)
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "ablation": _cmd_ablation,
        "correlation": _cmd_correlation,
        "train-stub": _cmd_train_stub,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
