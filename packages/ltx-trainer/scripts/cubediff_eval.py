#!/usr/bin/env python3
"""CubeDiff CLI (Kalischek et al., ICLR 2025, arXiv:2501.17162)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.cubediff.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.cubediff.config import PAPER_URL  # noqa: E402
from ltx_trainer.cubediff.datasets import datasets_card  # noqa: E402
from ltx_trainer.cubediff.paper import framework_card  # noqa: E402
from ltx_trainer.cubediff.pipeline import ablation_table_check, evaluation_demo_run, train_step  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="CubeDiff cubemap panorama generation stub")
    p.add_argument("command", choices=("knowledge", "tables", "dataset", "smoke", "demo", "ablation", "train-stub"))
    args = p.parse_args()
    print(f"# {PAPER_URL}", file=sys.stderr)
    if args.command == "knowledge":
        print(json.dumps(framework_card(), indent=2))
    elif args.command == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.command == "dataset":
        print(json.dumps(datasets_card(), indent=2))
    elif args.command == "smoke":
        from ltx_trainer.cubediff.mock import evaluation_smoke

        print(json.dumps(evaluation_smoke(), indent=2))
    elif args.command == "demo":
        print(json.dumps(evaluation_demo_run(), indent=2))
    elif args.command == "ablation":
        print(json.dumps(ablation_table_check(), indent=2))
    else:
        print(json.dumps(train_step(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
