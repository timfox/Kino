#!/usr/bin/env python3
"""Yi evaluation CLI (Liu, He, Tang; arXiv:2607.15576)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.yi.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.yi.config import YiConfig  # noqa: E402
from ltx_trainer.yi.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.yi.pipeline import evaluation_demo, evaluation_smoke, run_update_demo  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Yi in-place graph vector index update CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knowledge", "tables", "smoke", "demo", "plan", "update"):
        sp = sub.add_parser(name)
        if name == "update":
            sp.add_argument("--n-ops", type=int, default=20)
    args = p.parse_args()
    handlers = {
        "knowledge": lambda: framework_card(YiConfig()),
        "tables": benchmarks_bundle,
        "smoke": evaluation_smoke,
        "demo": evaluation_demo,
        "plan": integration_bundle,
        "update": lambda: run_update_demo(n_ops=args.n_ops),
    }
    print(json.dumps(handlers[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
