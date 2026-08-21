#!/usr/bin/env python3
"""PagedWeight evaluation CLI (Yang et al., arXiv:2607.16184)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.pagedweight.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.pagedweight.config import PagedWeightConfig  # noqa: E402
from ltx_trainer.pagedweight.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.pagedweight.pipeline import (  # noqa: E402
    evaluation_demo,
    evaluation_smoke,
    run_serving_demo,
)


def main() -> int:
    p = argparse.ArgumentParser(description="PagedWeight MoE serving CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knowledge", "tables", "smoke", "demo", "plan", "serve"):
        sp = sub.add_parser(name)
        if name == "serve":
            sp.add_argument("--free-blocks", type=int, default=2)
    args = p.parse_args()
    handlers = {
        "knowledge": lambda: framework_card(PagedWeightConfig()),
        "tables": benchmarks_bundle,
        "smoke": evaluation_smoke,
        "demo": evaluation_demo,
        "plan": integration_bundle,
        "serve": lambda: run_serving_demo(free_blocks=args.free_blocks),
    }
    print(json.dumps(handlers[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
