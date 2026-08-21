#!/usr/bin/env python3
"""DiffTestGen evaluation CLI (Hu, Cadar, Pradel; arXiv:2607.16024)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.difftestgen.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.difftestgen.config import DiffTestGenConfig  # noqa: E402
from ltx_trainer.difftestgen.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.difftestgen.pipeline import (  # noqa: E402
    evaluation_demo,
    evaluation_smoke,
    run_access_demo,
)


def main() -> int:
    p = argparse.ArgumentParser(description="DiffTestGen change-directed differential testing CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knowledge", "tables", "smoke", "demo", "plan", "access"):
        sub.add_parser(name)
    args = p.parse_args()
    handlers = {
        "knowledge": lambda: framework_card(DiffTestGenConfig()),
        "tables": benchmarks_bundle,
        "smoke": evaluation_smoke,
        "demo": evaluation_demo,
        "plan": integration_bundle,
        "access": run_access_demo,
    }
    print(json.dumps(handlers[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
