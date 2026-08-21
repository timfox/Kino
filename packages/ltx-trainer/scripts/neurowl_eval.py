#!/usr/bin/env python3
"""NeurOWL evaluation CLI (Yang et al., arXiv:2607.15776)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.neurowl.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.neurowl.config import NeurOWLConfig  # noqa: E402
from ltx_trainer.neurowl.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.neurowl.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="NeurOWL incomplete OWL reasoning CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knowledge", "tables", "smoke", "demo", "plan"):
        sub.add_parser(name)
    args = p.parse_args()
    handlers = {
        "knowledge": lambda: framework_card(NeurOWLConfig()),
        "tables": benchmarks_bundle,
        "smoke": evaluation_smoke,
        "demo": evaluation_demo,
        "plan": integration_bundle,
    }
    print(json.dumps(handlers[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
