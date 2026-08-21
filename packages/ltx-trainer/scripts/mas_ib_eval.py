#!/usr/bin/env python3
"""MAS-IB evaluation CLI (Yu, Zhou, et al., arXiv:2607.16133)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.mas_ib.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.mas_ib.config import MasIbConfig  # noqa: E402
from ltx_trainer.mas_ib.ib import evaluate_relay, mas_gain  # noqa: E402
from ltx_trainer.mas_ib.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.mas_ib.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="MAS-IB multi-agent information bottleneck CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knowledge", "tables", "smoke", "demo", "plan", "gain"):
        sp = sub.add_parser(name)
        if name == "gain":
            sp.add_argument("--beta", type=float, default=1.0)
            sp.add_argument("--delta", type=float, default=0.05)
            sp.add_argument("--h-given-m", type=float, default=56.0)
    args = p.parse_args()
    handlers = {
        "knowledge": lambda: framework_card(MasIbConfig()),
        "tables": benchmarks_bundle,
        "smoke": evaluation_smoke,
        "demo": evaluation_demo,
        "plan": integration_bundle,
        "gain": lambda: {
            "G": mas_gain(h_given_m=args.h_given_m, beta=args.beta, delta=args.delta),
            "relay": evaluate_relay(
                MasIbConfig(beta=args.beta, delta_loss=args.delta)
            ).as_dict(),
        },
    }
    print(json.dumps(handlers[args.cmd](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
