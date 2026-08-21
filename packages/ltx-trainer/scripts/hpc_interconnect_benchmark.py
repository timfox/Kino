#!/usr/bin/env python3
"""Run HPC interconnect congestion experiment stub (arXiv:2604.11432)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from ltx_trainer.hpc_interconnect.experiment import run_congestion_experiment, sweep_steady_heatmap  # noqa: E402
from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, SystemName  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="HPC interconnect congestion microbenchmark stub")
    p.add_argument("--system", default="leonardo", choices=[s.value for s in SystemName])
    p.add_argument("--nodes", type=int, default=64)
    p.add_argument("--aggressor", default="incast", choices=[a.value for a in AggressorPattern])
    p.add_argument("--message-bytes", type=int, default=32 * 1024)
    p.add_argument("--bursty", action="store_true")
    p.add_argument("--heatmap", action="store_true")
    args = p.parse_args()

    if args.heatmap:
        rows = sweep_steady_heatmap(args.system, aggressor=args.aggressor)
        print(json.dumps(rows, indent=2))
        return

    result = run_congestion_experiment(
        args.system,
        nodes=args.nodes,
        aggressor=args.aggressor,
        message_bytes=args.message_bytes,
        steady=not args.bursty,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
