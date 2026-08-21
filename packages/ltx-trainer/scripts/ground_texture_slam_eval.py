#!/usr/bin/env python3
"""Evaluate multi-session ground texture SLAM methods (Hart & Englot arXiv:2605.19701)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.ground_texture_slam.dataset import dataset_info  # noqa: E402
from ltx_trainer.ground_texture_slam.metrics import table_i_rmse  # noqa: E402
from ltx_trainer.ground_texture_slam.pipeline import evaluate_sessions  # noqa: E402
from ltx_trainer.ground_texture_slam.schema import DATASET_SESSIONS, LoopClosureMethod  # noqa: E402
from ltx_trainer.ground_texture_slam.synthetic import synthesize_session  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-session ground texture SLAM eval")
    p.add_argument(
        "--method",
        choices=[m.value for m in LoopClosureMethod],
        default=LoopClosureMethod.KLD.value,
    )
    p.add_argument("--sessions", type=int, default=DATASET_SESSIONS)
    p.add_argument("--poses", type=int, default=40)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--table-i", action="store_true", help="Print paper Table I RMSE")
    p.add_argument("--dataset-info", action="store_true")
    args = p.parse_args()

    if args.dataset_info:
        print(json.dumps(dataset_info(), indent=2))
        return
    if args.table_i:
        print(json.dumps(table_i_rmse(), indent=2))
        return

    method = LoopClosureMethod(args.method)
    sessions = [synthesize_session(k, size=args.size, n_poses=args.poses) for k in range(args.sessions)]
    report = evaluate_sessions(sessions, method=method)
    report["method"] = method.value
    report["paper_table_i"] = table_i_rmse().get(method.value, {})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
