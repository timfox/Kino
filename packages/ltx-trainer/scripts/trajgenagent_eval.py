#!/usr/bin/env python3
"""TrajGenAgent evaluation CLI (Li et al., arXiv:2606.12657)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.trajgenagent.anomaly_eval import anomaly_evaluation_summary  # noqa: E402
from ltx_trainer.trajgenagent.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig  # noqa: E402
from ltx_trainer.trajgenagent.integration import framework_card, integration_bundle  # noqa: E402
from ltx_trainer.trajgenagent.pipeline import evaluation_demo, evaluation_smoke  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(TrajGenAgentConfig()), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(), indent=2, default=str))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.workflow import generate_daily_trajectory

    result = generate_daily_trajectory(
        individual_id=args.individual,
        weekday=args.weekday,
        day_type=args.day_type,
        seed=args.seed,
    )
    traj = result["stage2"]["trajectory"]
    payload = {
        "activity_chain": result["activity_chain"],
        "n_visits": result["stage2"]["n_visits"],
        "trajectory_success": result["stage2"]["trajectory_success"],
        "visits": [
            {
                "activity": v.activity,
                "poi_id": v.poi_id,
                "start": v.start.isoformat(),
                "end": v.end.isoformat(),
                "duration_minutes": v.duration_minutes,
            }
            for v in traj.visits
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.ablation import ablation_summary

    print(json.dumps(ablation_summary(), indent=2, default=str))
    return 0


def _cmd_anomaly(_: argparse.Namespace) -> int:
    print(json.dumps(anomaly_evaluation_summary(TrajGenAgentConfig()), indent=2))
    return 0


def _cmd_evidence(args: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.evidence import evidence_demo

    print(json.dumps(evidence_demo(seed=args.seed), indent=2, default=str))
    return 0


def _cmd_stability(_: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.tool_stability import tool_stability_comparison

    print(json.dumps(tool_stability_comparison(), indent=2))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.batch_eval import dual_dataset_batch_summary

    print(json.dumps(dual_dataset_batch_summary(n=args.n, seed=args.seed), indent=2))
    return 0


def _cmd_ingest(_: argparse.Namespace) -> int:
    from ltx_trainer.trajgenagent.ingest import ingest_demo

    print(json.dumps(ingest_demo(), indent=2, default=str))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="TrajGenAgent mobility trajectory generation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("demo")
    sub.add_parser("plan")
    sub.add_parser("ablation")
    sub.add_parser("anomaly")
    ev = sub.add_parser("evidence")
    ev.add_argument("--seed", type=int, default=42)
    sub.add_parser("stability")
    batch = sub.add_parser("batch")
    batch.add_argument("--n", type=int, default=12)
    batch.add_argument("--seed", type=int, default=0)
    sub.add_parser("ingest")
    gen = sub.add_parser("generate", help="Run orchestrator + worker workflow stub")
    gen.add_argument("--individual", default="u_001")
    gen.add_argument("--weekday", default="Monday")
    gen.add_argument("--day-type", dest="day_type", default="weekday")
    gen.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "demo": _cmd_demo,
        "plan": _cmd_plan,
        "ablation": _cmd_ablation,
        "anomaly": _cmd_anomaly,
        "evidence": _cmd_evidence,
        "stability": _cmd_stability,
        "batch": _cmd_batch,
        "ingest": _cmd_ingest,
        "generate": _cmd_generate,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
