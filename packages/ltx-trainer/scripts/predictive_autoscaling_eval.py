#!/usr/bin/env python3
"""Predictive autoscaling survey CLI (Kumar et al., arXiv:2606.07046)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.predictive_autoscaling.benchmarks import (  # noqa: E402
    TABLE_I_SURVEY_COMPARISON,
    literature_funnel,
    table_iii_predictive_models,
    table_v_drift_mechanisms,
    table_vi_open_challenges,
)
from ltx_trainer.predictive_autoscaling.drift import drift_demo  # noqa: E402
from ltx_trainer.predictive_autoscaling.kubernetes_native import (  # noqa: E402
    k8s_autoscaler_catalog,
    operator_reconciliation_steps,
    predictive_autoscaler_crd_example,
)
from ltx_trainer.predictive_autoscaling.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.predictive_autoscaling.pipeline import plan_replicas, run_demo  # noqa: E402
from ltx_trainer.predictive_autoscaling.taxonomy import taxonomy_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_taxonomy(_: argparse.Namespace) -> int:
    print(json.dumps(taxonomy_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "literature_funnel": literature_funnel(),
                "table_i": TABLE_I_SURVEY_COMPARISON,
                "table_iii": table_iii_predictive_models(),
                "table_v": table_v_drift_mechanisms(),
                "table_vi": table_vi_open_challenges(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_kubernetes(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "autoscalers": k8s_autoscaler_catalog(),
                "crd_example": predictive_autoscaler_crd_example(),
                "operator_steps": operator_reconciliation_steps(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(run_demo(last_rpm=args.last_rpm, current_replicas=args.replicas), indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from ltx_trainer.predictive_autoscaling.config import PredictiveAutoscalingConfig

    cfg = PredictiveAutoscalingConfig()
    print(
        json.dumps(
            plan_replicas(args.forecast, cfg, current_replicas=args.replicas),
            indent=2,
        )
    )
    return 0


def _cmd_drift(args: argparse.Namespace) -> int:
    observed = [float(x) for x in args.observed.split(",")]
    predicted = [float(x) for x in args.predicted.split(",")]
    print(json.dumps(drift_demo(observed, predicted, tau_adi=args.tau), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.predictive_autoscaling.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Predictive autoscaling survey stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("taxonomy").set_defaults(func=_cmd_taxonomy)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("kubernetes").set_defaults(func=_cmd_kubernetes)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    pd = sub.add_parser("demo")
    pd.add_argument("--last-rpm", type=float, default=7200.0)
    pd.add_argument("--replicas", type=int, default=8)
    pd.set_defaults(func=_cmd_demo)

    pp = sub.add_parser("plan")
    pp.add_argument("forecast", type=float)
    pp.add_argument("--replicas", type=int, default=8)
    pp.set_defaults(func=_cmd_plan)

    dr = sub.add_parser("drift")
    dr.add_argument("--observed", default="0.7,0.75,0.82,0.78")
    dr.add_argument("--predicted", default="0.68,0.72,0.74,0.76")
    dr.add_argument("--tau", type=float, default=0.15)
    dr.set_defaults(func=_cmd_drift)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
