"""CLI: LongAV-Compass framework card, benchmarks, evaluation demo."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_framework(_: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass import framework_card, task_coverage_table

    print(
        json.dumps(
            {"framework": framework_card(), "task_coverage": task_coverage_table()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_benchmarks(_: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass import benchmarks_bundle

    print(json.dumps(benchmarks_bundle(), indent=2, ensure_ascii=False))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass import evaluation_demo

    print(json.dumps(evaluation_demo(), indent=2, ensure_ascii=False))
    return 0


def _cmd_catalog(_: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass.catalog import benchmark_catalog_outline

    print(json.dumps(benchmark_catalog_outline(), indent=2, ensure_ascii=False))
    return 0


def _cmd_ltx_plan(_: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass import ltx_minute_av_eval_plan, plan_for_case
    from ltx_trainer.longav_compass.annotation import example_t2av_performance_ads_l4

    print(
        json.dumps(
            {
                "plan": ltx_minute_av_eval_plan(),
                "example_case": plan_for_case(example_t2av_performance_ads_l4()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_checkout(_: argparse.Namespace) -> int:
    from gopex_datasets.longav_compass.io import checkout_status

    print(json.dumps(checkout_status(), indent=2, ensure_ascii=False))
    return 0


def _cmd_load_case(args: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass.annotation import case_to_dict
    from ltx_trainer.longav_compass.io import load_case_from_upstream, load_case_json

    if args.path:
        case = load_case_json(args.path)
    else:
        if not args.case_id:
            print("provide case_id or --path", file=sys.stderr)
            return 1
        root = args.root
        if not root:
            from gopex_datasets.longav_compass.io import upstream_root

            up = upstream_root()
            if up is None:
                print("set GOPEX_LONGAV_COMPASS_ROOT or pass --root", file=sys.stderr)
                return 1
            root = str(up)
        case = load_case_from_upstream(root, args.case_id)
    print(json.dumps(case_to_dict(case), indent=2, ensure_ascii=False))
    return 0


def _cmd_eval_case(args: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass.batch import evaluate_case_object, evaluate_upstream_case
    from ltx_trainer.longav_compass.io import load_case_json

    if args.path:
        from ltx_trainer.longav_compass.io import case_from_dict
        import json as _json
        from pathlib import Path

        data = _json.loads(Path(args.path).read_text(encoding="utf-8"))
        out = evaluate_case_object(case_from_dict(data))
    elif args.case_id and args.root:
        out = evaluate_upstream_case(
            args.root,
            args.case_id,
            model_slug=args.model or None,
        )
    elif args.case_id:
        from gopex_datasets.longav_compass.io import upstream_root

        up = upstream_root()
        if up is None:
            print("set GOPEX_LONGAV_COMPASS_ROOT or pass --root", file=sys.stderr)
            return 1
        out = evaluate_upstream_case(up, args.case_id, model_slug=args.model or None)
    else:
        print("provide --path or case_id", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _cmd_eval_batch(args: argparse.Namespace) -> int:
    from gopex_datasets.longav_compass.io import upstream_root
    from ltx_trainer.longav_compass.batch import evaluate_checkout_batch

    root = args.root or upstream_root()
    if root is None:
        print("set GOPEX_LONGAV_COMPASS_ROOT or pass --root", file=sys.stderr)
        return 1
    out = evaluate_checkout_batch(root, limit=int(args.limit))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    from ltx_trainer.longav_compass.annotation import (
        build_generation_prompt,
        example_i2av_performance_ads_l4,
        example_t2av_performance_ads_l4,
        example_v2av_content_creator_l4,
    )

    cases = {
        "t2av": example_t2av_performance_ads_l4(),
        "i2av": example_i2av_performance_ads_l4(),
        "v2av": example_v2av_content_creator_l4(),
    }
    case = cases.get(args.task.lower())
    if case is None:
        print(f"unknown task {args.task!r}; choose t2av, i2av, or v2av", file=sys.stderr)
        return 1
    print(json.dumps(build_generation_prompt(case), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LongAV-Compass benchmark utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fw = sub.add_parser("framework", help="Print framework card and task coverage")
    p_fw.set_defaults(func=_cmd_framework)

    p_b = sub.add_parser("benchmarks", help="Print paper tables bundle")
    p_b.set_defaults(func=_cmd_benchmarks)

    p_d = sub.add_parser("demo", help="Run hierarchical evaluation demo (T2AV/I2AV/V2AV)")
    p_d.set_defaults(func=_cmd_demo)

    p_p = sub.add_parser("prompt", help="Build generation prompt for an example task")
    p_p.add_argument("task", choices=["t2av", "i2av", "v2av", "T2AV", "I2AV", "V2AV"])
    p_p.set_defaults(func=_cmd_prompt)

    p_l = sub.add_parser("ltx-plan", help="Print LTX minute-scale evaluation plan")
    p_l.set_defaults(func=_cmd_ltx_plan)

    p_c = sub.add_parser("catalog", help="Print benchmark catalog outline (counts, layout)")
    p_c.set_defaults(func=_cmd_catalog)

    p_ck = sub.add_parser("checkout", help="Show GOPEX_LONGAV_COMPASS_ROOT inventory")
    p_ck.set_defaults(func=_cmd_checkout)

    p_lc = sub.add_parser("load-case", help="Load annotation.json (path or upstream case_id)")
    p_lc.add_argument("case_id", nargs="?", default=None)
    p_lc.add_argument("--root", default=None, help="Upstream checkout root")
    p_lc.add_argument("--path", default=None, help="Direct path to annotation.json")
    p_lc.set_defaults(func=_cmd_load_case)

    p_ec = sub.add_parser("eval-case", help="Stub-evaluate one case (path or upstream)")
    p_ec.add_argument("case_id", nargs="?", default=None)
    p_ec.add_argument("--root", default=None)
    p_ec.add_argument("--path", default=None)
    p_ec.add_argument("--model", default=None, help="Model slug under results/")
    p_ec.set_defaults(func=_cmd_eval_case)

    p_eb = sub.add_parser("eval-batch", help="Stub-evaluate first N cases in checkout")
    p_eb.add_argument("--root", default=None)
    p_eb.add_argument("--limit", default="8")
    p_eb.set_defaults(func=_cmd_eval_batch)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
