"""CLI for LiveBrowseComp (arXiv:2605.28721)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.livebrowsecomp.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
    score_drop_analysis,
)


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LiveBrowseComp — dataset load, eval, IKD diagnostics")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("framework")
    sub.add_parser("knowledge")
    sub.add_parser("benchmarks")
    sub.add_parser("drops")
    sub.add_parser("checkout")

    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--limit", type=int, default=5)
    d.add_argument("--no-download", action="store_true")

    ld = sub.add_parser("load")
    ld.add_argument("--path", type=str, default=None)
    ld.add_argument("--limit", type=int, default=3)
    ld.add_argument("--no-download", action="store_true")

    ev = sub.add_parser("eval")
    ev.add_argument("submission", type=str, help="JSONL with idx + samples[]")
    ev.add_argument("--dataset", type=str, default=None)
    ev.add_argument("--limit", type=int, default=None)

    dc = sub.add_parser("decrypt")
    dc.add_argument("ciphertext", type=str)

    args = p.parse_args(argv)

    if args.command == "framework":
        _print(framework_card())
    elif args.command == "knowledge":
        _print(knowledge_card())
    elif args.command == "benchmarks":
        _print(benchmarks_bundle())
    elif args.command == "drops":
        _print(score_drop_analysis())
    elif args.command == "checkout":
        from ltx_trainer.livebrowsecomp.io import checkout_status

        _print(checkout_status())
    elif args.command == "demo":
        _print(evaluation_demo(seed=args.seed, load_limit=args.limit, download=not args.no_download))
    elif args.command == "load":
        from ltx_trainer.livebrowsecomp.io import dataset_stats, load_items

        items = load_items(args.path, download=not args.no_download, limit=args.limit)
        _print({"stats": dataset_stats(items), "items": [i.to_dict() for i in items]})
    elif args.command == "eval":
        from ltx_trainer.livebrowsecomp.eval import evaluate_submission_file

        report = evaluate_submission_file(
            args.submission,
            dataset_path=args.dataset,
            limit=args.limit,
        )
        _print(report.to_dict())
    elif args.command == "decrypt":
        from ltx_trainer.livebrowsecomp.crypto import decrypt_string

        print(decrypt_string(args.ciphertext))
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
