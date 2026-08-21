#!/usr/bin/env python3
"""HD LoRA attention theory CLI (Duranthon et al., arXiv:2606.05899)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.lora_hd_attn.attention import attention_card  # noqa: E402
from ltx_trainer.lora_hd_attn.benchmarks import fig1_curve_rows, summary_anchors  # noqa: E402
from ltx_trainer.lora_hd_attn.config import LoraHdAttnConfig  # noqa: E402
from ltx_trainer.lora_hd_attn.effective_noise import effective_noise_card  # noqa: E402
from ltx_trainer.lora_hd_attn.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.lora_hd_attn.order_params import order_params_card, pretrain_order_params  # noqa: E402
from ltx_trainer.lora_hd_attn.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.lora_hd_attn.pipeline import run_active_ft_demo, run_demo, run_reused_sequences_demo  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps({"fig1": fig1_curve_rows(), "summary": summary_anchors()}, indent=2))
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    cfg = LoraHdAttnConfig()
    pre = pretrain_order_params(cfg)
    print(
        json.dumps(
            {
                "attention": attention_card(),
                "order_params": order_params_card(pre, {"m": 0.0, "q": 0.0, "v": 0.0, "o_w": 0.0}),
                "effective_noise": effective_noise_card(
                    delta=cfg.delta,
                    Q0=pre["Q0"],
                    M=pre["M"],
                    Q=pre["Q"],
                    V=pre["V"],
                    T=cfg.T,
                    e=cfg.e,
                ),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = LoraHdAttnConfig(
        D=args.D,
        T=args.T,
        alpha=args.alpha,
        alpha_prime=args.alpha_prime,
        delta=args.delta,
        lam=args.lam,
        lam_prime=args.lam_prime,
        e=args.e,
    )
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_active(_: argparse.Namespace) -> int:
    print(json.dumps(run_active_ft_demo(), indent=2))
    return 0


def _cmd_reused(_: argparse.Namespace) -> int:
    print(json.dumps(run_reused_sequences_demo(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HD LoRA fine-tuning in solvable attention stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("active", _cmd_active),
        ("reused", _cmd_reused),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--D", type=int, default=150)
    demo.add_argument("--T", type=int, default=3)
    demo.add_argument("--alpha", type=float, default=0.1)
    demo.add_argument("--alpha-prime", dest="alpha_prime", type=float, default=3.0)
    demo.add_argument("--delta", type=float, default=0.5)
    demo.add_argument("--lam", type=float, default=0.01)
    demo.add_argument("--lam-prime", dest="lam_prime", type=float, default=0.05)
    demo.add_argument("--e", type=int, default=0, choices=(0, 1))
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
