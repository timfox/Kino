#!/usr/bin/env python3
"""X2HDR SDR→HDR video diffusion CLI (arXiv:2602.04814)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.x2hdr.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle  # noqa: E402
from ltx_trainer.x2hdr.model import X2Hdr, X2HdrConfig  # noqa: E402
from ltx_trainer.x2hdr.paper import framework_card  # noqa: E402
from ltx_trainer.x2hdr.pipeline import evaluation_demo, pipeline_demo, train_step  # noqa: E402
from ltx_trainer.x2hdr.pu21_codec import scene_linear_to_vae_pixels  # noqa: E402
from ltx_trainer.x2hdr.synthetic import synthesize_hdr_scene, synthesize_sdr_from_hdr  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "name": "X2HDR",
                "arxiv": PAPER_ARXIV,
                "paper": PAPER_TITLE,
                "encoding": "pu21",
                "framework": framework_card(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.x2hdr.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(pipeline_demo(device=args.device), indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(device=args.device), indent=2))
    return 0


def _cmd_pu21(args: argparse.Namespace) -> int:
    hdr = synthesize_hdr_scene(args.size, args.size)
    pu21 = scene_linear_to_vae_pixels(hdr)
    print(
        json.dumps(
            {
                "shape": list(pu21.shape),
                "mean": float(pu21.mean()),
                "finite": bool(torch.isfinite(pu21).all()),
            },
            indent=2,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = X2Hdr(X2HdrConfig()).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        hdr = synthesize_hdr_scene(args.size, args.size).to(device)
        sdr = synthesize_sdr_from_hdr(hdr).to(device)
        stats = train_step(model, sdr, hdr, optimizer=opt)
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss']:.4f}")

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "x2hdr.pt")
    print(json.dumps({"steps": args.steps, "log_tail": log[-3:]}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("smoke", _cmd_smoke),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)

    sp = sub.add_parser("demo")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_demo)

    sp = sub.add_parser("eval")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("pu21")
    sp.add_argument("--size", type=int, default=64)
    sp.set_defaults(func=_cmd_pu21)

    sp = sub.add_parser("train")
    sp.add_argument("-o", "--output-dir", default="x2hdr_train")
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_train)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
