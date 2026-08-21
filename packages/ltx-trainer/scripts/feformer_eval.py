#!/usr/bin/env python3
"""FEFormer volumetric segmentation CLI (Yang et al. arXiv:2605.11434)."""

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

from ltx_trainer.feformer.config import FEFormerConfig, PAPER_TITLE, PAPER_URL  # noqa: E402
from ltx_trainer.feformer.dataset import dataset_summary  # noqa: E402
from ltx_trainer.feformer.model import FEFormer  # noqa: E402
from ltx_trainer.feformer.pipeline import count_parameters, paper_report, train_step  # noqa: E402
from ltx_trainer.feformer.synthetic import synthetic_volume  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "FEFormer",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "modules": ["FDSA", "FGMLP", "WAFF", "FCSB"],
                "datasets": ds["names"],
                "patch_size": ds["patch_size"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(paper_report(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_summary(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = FEFormerConfig(image_size=args.size, num_classes=args.classes)
    model = FEFormer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        x, y = synthetic_volume(batch=1, size=args.size, num_classes=args.classes, seed=step)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, x, y)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 5) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss_seg']:.4f} acc={stats['pixel_acc']:.3f}")

    out = {
        "steps": args.steps,
        "params": count_parameters(model),
        "final": log[-1] if log else {},
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    cfg = FEFormerConfig(image_size=args.size, num_classes=args.classes)
    model = FEFormer(cfg)
    model.eval()
    x, _ = synthetic_volume(batch=1, size=args.size, num_classes=args.classes, seed=args.seed)
    with torch.no_grad():
        pred = model(x)
    print(
        json.dumps(
            {
                "input_shape": list(x.shape),
                "output_shape": list(pred.shape),
                "params": count_parameters(model),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=_cmd_knowledge)

    sp = sub.add_parser("tables")
    sp.set_defaults(func=_cmd_tables)

    sp = sub.add_parser("dataset")
    sp.set_defaults(func=_cmd_dataset)

    sp = sub.add_parser("train")
    sp.add_argument("--steps", type=int, default=50)
    sp.add_argument("--size", type=int, default=32)
    sp.add_argument("--classes", type=int, default=4)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("infer")
    sp.add_argument("--size", type=int, default=32)
    sp.add_argument("--classes", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=_cmd_infer)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
