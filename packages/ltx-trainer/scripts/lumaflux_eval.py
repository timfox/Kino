#!/usr/bin/env python3
"""LumaFlux SDR→HDR ITM CLI (Saini et al. arXiv:2604.02787)."""

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

from ltx_trainer.lumaflux.config import PAPER_TITLE, PAPER_URL, LumaFluxConfig  # noqa: E402
from ltx_trainer.lumaflux.dataset import dataset_summary  # noqa: E402
from ltx_trainer.lumaflux.losses import LumaFluxLoss  # noqa: E402
from ltx_trainer.lumaflux.model import LumaFlux  # noqa: E402
from ltx_trainer.lumaflux.pipeline import paper_report, save_checkpoint, train_step  # noqa: E402
from ltx_trainer.lumaflux.synthetic import synthesize_pair  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "LumaFlux",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "modules": ["PGA", "PCM", "HDR Residual Coupler", "RQS decoder"],
                "backbone": "frozen Flux MM-DiT (stub)",
                "training_pairs": ds["stats"]["total_sdr_hdr_pairs"],
                "benchmark": "Luma-Eval",
                "tmo_operators": ds["tmo_operators"],
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
    cfg = LumaFluxConfig(
        image_size=args.size,
        use_pga=not args.no_pga,
        use_pcm=not args.no_pcm,
        use_coupler=not args.no_coupler,
    )
    model = LumaFlux(cfg).to(device)
    loss_fn = LumaFluxLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        sdr, hdr = synthesize_pair(args.size, seed=step, crf=31)
        sdr, hdr = sdr.to(device), hdr.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, loss_fn, sdr=sdr, hdr_gt=hdr, t=1.0 - step / max(args.steps, 1))
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss_total']:.4f}")

    save_checkpoint(model, out / "lumaflux.pt")
    summary = {"steps": args.steps, "log_tail": log[-3:]}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = LumaFluxConfig(image_size=args.size)
    model = LumaFlux(cfg).to(device).eval()
    sdr, hdr = synthesize_pair(args.size, seed=args.seed)
    sdr = sdr.to(device)
    with torch.no_grad():
        out = model(sdr, t=0.1)
    print(
        json.dumps(
            {
                "input_shape": list(sdr.shape),
                "hdr_shape": list(out.hdr.shape),
                "hdr_mean": float(out.hdr.mean()),
                "paper_luma_eval_psnr": 36.92,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=_cmd_knowledge)

    sp = sub.add_parser("tables")
    sp.set_defaults(func=_cmd_tables)

    sp = sub.add_parser("dataset")
    sp.set_defaults(func=_cmd_dataset)

    sp = sub.add_parser("train")
    sp.add_argument("-o", "--output-dir", default="lumaflux_train")
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--no-pga", action="store_true")
    sp.add_argument("--no-pcm", action="store_true")
    sp.add_argument("--no-coupler", action="store_true")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("infer")
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_infer)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
