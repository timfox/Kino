#!/usr/bin/env python3
"""Train EIC-LIE on synthetic low-light / event / normal-light triplets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: E402, F401

import torch  # noqa: E402

from ltx_trainer.eic_lie.losses import EicLieLoss  # noqa: E402
from ltx_trainer.eic_lie.model import EicLie, EicLieConfig  # noqa: E402
from ltx_trainer.eic_lie.pipeline import save_checkpoint  # noqa: E402
from ltx_trainer.eic_lie.synthetic import augment_pair, synthesize_low_light_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train EIC-LIE (synthetic / RLE-style)")
    p.add_argument("-o", "--output-dir", default="eic_lie_train")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--event-bins", type=int, default=5)
    p.add_argument("--eici-stages", default="2,2,2", help="Comma-separated EICI+IAEF repeats")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    stages = tuple(int(x) for x in args.eici_stages.split(",") if x.strip())
    cfg = EicLieConfig(
        base_channels=args.channels,
        event_bins=args.event_bins,
        eici_stages=stages,
        iaef_lite=True,
    )
    model = EicLie(cfg).to(device)
    loss_fn = EicLieLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        low, gt, voxel = synthesize_low_light_pair(args.size, args.size, event_bins=args.event_bins, device=torch.device(device))
        if args.augment:
            low, gt, voxel = augment_pair(low, gt, voxel)
        pred = model(low.unsqueeze(0), voxel.unsqueeze(0)).squeeze(0)
        loss, stats = loss_fn(pred, gt)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss_total']:.4f} ssim={stats['ssim']:.4f}")

    save_checkpoint(model, out / "eic_lie_checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Saved {out / 'eic_lie_checkpoint.pt'}")


if __name__ == "__main__":
    main()
