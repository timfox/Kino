#!/usr/bin/env python3
"""Train CineMatte on synthetic compositing (distractor strategy)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: E402

import torch  # noqa: E402

from ltx_trainer.cinematte.losses import CineMatteLoss  # noqa: E402
from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig  # noqa: E402
from ltx_trainer.cinematte.synthetic import composite_training_sample  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train CineMatte (synthetic distractor compositing)")
    p.add_argument("-o", "--output-dir", default="cinematte_train")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--backbone", default="stub")
    p.add_argument("--fbam-layers", type=int, default=2)
    p.add_argument("--lr-fbam", type=float, default=1e-5)
    p.add_argument("--lr-decoder", type=float, default=1e-5)
    p.add_argument("--lr-upsampler", type=float, default=1e-6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--augment", action="store_true", help="Color jitter / flip / affine (Sec. 5.1)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = CineMatteConfig(backbone=args.backbone, fbam_layers=args.fbam_layers)
    model = CineMatte(cfg).to(device)
    loss_fn = CineMatteLoss()

    upsampler_params = list(model.upsampler.parameters())
    trainable_ids = {id(p) for p in upsampler_params}
    fbam_decoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in trainable_ids]

    opt = torch.optim.Adam(
        [
            {"params": fbam_decoder_params, "lr": args.lr_fbam},
            {"params": upsampler_params, "lr": args.lr_upsampler},
        ]
    )

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        img, bg, alpha = composite_training_sample(
            args.size, args.size, augment=args.augment, device=torch.device(device)
        )
        pred = model(img.unsqueeze(0), bg.unsqueeze(0)).squeeze(1)
        loss, stats = loss_fn(pred, alpha.unsqueeze(0))
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss_total']:.4f}")

    ckpt = {"config": asdict(cfg), "state_dict": model.state_dict()}
    torch.save(ckpt, out / "cinematte_checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")
    print(f"Saved {out / 'cinematte_checkpoint.pt'}")


if __name__ == "__main__":
    main()
