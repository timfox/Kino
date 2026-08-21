#!/usr/bin/env python3
"""Train UniMatch V2 semi-supervised seg on synthetic WeatherProof pairs."""

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

from ltx_trainer.weatherproof.losses import SemiSupervisedLoss, SemiSupervisedLossConfig  # noqa: E402
from ltx_trainer.weatherproof.metrics import mean_dice, mean_iou  # noqa: E402
from ltx_trainer.weatherproof.model import UniMatchV2Seg  # noqa: E402
from ltx_trainer.weatherproof.pipeline import save_checkpoint, train_step  # noqa: E402
from ltx_trainer.weatherproof.synthetic import synthesize_pair  # noqa: E402
from ltx_trainer.weatherproof.tta import predict_with_tta  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train WeatherProof UniMatch V2")
    p.add_argument("-o", "--output-dir", default="weatherproof_train")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda-unsup", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = UniMatchV2Seg().to(device)
    loss_fn = SemiSupervisedLoss(SemiSupervisedLossConfig(lambda_unsup=args.lambda_unsup))
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr * 40.0)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        clean, degraded, mask = synthesize_pair(size=args.size)
        clean, degraded, mask = clean.to(device), degraded.to(device), mask.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, loss_fn, clean=clean, mask=mask, degraded=degraded)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(
                f"step {step}/{args.steps} total={stats['loss_total']:.4f} "
                f"clean={stats['loss_clean']:.4f} deg={stats['loss_degraded']:.4f}"
            )

    clean, degraded, mask = synthesize_pair(size=args.size)
    clean, mask = clean.to(device), mask.to(device)
    pred = predict_with_tta(model, clean)
    miou = mean_iou(pred, mask)
    mdice = mean_dice(pred, mask)
    save_checkpoint(model, out / "weatherproof.pt")
    summary = {"miou_tta": miou, "mdice_tta": mdice}
    (out / "train_log.json").write_text(json.dumps({"log": log[-20:], "eval": summary}, indent=2))
    print(f"Saved {out / 'weatherproof.pt'} mIoU={miou:.3f} mDice={mdice:.3f}")


if __name__ == "__main__":
    main()
