#!/usr/bin/env python3
"""Train FogNet on synthetic FogAct pairs (Liu et al. arXiv:2605.20645)."""

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

from ltx_trainer.fognet.classes import FOGACT_CLASSES  # noqa: E402
from ltx_trainer.fognet.losses import FogNetLoss, FogNetLossConfig  # noqa: E402
from ltx_trainer.fognet.metrics import top1_accuracy, top5_accuracy  # noqa: E402
from ltx_trainer.fognet.model import FogNet  # noqa: E402
from ltx_trainer.fognet.pipeline import predict_action, save_checkpoint, train_step  # noqa: E402
from ltx_trainer.fognet.synthetic import synthesize_batch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train FogNet fog-invariant action recognition")
    p.add_argument("-o", "--output-dir", default="fognet_train")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda-clean", type=float, default=0.4)
    p.add_argument("--beta-temp", type=float, default=0.1)
    p.add_argument("--no-fas", action="store_true")
    p.add_argument("--no-me", action="store_true")
    p.add_argument("--no-csa", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    from ltx_trainer.fognet.model import FogNetConfig

    cfg = FogNetConfig(
        num_frames=args.frames,
        use_fas=not args.no_fas,
        use_me=not args.no_me,
        use_csa=not args.no_csa,
    )
    model = FogNet(cfg).to(device)
    loss_fn = FogNetLoss(
        FogNetLossConfig(lambda_clean=args.lambda_clean, beta_temp=args.beta_temp)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        clean, foggy, labels = synthesize_batch(args.batch_size, frames=args.frames, size=args.size)
        clean, foggy, labels = clean.to(device), foggy.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, loss_fn, foggy=foggy, clean=clean, labels=labels)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(
                f"step {step}/{args.steps} total={stats['loss_total']:.4f} "
                f"fog={stats['loss_fog']:.4f} clean={stats['loss_clean']:.4f}"
            )

    clean, foggy, labels = synthesize_batch(8, frames=args.frames, size=args.size)
    foggy, labels = foggy.to(device), labels.to(device)
    logits = torch.stack([model.forward_infer(foggy[i]).cpu() for i in range(foggy.shape[0])])
    t1 = top1_accuracy(logits, labels.cpu())
    t5 = top5_accuracy(logits, labels.cpu())
    save_checkpoint(model, out / "fognet.pt")
    summary = {"top1": t1, "top5": t5, "num_classes": len(FOGACT_CLASSES)}
    (out / "train_log.json").write_text(json.dumps({"log": log[-20:], "eval": summary}, indent=2))
    print(f"Saved {out / 'fognet.pt'} top1={t1:.3f} top5={t5:.3f}")


if __name__ == "__main__":
    main()
