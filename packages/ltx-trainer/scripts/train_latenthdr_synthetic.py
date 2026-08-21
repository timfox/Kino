#!/usr/bin/env python3
"""Train LatentHDR exposure head + stub VAE on synthetic HDR (L_ev, arXiv:2605.11115)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.latenthdr import exposure_latent_mse  # noqa: E402
from ltx_trainer.latenthdr.model import LatentHdr, LatentHdrConfig  # noqa: E402
from ltx_trainer.latenthdr.synthetic import synthesize_hdr_scene  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train LatentHDR (synthetic L_ev)")
    p.add_argument("-o", "--output-dir", default="latenthdr_train", dest="output_dir")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head", choices=("unet", "film_mlp"), default="unet")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = LatentHdrConfig(head_type=args.head, latent_channels=16)
    model = LatentHdr(cfg).to(device)
    opt = torch.optim.Adam(
        list(model.exposure_head.parameters()) + list(model.codec.parameters()),
        lr=args.lr,
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        hdr = synthesize_hdr_scene(args.size, args.size).to(device)
        z_base, z_stack, evs = model.training_targets_from_hdr(hdr)
        j = random.randrange(len(evs))
        ev = torch.tensor(float(evs[j]), device=device)
        z_tgt = z_stack[j]
        if z_tgt.dim() == 4 and z_tgt.shape[0] == cfg.latent_channels:
            z_in = z_base.unsqueeze(0) if z_base.dim() == 3 else z_base
            pred = model.exposure_head(z_in, ev)
            if pred.dim() == 4 and z_tgt.dim() == 3:
                pred = pred.squeeze(0)
        else:
            pred = model.predict_exposure_latent(z_base, ev)
        loss = exposure_latent_mse(pred, z_tgt if z_tgt.dim() == pred.dim() else z_tgt.unsqueeze(0).squeeze(0))
        loss.backward()
        opt.step()
        stats = {"loss_ev": float(loss.detach()), "step": float(step)}
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} L_ev={stats['loss_ev']:.6f}")

    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, out / "latenthdr_checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")
    print(f"Saved {out / 'latenthdr_checkpoint.pt'}")


if __name__ == "__main__":
    main()
