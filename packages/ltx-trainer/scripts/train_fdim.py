#!/usr/bin/env python3
"""Train FDIM deep branch with ranking loss on synthetic distortions."""

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

from ltx_trainer.fdim.model import FDIM, FDIMConfig  # noqa: E402
from ltx_trainer.fdim.ranking_loss import RankingLoss  # noqa: E402
from ltx_trainer.fdim.synthetic import sample_training_pair, synthesize_distortion  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train FDIM deep branch (ranking)")
    p.add_argument("-o", "--output-dir", default="fdim_train")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = FDIMConfig()
    model = FDIM(cfg).to(device)
    loss_fn = RankingLoss()
    opt = torch.optim.AdamW(model.deep.parameters(), lr=args.lr, weight_decay=5e-4)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        ref, _, _, mos_i, sig_i = sample_training_pair(args.size, args.size)
        ref2, _, _, mos_j, sig_j = sample_training_pair(args.size, args.size)
        sev_i = max(0.05, min(0.95, 1.0 - (mos_i - 1.0) / 5.0))
        sev_j = max(0.05, min(0.95, 1.0 - (mos_j - 1.0) / 5.0))
        dist_i = synthesize_distortion(ref, sev_i).to(device)
        dist_j = synthesize_distortion(ref2, sev_j).to(device)
        ref, ref2 = ref.to(device), ref2.to(device)
        qi, si = model.deep.forward_frame(ref, dist_i)
        qj, sj = model.deep.forward_frame(ref2, dist_j)
        mu_i = torch.tensor(mos_i, device=device)
        mu_j = torch.tensor(mos_j, device=device)
        sig_i_t = torch.tensor(sig_i, device=device)
        sig_j_t = torch.tensor(sig_j, device=device)
        loss = loss_fn(qi.reshape(1), qj.reshape(1), si.reshape(1), sj.reshape(1), mu_i, mu_j, sig_i_t, sig_j_t)
        loss.backward()
        opt.step()
        stats = {"loss_rank": float(loss.detach()), "step": float(step)}
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} rank={stats['loss_rank']:.6f}")

    torch.save({"config": {"use_pu21": cfg.use_pu21, "l_peak": cfg.l_peak}, "state_dict": model.state_dict()}, out / "fdim_checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")
    print(f"Saved {out / 'fdim_checkpoint.pt'}")


if __name__ == "__main__":
    main()
