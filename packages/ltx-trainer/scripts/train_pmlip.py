#!/usr/bin/env python3
"""Train P-EGNN with fair CRPS (Zaghen et al. arXiv:2605.19939)."""

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

from ltx_trainer.pmlip.egnn import PEGNN  # noqa: E402
from ltx_trainer.pmlip.losses import PMLIPLoss, PMLIPLossConfig  # noqa: E402
from ltx_trainer.pmlip.model import PMLIPConfig, PerturbedMLIP  # noqa: E402
from ltx_trainer.pmlip.pipeline import evaluate_batch, save_checkpoint, train_step  # noqa: E402
from ltx_trainer.pmlip.synthetic import synthesize_nbody  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train P-MLIP (P-EGNN)")
    p.add_argument("-o", "--output-dir", default="pmlip_train")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--k-train", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--noise-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = PMLIPConfig(hidden=args.hidden, layers=args.layers, noise_dim=args.noise_dim, k_train=args.k_train)
    pegnn = PEGNN(cfg.hidden, cfg.layers, cfg.noise_dim).to(device)
    model = PerturbedMLIP(cfg, backend="egnn")
    model.model = pegnn
    loss_fn = PMLIPLoss(PMLIPLossConfig(k_train=args.k_train))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        h, x, edge_index, target = synthesize_nbody(n_particles=8, seed=step)
        h, x, edge_index, target = h.to(device), x.to(device), edge_index.to(device), target.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(pegnn, loss_fn, h=h, x=x, edge_index=edge_index, target=target)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} crps={stats['loss_crps']:.4f} mse={stats['loss_mse_mean']:.4f}")

    h, x, edge_index, target = synthesize_nbody(seed=999)
    h, x, edge_index, target = h.to(device), x.to(device), edge_index.to(device), target.to(device)
    metrics = evaluate_batch(pegnn, h, x, edge_index, target, k=50)
    save_checkpoint(model, out / "pmlip.pt")
    summary = {**metrics, "noise_param_ratio": pegnn.extra_param_ratio()}
    (out / "train_log.json").write_text(json.dumps({"log": log[-20:], "eval": summary}, indent=2))
    print(f"Saved {out / 'pmlip.pt'} CRPS={metrics['crps']:.4f} SSR={metrics['ssr']:.3f}")


if __name__ == "__main__":
    main()
