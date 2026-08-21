#!/usr/bin/env python3
"""BiLT-Autoencoder spectral unmixing CLI (Hohmann, arXiv:2605.11829)."""

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

from ltx_trainer.bilt.augment import apply_augmentation, spectral_shift  # noqa: E402
from ltx_trainer.bilt.config import BiLTConfig, PAPER_TITLE, PAPER_URL  # noqa: E402
from ltx_trainer.bilt.model import BiLTAutoencoder  # noqa: E402
from ltx_trainer.bilt.pipeline import count_parameters, curriculum_phase, paper_report, train_step  # noqa: E402
from ltx_trainer.bilt.synthetic import constituent_spectra, dataset_summary, synthetic_batch  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "BiLT-Autoencoder",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "encoder": "BiLT cross-attention scanner",
                "decoder": "physics-constrained linear",
                "latent_dim": 3,
                "spectral_points": ds["points"],
                "dataset": ds["name"],
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


def _cmd_augment(args: argparse.Namespace) -> int:
    x, _, _, _ = synthetic_batch(batch=1, seed=0)
    out = apply_augmentation(
        x,
        max_shift=args.max_shift,
        max_noise=args.max_noise,
        ratio=1.0,
    )
    shifted = spectral_shift(x, args.shift)
    print(
        json.dumps(
            {
                "input_shape": list(x.shape),
                "augmented_mean": float(out.mean()),
                "shifted_mean": float(shifted.mean()),
                "max_shift": args.max_shift,
                "shift_demo": args.shift,
            },
            indent=2,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = BiLTConfig(use_importance_gating=not args.no_gating)
    model = BiLTAutoencoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        x, mu_a, mu_s, _ = synthetic_batch(batch=args.batch, seed=step)
        x, mu_a, mu_s = x.to(device), mu_a.to(device), mu_s.to(device)
        phase = curriculum_phase(min(step + 1, 8000))
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, x, mu_a, mu_s, augment=phase["augment_strength"] > 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats["step"] = float(step)
        stats["phase"] = float(phase["phase"])
        log.append(stats)
        if step % max(1, args.steps // 5) == 0:
            print(
                f"step {step}/{args.steps} loss={stats['loss']:.4f} "
                f"r2_a={stats['r2_mu_a']:.3f} r2_s={stats['r2_mu_s']:.3f}"
            )

    out = {
        "steps": args.steps,
        "params": count_parameters(model),
        "final": log[-1] if log else {},
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = BiLTAutoencoder(BiLTConfig()).to(device)
    model.eval()
    x, mu_a, mu_s, z = synthetic_batch(batch=1, seed=args.seed)
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    result = {
        "latent": out["decode_output"].cpu().tolist(),
        "mu_a_shape": list(out["mu_a"].shape),
        "mu_s_shape": list(out["mu_s"].shape),
        "probe_shape": list(out["probes"].shape),
        "ground_truth_latent": z.tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="BiLT-Autoencoder CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")

    pa = sub.add_parser("augment")
    pa.add_argument("--max-shift", type=int, default=7)
    pa.add_argument("--max-noise", type=float, default=0.03)
    pa.add_argument("--shift", type=int, default=3)

    pt = sub.add_parser("train")
    pt.add_argument("--steps", type=int, default=20)
    pt.add_argument("--batch", type=int, default=8)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--no-gating", action="store_true")

    pi = sub.add_parser("infer")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--device", default="cpu")

    args = p.parse_args()
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "augment": _cmd_augment,
        "train": _cmd_train,
        "infer": _cmd_infer,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
