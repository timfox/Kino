#!/usr/bin/env python3
"""Evaluate CineMatte on CineMatte-4K or folder pairs; optional background-shift stress test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: E402, F401

import torch  # noqa: E402

from ltx_trainer.cinematte.dataset import CineMatte4KImageDataset, write_manifest_template  # noqa: E402
from ltx_trainer.cinematte.metrics import (  # noqa: E402
    connectivity_error,
    dtssd,
    gradient_error,
    matting_mad,
    matting_mse,
)
from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig  # noqa: E402
from ltx_trainer.cinematte.pipeline import apply_background_shift, load_cinematte_checkpoint  # noqa: E402


def _eval_pairs(model, pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], device: str) -> dict[str, float]:
    mads, mses, grads, conns = [], [], [], []
    preds_seq: list[torch.Tensor] = []
    gts_seq: list[torch.Tensor] = []
    for img, bg, gt in pairs:
        img = img.to(device).unsqueeze(0)
        bg = bg.to(device).unsqueeze(0)
        gt = gt.to(device)
        with torch.no_grad():
            pred = model(img, bg).squeeze(0)
        mads.append(matting_mad(pred, gt))
        mses.append(matting_mse(pred, gt))
        grads.append(gradient_error(pred, gt))
        conns.append(connectivity_error(pred, gt))
        preds_seq.append(pred.squeeze(0).cpu())
        gts_seq.append(gt.squeeze(0).cpu())
    out = {
        "MAD": sum(mads) / len(mads),
        "MSE": sum(mses) / len(mses),
        "Grad": sum(grads) / len(grads),
        "Conn": sum(conns) / len(conns),
        "n": float(len(pairs)),
    }
    if len(preds_seq) >= 2:
        out["dtSSD"] = dtssd(torch.stack(preds_seq), torch.stack(gts_seq))
    return out


def _shift_stress(
    model,
    pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    *,
    angle: float,
    scale: float,
    shear: float,
) -> dict[str, float]:
    shifted_pairs = []
    for img, bg, gt in pairs:
        shifted_pairs.append((img, apply_background_shift(bg, angle_deg=angle, scale=scale, shear=shear), gt))
    return _eval_pairs(model, shifted_pairs, device)


def main() -> None:
    p = argparse.ArgumentParser(description="CineMatte evaluation")
    p.add_argument("--dataset", default="", help="CineMatte-4K root with cinematte4k_manifest.json")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--backbone", default="stub")
    p.add_argument("--split", default="test")
    p.add_argument("--long-edge", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--write-manifest-template", metavar="DIR", default="")
    p.add_argument("--shift-stress", action="store_true", help="Table 3 style background misalignment")
    p.add_argument("-o", "--output", default="cinematte_eval.json")
    args = p.parse_args()

    if args.write_manifest_template:
        path = write_manifest_template(Path(args.write_manifest_template) / "cinematte4k_manifest.json")
        print(f"Wrote template {path}")
        return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_cinematte_checkpoint(args.checkpoint, device=device)
    else:
        model = CineMatte(CineMatteConfig(backbone=args.backbone)).to(device)
    model.eval()

    if not args.dataset:
        print("Provide --dataset or --write-manifest-template", file=sys.stderr)
        sys.exit(1)

    ds = CineMatte4KImageDataset(args.dataset, split=args.split, long_edge=args.long_edge)
    pairs = [ds[i] for i in range(len(ds))]
    metrics = _eval_pairs(model, pairs, device)
    report: dict[str, object] = {"split": args.split, "metrics": metrics}

    if args.shift_stress:
        report["shift_stress"] = {
            "mild": _shift_stress(model, pairs, device, angle=2.0, scale=1.05, shear=0.02),
            "strong": _shift_stress(model, pairs, device, angle=5.0, scale=1.10, shear=0.07),
        }

    out = Path(args.output).expanduser().resolve()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
