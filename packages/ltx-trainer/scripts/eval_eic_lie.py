#!/usr/bin/env python3
"""Evaluate EIC-LIE on RLE manifest (PSNR / SSIM)."""

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

from ltx_trainer.eic_lie.dataset import RLEDataset, write_rle_manifest_template  # noqa: E402
from ltx_trainer.eic_lie.metrics import psnr, ssim  # noqa: E402
from ltx_trainer.eic_lie.model import EicLie, EicLieConfig  # noqa: E402
from ltx_trainer.eic_lie.pipeline import load_eic_lie_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EIC-LIE RLE evaluation")
    p.add_argument("--dataset", required=False, help="RLE root with rle_manifest.json")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--write-manifest-template", metavar="DIR", default="")
    p.add_argument("-o", "--output", default="eic_lie_eval.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.write_manifest_template:
        path = write_rle_manifest_template(Path(args.write_manifest_template) / "rle_manifest.json")
        print(f"Wrote {path}")
        return

    if not args.dataset:
        print("Provide --dataset or --write-manifest-template", file=sys.stderr)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_eic_lie_checkpoint(args.checkpoint, device=device)
    else:
        model = EicLie(EicLieConfig()).to(device)
    model.eval()

    ds = RLEDataset(args.dataset, split=args.split)
    psnrs, ssims = [], []
    for i in range(len(ds)):
        low, gt, voxel = ds[i]
        with torch.no_grad():
            pred = model(low.unsqueeze(0).to(device), voxel.unsqueeze(0).to(device)).squeeze(0).cpu()
        psnrs.append(psnr(pred.unsqueeze(0), gt.unsqueeze(0)))
        ssims.append(ssim(pred.unsqueeze(0), gt.unsqueeze(0)))

    report = {
        "split": args.split,
        "n": len(ds),
        "PSNR": sum(psnrs) / len(psnrs),
        "SSIM": sum(ssims) / len(ssims),
    }
    out = Path(args.output).expanduser().resolve()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
