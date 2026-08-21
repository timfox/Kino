#!/usr/bin/env python3
"""Score reference/distorted video or frame pair with FDIM (arXiv:2604.24123)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402
from torchvision.transforms.functional import to_tensor  # noqa: E402
from PIL import Image  # noqa: E402

from ltx_trainer.fdim.model import FDIM, FDIMConfig  # noqa: E402
from ltx_trainer.fdim.pipeline import load_fdim_checkpoint, score_video_pair  # noqa: E402


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return to_tensor(im)


def main() -> None:
    p = argparse.ArgumentParser(description="FDIM full-reference VQA score")
    p.add_argument("reference", help="Reference image or frame")
    p.add_argument("distorted", help="Distorted image or frame")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--hdr", action="store_true", help="Apply PU21 preprocessing (FDIM+PU21)")
    p.add_argument("--deep-only", action="store_true")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_fdim_checkpoint(args.checkpoint, device=device)
    else:
        model = FDIM(FDIMConfig(use_pu21=args.hdr)).to(device)

    ref = _load_rgb(Path(args.reference), args.size).to(device)
    dist = _load_rgb(Path(args.distorted), args.size).to(device)
    model.eval()
    with torch.no_grad():
        if args.deep_only:
            q = model.deep_only(ref, dist)
        else:
            q = model(ref, dist)
    print(f"FDIM score: {float(q.reshape(-1)[0].cpu()):.4f}")


if __name__ == "__main__":
    main()
