#!/usr/bin/env python3
"""One-step HDR from LDR via ExpoCM (Liu et al. arXiv:2605.02464)."""

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
from torchvision.utils import save_image  # noqa: E402
from PIL import Image  # noqa: E402

from ltx_trainer.expo_cm.model import ExpoCM, ExpoCMConfig  # noqa: E402
from ltx_trainer.expo_cm.pipeline import load_expocm_checkpoint  # noqa: E402


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return to_tensor(im)


def main() -> None:
    p = argparse.ArgumentParser(description="ExpoCM one-step HDR reconstruction")
    p.add_argument("image", help="Input LDR RGB")
    p.add_argument("-o", "--output", default="expocm_preview.png")
    p.add_argument("--hdr-linear", default="", help="Save linear HDR tensor (.pt)")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_expocm_checkpoint(args.checkpoint, device=device)
    else:
        model = ExpoCM(ExpoCMConfig()).to(device)

    img = _load_rgb(Path(args.image), args.size).to(device)
    model.eval()
    with torch.no_grad():
        hdr = model.one_step(img)

    hdr_cpu = hdr.cpu()
    if args.hdr_linear:
        torch.save(hdr_cpu, args.hdr_linear)

    peak = hdr_cpu.amax().clamp(min=1e-6)
    preview = (hdr_cpu / peak) / (1.0 + hdr_cpu / peak)
    save_image(preview, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
