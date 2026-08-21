#!/usr/bin/env python3
"""Unwrap a modulo-encoded image with ModuloSpikeHdr stub (Zhou et al. arXiv:2604.14632)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

TRAINER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from ltx_trainer.modulo_spike_hdr.model import ModuloSpikeHdrUnwrapper  # noqa: E402


def load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return to_tensor(img)


def main() -> None:
    p = argparse.ArgumentParser(description="Modulo spike HDR unwrap (stub)")
    p.add_argument("modulo_image", type=Path, help="Modulo-encoded RGB (LDR-style PNG)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output μ-law HDR PNG")
    p.add_argument("--ckpt", type=Path, default=None, help="Optional checkpoint")
    args = p.parse_args()

    modulo = load_rgb(args.modulo_image)
    model = ModuloSpikeHdrUnwrapper()
    if args.ckpt and args.ckpt.is_file():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    with torch.no_grad():
        i_mu, _, _ = model(modulo)
    out = (i_mu.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(out).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
