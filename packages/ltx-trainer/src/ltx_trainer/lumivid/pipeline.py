"""Inference, checkpoints, EXR export stub."""

from __future__ import annotations

import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ltx_trainer.lumivid.degradations import apply_sdr_degradations
from ltx_trainer.lumivid.model import LumiVid, LumiVidConfig


def sdr_video_to_hdr(model: LumiVid, sdr_cfhw: Tensor, *, seed: int | None = 42) -> Tensor:
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=sdr_cfhw.device).manual_seed(seed)
    with torch.no_grad():
        c, f, h, w = sdr_cfhw.shape
        outs: list[Tensor] = []
        for i in range(f):
            sdr_deg = apply_sdr_degradations(sdr_cfhw[:, i], model.degrade_cfg)
            z_ref = model.encode_vae(sdr_deg)
            z_hdr = model.sample_hdr_latent(z_ref, generator=gen)
            outs.append(model.latent_to_scene(z_hdr, out_hw=(h, w)))
        return torch.stack(outs, dim=1)


def load_lumivid_checkpoint(path: Path | str, *, device: str = "cpu") -> LumiVid:
    ckpt = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = LumiVidConfig(**cfg_dict)
    model = LumiVid(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model.to(device).eval()


def save_exr_half(path: Path | str, hdr_chw: Tensor) -> None:
    """Minimal float16 RGB EXR writer for scene-linear output (Sec. 3.2)."""
    path = Path(path)
    arr = hdr_chw.detach().cpu().float().permute(1, 2, 0).numpy().astype(np.float16)
    h, w, _ = arr.shape
    header = b"".join(
        [
            b"version\n2\n",
            b"channels\nchlist\n",
            b"  B half 0 1 1\n  G half 0 1 1\n  R half 0 1 1\n",
            b"compression\nno compression\n",
            f"dataWindow\n0 0 {w - 1} {h - 1}\n".encode(),
            f"displayWindow\n0 0 {w - 1} {h - 1}\n".encode(),
            b"lineOrder\nincreasingY\n",
            b"pixelAspectRatio\n1\n",
            b"endHeader\n",
        ]
    )
    # Scanline EXR: 4-byte y + 4-byte data size + interleaved RGB half
    with path.open("wb") as f:
        f.write(header)
        for y in range(h):
            row = arr[y]
            interleaved = row.tobytes()
            f.write(struct.pack("<i", y))
            f.write(struct.pack("<i", len(interleaved)))
            f.write(interleaved)


def save_checkpoint(model: LumiVid, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)
