#!/usr/bin/env python3
# ruff: noqa: T201
"""Train FiLMResidualExposureHead on precomputed hdr_ldr_ev_stack targets (phase 1 LatentHDR)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ltx_trainer.latenthdr import FiLMResidualExposureHead, exposure_latent_mse


class HdrEvStackDataset(Dataset):
    def __init__(self, latents_dir: Path, *, max_samples: int = 0) -> None:
        self.paths = sorted(latents_dir.rglob("*.pt"))
        if max_samples > 0:
            random.shuffle(self.paths)
            self.paths = self.paths[:max_samples]
        self._valid: list[Path] = []
        for p in self.paths:
            try:
                d = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                continue
            if isinstance(d, dict) and "latents" in d and "hdr_ldr_ev_stack" in d and "hdr_ev_list" in d:
                self._valid.append(p)
        if not self._valid:
            raise RuntimeError(
                f"No .pt with latents + hdr_ldr_ev_stack + hdr_ev_list under {latents_dir}. "
                "Re-run process_dataset.py with --hdr-ingest --hdr-synth-bracket-ev."
            )

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = torch.load(self._valid[idx], map_location="cpu", weights_only=True)
        z_base = d["latents"].float()
        stack = d["hdr_ldr_ev_stack"].float()
        evs = d["hdr_ev_list"]
        if not isinstance(evs, list):
            evs = list(evs)
        n = min(stack.shape[0], len(evs))
        j = random.randrange(n) if n > 1 else 0
        z_tgt = stack[j]
        if z_tgt.shape != z_base.shape:
            z_tgt = torch.nn.functional.interpolate(
                z_tgt.unsqueeze(0),
                size=z_base.shape[-3:],
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return {
            "z_base": z_base,
            "z_tgt": z_tgt,
            "ev": torch.tensor(float(evs[j]), dtype=torch.float32),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("latents_dir", type=Path, help="Precomputed latents/ tree with HDR stacks")
    p.add_argument("model_path", type=Path, help="LTX checkpoint (for latent channel count metadata only)")
    p.add_argument("output", type=Path, help="Output .pt for trained exposure head state_dict")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--latent-channels", type=int, default=128)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ds = HdrEvStackDataset(args.latents_dir.expanduser().resolve(), max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    head = FiLMResidualExposureHead(latent_channels=args.latent_channels, cond_dim=128, num_layers=2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for batch in loader:
            z_base = batch["z_base"].to(device)
            z_tgt = batch["z_tgt"].to(device)
            ev = batch["ev"].to(device)
            if z_base.dim() == 4:
                z_base = z_base.unsqueeze(0)
                z_tgt = z_tgt.unsqueeze(0)
            pred = head(z_base, ev)
            loss = exposure_latent_mse(pred, z_tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"epoch {epoch + 1}/{args.epochs}  L_ev={total / max(n, 1):.6f}  samples={len(ds)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "latent_channels": args.latent_channels,
            "model_path": str(args.model_path),
            "latents_dir": str(args.latents_dir),
        },
        args.output,
    )
    print(f"Saved exposure head → {args.output}")


if __name__ == "__main__":
    main()
