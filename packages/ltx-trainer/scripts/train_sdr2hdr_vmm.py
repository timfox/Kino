#!/usr/bin/env python3
"""Train Video Merging Model on HDR preprocessed shards (Tedla et al. VMM)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ltx_trainer.hdr_ingest import synthetic_gamma_ldr_stack_from_linear_hdr
from ltx_trainer.sdr2hdr.mevm_proxy import simulate_vae_roundtrip
from ltx_trainer.sdr2hdr.vmm import VideoMergingModel, vmm_log_loss


class _HdrLatentShardDataset(Dataset):
    def __init__(self, latents_dir: Path, *, max_shards: int | None = None) -> None:
        self.paths = sorted(latents_dir.glob("*.pt"))
        if max_shards:
            self.paths = self.paths[:max_shards]
        if not self.paths:
            raise FileNotFoundError(f"No .pt shards in {latents_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        data = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        hdr = data.get("hdr_latent") or data.get("hdr_linear")
        if hdr is None:
            raise KeyError(f"{self.paths[idx]} missing hdr_latent")
        hdr = hdr.float()
        if hdr.ndim == 5:
            hdr = hdr.squeeze(0)
        brackets, evs = synthetic_gamma_ldr_stack_from_linear_hdr(hdr, -4.0, 4.0, 4.0, normalize="p999")
        brackets = simulate_vae_roundtrip(brackets, noise_std=0.005, downscale=4)
        return brackets, hdr, evs


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    return torch.stack([b[0] for b in batch], dim=0), torch.stack([b[1] for b in batch], dim=0), batch[0][2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("latents_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-shards", type=int, default=None)
    args = p.parse_args(argv)

    ds = _HdrLatentShardDataset(args.latents_dir, max_shards=args.max_shards)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    model = VideoMergingModel(num_exposures=3).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total = 0.0
        n = 0
        for brackets_b, hdr_b, evs in loader:
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for i in range(brackets_b.shape[0]):
                pred = model(brackets_b[i].to(args.device), evs)
                loss_acc = loss_acc + vmm_log_loss(pred, hdr_b[i].to(args.device))
            loss = loss_acc / brackets_b.shape[0]
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={total / max(n, 1):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": {"num_exposures": 3, "hidden": 128, "embed": 64, "num_heads": 4}},
        args.output,
    )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
