"""Trainer hooks: load exposure head, sample HDR stacks, compute L_ev."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead, exposure_latent_mse


def load_exposure_head_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> FiLMResidualExposureHead:
    ckpt = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    channels = int(ckpt.get("latent_channels", 128))
    head = FiLMResidualExposureHead(latent_channels=channels)
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.to(device)
    return head


def _batch_hdr_fields(latents_batch: dict[str, Any]) -> tuple[Tensor, Tensor, list[list[float]]] | None:
    if "hdr_ldr_ev_stack" not in latents_batch or "hdr_ev_list" not in latents_batch:
        return None
    stack = latents_batch["hdr_ldr_ev_stack"]
    ev_raw = latents_batch["hdr_ev_list"]
    if not isinstance(stack, Tensor) or stack.ndim not in (5, 6):
        return None
    if isinstance(ev_raw, Tensor):
        ev_lists = ev_raw.tolist()
    elif isinstance(ev_raw, (list, tuple)):
        ev_lists = []
        for row in ev_raw:
            if isinstance(row, Tensor):
                ev_lists.append([float(x) for x in row.flatten().tolist()])
            elif isinstance(row, (list, tuple)):
                ev_lists.append([float(x) for x in row])
            else:
                ev_lists.append([float(row)])
    else:
        return None
    return latents_batch["latents"], stack, ev_lists


def sample_ev_targets_from_batch(
    latents_batch: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Return ``(z_base, z_target, ev)`` per batch for L_ev."""
    parsed = _batch_hdr_fields(latents_batch)
    if parsed is None:
        return None
    z_base, stack, ev_lists = parsed
    if z_base.ndim != 5:
        return None

    b = z_base.shape[0]
    batched_stack = stack.ndim == 6
    z_tgt_parts: list[Tensor] = []
    ev_parts: list[Tensor] = []
    for i in range(b):
        stack_i = stack[i] if batched_stack else stack
        ev_row = ev_lists[i] if i < len(ev_lists) else ev_lists[0]
        n = min(stack_i.shape[0], len(ev_row))
        if n <= 0:
            return None
        j = random.randrange(n) if n > 1 else 0
        tgt = stack_i[j]
        if tgt.shape != z_base[i].shape:
            tgt = torch.nn.functional.interpolate(
                tgt.unsqueeze(0),
                size=z_base[i].shape[-3:],
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        z_tgt_parts.append(tgt)
        ev_parts.append(torch.tensor(float(ev_row[j]), dtype=torch.float32))
    return z_base, torch.stack(z_tgt_parts, dim=0), torch.stack(ev_parts, dim=0)


def compute_latenthdr_ev_loss(
    batch: dict[str, Any],
    head: FiLMResidualExposureHead,
    *,
    device: torch.device,
) -> Tensor | None:
    """L_ev on clean VAE latents vs synthetic γ-LDR stack slice (LatentHDR phase 2)."""
    latents_batch = batch.get("latents")
    if not isinstance(latents_batch, dict):
        return None
    sampled = sample_ev_targets_from_batch(latents_batch)
    if sampled is None:
        return None
    z_base, z_tgt, ev = sampled
    z_base = z_base.to(device)
    z_tgt = z_tgt.to(device)
    ev = ev.to(device)
    pred = head(z_base, ev)
    return exposure_latent_mse(pred, z_tgt)
