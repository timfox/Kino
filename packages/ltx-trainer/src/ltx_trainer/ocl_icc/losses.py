"""Reconstruction, ECC, ICC objectives — Eq. (2), (4), (8)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.ocl_icc.streams import StreamOutput


def reconstruction_loss(recons: list[Tensor], targets: list[Tensor]) -> Tensor:
    """L_recon — Eq. (2)."""
    return sum(torch.mean((r - t) ** 2) for r, t in zip(recons, targets, strict=True)) / len(recons)


def explicit_cycle_loss(fw_slots: list[Tensor], bw_slots: list[Tensor]) -> Tensor:
    """L_ECC — Eq. (4); latent hard alignment."""
    pairs = zip(fw_slots[:-1], bw_slots[:-1], strict=True)
    return sum(torch.mean((sf - sb) ** 2) for sf, sb in pairs) / max(len(fw_slots) - 1, 1)


def implicit_cycle_loss(bw_recons: list[Tensor], targets: list[Tensor]) -> Tensor:
    """L_ICC — Eq. (8); backward reconstruction on observation manifold."""
    return reconstruction_loss(bw_recons[:-1], targets[:-1])


def hungarian_ecc_loss(fw_slots: list[Tensor], bw_slots: list[Tensor]) -> Tensor:
    """Hungarian-matched ECC ablation — Table 4 (greedy per-slot match stub)."""
    total = fw_slots[0].new_zeros(())
    count = 0
    for sf, sb in zip(fw_slots[:-1], bw_slots[:-1], strict=True):
        sim = torch.einsum("bid,bjd->bij", sf, sb) / (sf.shape[-1] ** 0.5)
        perm = sim.argmax(dim=-1)
        b_idx = torch.arange(sb.shape[0], device=sb.device).unsqueeze(-1)
        matched = sb[b_idx, perm]
        total = total + torch.mean((sf - matched) ** 2)
        count += 1
    return total / max(count, 1)


def total_objective(
    fw: StreamOutput,
    bw: StreamOutput,
    targets: list[Tensor],
    *,
    mode: str = "icc",
    lambda_ecc: float = 1.0,
) -> dict[str, Tensor]:
    """Joint loss for baseline / ecc / icc modes."""
    l_fw = reconstruction_loss(fw.reconstructions, targets)
    out: dict[str, Tensor] = {"l_recon_fw": l_fw}
    if mode == "baseline":
        out["total"] = l_fw
        return out
    if mode == "ecc":
        l_ecc = explicit_cycle_loss(fw.slots, bw.slots)
        out["l_ecc"] = l_ecc
        out["total"] = l_fw + lambda_ecc * l_ecc
        return out
    if mode == "hungarian_ecc":
        l_hecc = hungarian_ecc_loss(fw.slots, bw.slots)
        out["l_hungarian_ecc"] = l_hecc
        out["total"] = l_fw + lambda_ecc * l_hecc
        return out
    if mode == "ncr":
        l_bw = reconstruction_loss(bw.reconstructions, targets)
        out["l_recon_bw"] = l_bw
        out["total"] = l_fw + l_bw
        return out
    # icc (default)
    l_icc = implicit_cycle_loss(bw.reconstructions, targets)
    out["l_icc"] = l_icc
    out["total"] = l_fw + l_icc
    return out
