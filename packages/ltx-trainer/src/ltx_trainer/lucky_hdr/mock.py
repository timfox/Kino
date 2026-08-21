"""LuckyHDR evaluation smoke with computed forward pass."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lucky_hdr.losses import LuckyHdrLoss
from ltx_trainer.lucky_hdr.model import LuckyHdr, LuckyHdrConfig
from ltx_trainer.lucky_hdr.synthetic import synthesize_bracket_burst
from ltx_trainer.lucky_hdr.tonemap import normalize_exposure


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    cfg = LuckyHdrConfig()
    model = LuckyHdr(cfg)
    model.eval()
    hdr = torch.rand(3, 32, 32) * 0.5 + 0.1
    stack, gt, evs = synthesize_bracket_burst(hdr, num_frames=3, shake_px=0.0, seed=seed)
    no_shift, _, _ = synthesize_bracket_burst(hdr, num_frames=3, shake_px=0.0, seed=seed + 1)
    pred, warp_terms, shifts = model.forward_train(stack, no_shift, evs)
    loss_fn = LuckyHdrLoss()
    loss, stats = loss_fn(pred, gt, warp_terms=warp_terms, shifts=shifts)

    normed = torch.stack([normalize_exposure(stack[i], evs[i], evs[0]) for i in range(stack.shape[0])], dim=0)
    lo = normed.amin(dim=0)
    hi = normed.amax(dim=0)
    with torch.no_grad():
        out, _, _ = model(stack, evs)
    convex_ok = bool((out >= lo - 1e-3).all() and (out <= hi + 1e-3).all())
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "package": "lucky_hdr",
        "paper": cfg.paper_arxiv,
        "torch": True,
        "params": n_params,
        "params_under_100k": n_params < 100_000,
        "loss_total": stats["loss_total"],
        "convex_hull_ok": convex_ok,
        "merge_readiness": "computed",
    }
