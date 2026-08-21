"""DeblurNVS training and inference smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.deblur_nvs.benchmarks import table2_ours
from ltx_trainer.deblur_nvs.config import DeblurNVSConfig
from ltx_trainer.deblur_nvs.latent_models import DeblurNVSStub
from ltx_trainer.deblur_nvs.synthetic import synthetic_views


def context_latent_loss(
    model: DeblurNVSStub,
    blur_ctx: Tensor,
    sharp_ctx: Tensor,
) -> Tensor:
    """Eq. 9 — context restoration with zero-padded camera (c=0)."""
    b, k, _, h, w = blur_ctx.shape
    blur_flat = blur_ctx.reshape(b * k, 3, h, w)
    sharp_flat = sharp_ctx.reshape(b * k, 3, h, w)
    z_blur = model.encoder(blur_flat)
    with torch.no_grad():
        z_sharp = model.encoder(sharp_flat)
    t = torch.randn_like(z_sharp)
    z_noisy = 0.5 * z_sharp + 0.5 * t
    ctx = z_blur.mean(dim=0, keepdim=True).expand_as(z_noisy)
    pred = model.ctx_diffusion(z_noisy, ctx)
    return F.mse_loss(pred, t)


def target_latent_loss(
    model: DeblurNVSStub,
    restored_ctx: Tensor,
    target_sharp: Tensor,
    camera: Tensor,
) -> Tensor:
    """Eq. 13 — target latent synthesis with camera conditioning."""
    b, _, h, w = target_sharp.shape
    z_tgt = model.encoder(target_sharp)
    t = torch.randn_like(z_tgt)
    z_noisy = 0.5 * z_tgt + 0.5 * t
    ctx = restored_ctx.mean(dim=0, keepdim=True)
    pred = model.tgt_diffusion(z_noisy, ctx, camera=camera)
    return F.mse_loss(pred, t)


def rgb_loss(model: DeblurNVSStub, pred_rgb: Tensor, target_rgb: Tensor, cfg: DeblurNVSConfig) -> Tensor:
    """Eq. 14 — L1 + LPIPS proxy + GAN proxy."""
    l1 = F.l1_loss(pred_rgb, target_rgb)
    lpips_proxy = F.mse_loss(
        F.avg_pool2d(pred_rgb, 4),
        F.avg_pool2d(target_rgb, 4),
    )
    gan_proxy = F.relu(1.0 - pred_rgb.mean() + target_rgb.mean())
    return cfg.lambda_l1 * l1 + cfg.lambda_lpips * lpips_proxy + cfg.lambda_gan * gan_proxy


def train_step(
    model: DeblurNVSStub,
    batch: dict[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train()
    cfg = model.cfg
    blur = batch["blur_context"]
    sharp = batch["sharp_context"]
    b, k, _, h, w = blur.shape
    blur_flat = blur.reshape(b * k, 3, h, w)
    z_blur = model.encoder(blur_flat).reshape(b, k, cfg.latent_channels, h, w)

    l_ctx = context_latent_loss(model, blur, sharp)
    restored = model.restore_context(z_blur.reshape(b * k, cfg.latent_channels, h, w))
    restored_bk = restored.reshape(b, k, cfg.latent_channels, h, w)
    l_tgt = target_latent_loss(model, restored, batch["target_sharp"], batch["camera"])
    z_syn = model.synthesize_target(restored_bk[:, 0], camera=batch["camera"])
    pred = model.decode_rgb(z_syn)
    l_rgb = rgb_loss(model, pred, batch["target_sharp"], cfg)
    loss = l_ctx + l_tgt + l_rgb
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {
        "loss": float(loss.item()),
        "l_ctx": float(l_ctx.item()),
        "l_tgt": float(l_tgt.item()),
        "l_rgb": float(l_rgb.item()),
    }


def infer_novel_view(
    model: DeblurNVSStub,
    blur_context: Tensor,
    camera: Tensor,
) -> dict[str, Tensor]:
    """Single-run inference: blur context → sharp novel view (Fig. 1)."""
    model.eval()
    cfg = model.cfg
    b, k, _, h, w = blur_context.shape
    with torch.no_grad():
        z_blur = model.encoder(blur_context.reshape(b * k, 3, h, w))
        restored = model.restore_context(z_blur)
        restored_ctx = restored.reshape(b, k, cfg.latent_channels, h, w)
        z_tgt = model.synthesize_target(restored_ctx[:, 0], camera=camera)
        rgb = model.decode_rgb(z_tgt)
    return {"restored_latents": restored_ctx, "target_latent": z_tgt, "novel_view": rgb}


def evaluation_demo_run(cfg: DeblurNVSConfig | None = None, *, device: str = "cpu", seed: int = 0) -> dict[str, Any]:
    cfg = cfg or DeblurNVSConfig(height=56, width=96, context_views=3)
    dev = torch.device(device)
    batch = synthetic_views(cfg, batch_size=1, device=dev, seed=seed)
    model = DeblurNVSStub(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = train_step(model, batch, optimizer=opt)
    out = infer_novel_view(model, batch["blur_context"], batch["camera"])
    ref = table2_ours()
    return {
        "device": str(dev),
        "train": metrics,
        "ref_lpips": ref["lpips"],
        "ref_fid": ref["fid"],
        "novel_view_shape": list(out["novel_view"].shape),
        "context_views": cfg.context_views,
    }
