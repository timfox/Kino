"""LatentHDR end-to-end model (l2h + t2h stub)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.hdr_ingest import ev_list_arange, merge_log_domain_radiance, synthetic_gamma_ldr_stack_from_linear_hdr
from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead
from ltx_trainer.latenthdr.film_unet import FiLMExposureUNet
from ltx_trainer.latenthdr.vae_stub import StubVaeLatentCodec


@dataclass
class LatentHdrConfig:
    latent_channels: int = 16
    ev_min: float = -7.0
    ev_max: float = 5.0
    ev_step: float = 1.0
    gamma: float = 2.2
    head_type: str = "unet"  # "unet" | "film_mlp"
    use_stub_vae: bool = True


class LatentHdr(nn.Module):
    """Decoupled scene latent + deterministic exposure mapping (arXiv:2605.11115)."""

    def __init__(self, cfg: LatentHdrConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LatentHdrConfig()
        c = self.cfg.latent_channels
        if self.cfg.head_type == "film_mlp":
            self.exposure_head: nn.Module = FiLMResidualExposureHead(latent_channels=c)
        else:
            self.exposure_head = FiLMExposureUNet(latent_channels=c)
        self.codec = StubVaeLatentCodec(latent_channels=c) if self.cfg.use_stub_vae else None
        self._ev_list = ev_list_arange(self.cfg.ev_min, self.cfg.ev_max, self.cfg.ev_step)

    @property
    def ev_list(self) -> list[float]:
        return list(self._ev_list)

    def encode_ldr(self, ldr: Tensor) -> Tensor:
        """Posterior mean path: γ-LDR → latent anchor (EV=0)."""
        if self.codec is None:
            raise RuntimeError("No VAE codec; pass precomputed z_base to forward.")
        if ldr.dim() == 3:
            return self.codec.encode(ldr)
        if ldr.dim() == 4 and ldr.shape[0] == 3:
            return self.codec.encode_video(ldr)
        return self.codec.encode(ldr)

    def predict_exposure_latent(self, z_base: Tensor, ev: float | Tensor) -> Tensor:
        ev_t = ev if isinstance(ev, Tensor) else torch.tensor(float(ev), device=z_base.device)
        from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead

        if z_base.dim() == 3:
            z_in: Tensor = z_base.unsqueeze(0)
            if isinstance(self.exposure_head, FiLMResidualExposureHead):
                z_in = z_in.unsqueeze(2)
            out = self.exposure_head(z_in, ev_t)
            if out.dim() == 5:
                return out.squeeze(0).squeeze(1)
            return out.squeeze(0)
        if z_base.dim() == 4 and z_base.shape[0] == self.cfg.latent_channels:
            z_in = z_base.unsqueeze(0)
            if isinstance(self.exposure_head, FiLMResidualExposureHead):
                z_in = z_in.unsqueeze(2)
            out = self.exposure_head(z_in, ev_t)
            if out.dim() == 5:
                return out.squeeze(0).squeeze(1)
            return out.squeeze(0)
        return self.exposure_head(z_base, ev_t)

    def decode_latent(self, z: Tensor, *, out_hw: tuple[int, int] | None = None) -> Tensor:
        if self.codec is None:
            raise RuntimeError("No VAE codec configured.")
        if z.dim() == 3:
            return self.codec.decode(z, out_hw=out_hw)
        if z.dim() == 4 and z.shape[0] == self.cfg.latent_channels:
            oh, ow = out_hw or (z.shape[-2] * self.codec.scale, z.shape[-1] * self.codec.scale)
            return self.codec.decode_video(z, out_h=oh, out_w=ow)
        return self.codec.decode(z, out_hw=out_hw).squeeze(0)

    def forward(
        self,
        ldr_or_z: Tensor,
        *,
        z_base: Tensor | None = None,
        ev_list: list[float] | None = None,
        return_stack: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """l2h: LDR ``[3,H,W]`` → linear HDR ``[3,H,W]``. With ``return_stack``, also returns γ-LDR stack."""
        evs = ev_list or self.ev_list
        if z_base is None:
            z_anchor = self.encode_ldr(ldr_or_z)
            out_hw = (ldr_or_z.shape[-2], ldr_or_z.shape[-1]) if ldr_or_z.dim() >= 3 else None
        else:
            z_anchor = z_base
            out_hw = None
            if ldr_or_z.dim() >= 3 and ldr_or_z.shape[-2:] != (1, 1):
                out_hw = (ldr_or_z.shape[-2], ldr_or_z.shape[-1])

        is_video = z_anchor.dim() == 4 and z_anchor.shape[0] == self.cfg.latent_channels
        decoded: list[Tensor] = []
        for e in evs:
            z_e = self.predict_exposure_latent(z_anchor, e)
            decoded.append(self.decode_latent(z_e, out_hw=out_hw))

        if is_video:
            stack = torch.stack(decoded, dim=0)
            hdr = merge_log_domain_radiance(stack, evs, gamma=self.cfg.gamma)
            return (hdr, stack) if return_stack else hdr

        stack = torch.stack(decoded, dim=0).unsqueeze(2)
        hdr = merge_log_domain_radiance(stack, evs, gamma=self.cfg.gamma).squeeze(1)
        return (hdr, stack.squeeze(2)) if return_stack else hdr

    def training_targets_from_hdr(self, hdr_linear: Tensor) -> tuple[Tensor, Tensor, list[float]]:
        """Build synthetic γ-stack latents for L_ev (trainer / synthetic loop)."""
        if hdr_linear.dim() == 3:
            hdr_linear = hdr_linear.unsqueeze(1)
        stack, evs = synthetic_gamma_ldr_stack_from_linear_hdr(
            hdr_linear, self.cfg.ev_min, self.cfg.ev_max, self.cfg.ev_step, gamma=self.cfg.gamma
        )
        z_parts = []
        for i in range(stack.shape[0]):
            frame = stack[i]
            if frame.shape[1] == 1:
                z_parts.append(self.codec.encode(frame[:, 0]))
            else:
                z_parts.append(self.codec.encode_video(frame))
        z_stack = torch.stack([z.squeeze(0) if z.dim() == 4 and z.shape[0] == 1 else z for z in z_parts], dim=0)
        base_idx = evs.index(0.0) if 0.0 in evs else len(evs) // 2
        z_base = z_parts[base_idx]
        if z_base.dim() == 4 and z_base.shape[0] == 1:
            z_base = z_base.squeeze(0)
        return z_base.detach(), z_stack.detach(), evs
