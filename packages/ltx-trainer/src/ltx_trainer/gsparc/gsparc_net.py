"""GSpaRC Gaussian field + rendering stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.gsparc.confidence import ConfidenceMLP
from ltx_trainer.gsparc.config import GSpaRCConfig
from ltx_trainer.gsparc.emission import EmissionMLP
from ltx_trainer.gsparc.losses import confidence_raw, normalize_confidence
from ltx_trainer.gsparc.rendering import render_spectrum_stub, spectrum_to_channel


class GSpaRCStub(nn.Module):
    def __init__(self, cfg: GSpaRCConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GSpaRCConfig()
        g = self.cfg.num_gaussians
        self.mu = nn.Parameter(torch.randn(g, 3) * 0.5)
        self.log_scale = nn.Parameter(torch.zeros(g, 3))
        self.logit_opacity = nn.Parameter(torch.full((g,), -2.0))
        self.x_tx = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))
        self.emission = EmissionMLP(self.cfg.emission_hidden)
        self.confidence_mlp = ConfidenceMLP(self.cfg.confidence_hidden)

    def gaussian_opacity(self) -> Tensor:
        return torch.sigmoid(self.logit_opacity)

    def log_tx_distance(self) -> Tensor:
        d = torch.linalg.norm(self.mu - self.x_tx.unsqueeze(0), dim=1).clamp(min=1e-3)
        return torch.log(d)

    def forward(
        self,
        x_rx: Tensor,
        *,
        x_tx: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if x_tx is not None:
            tx = x_tx
        else:
            tx = self.x_tx
        log_d = torch.log(
            torch.linalg.norm(self.mu - tx.unsqueeze(0), dim=1).clamp(min=1e-3)
        )
        emissions = self.emission(x_rx, log_d)
        alphas = self.gaussian_opacity()
        depths = torch.linalg.norm(self.mu - x_rx.unsqueeze(0), dim=1).clamp(min=1e-3)
        tx_dist = torch.linalg.norm(self.mu - tx.unsqueeze(0), dim=1).clamp(min=1e-3)

        z_hat = render_spectrum_stub(
            emissions,
            alphas,
            height=self.cfg.spectrum_height,
            width=self.cfg.spectrum_width,
            use_distance_weight=self.cfg.use_distance_attenuation,
            distance_scale=tx_dist,
            depths=depths,
        )
        h_hat = spectrum_to_channel(z_hat)
        conf_raw = confidence_raw(self.confidence_mlp(x_rx))
        return {
            "spectrum": z_hat,
            "channel": h_hat,
            "confidence_raw": conf_raw,
            "opacity": alphas,
            "num_gaussians": torch.tensor(self.cfg.num_gaussians, device=z_hat.device),
        }

    def training_losses(
        self,
        x_rx: Tensor,
        target_spectrum: Tensor | None = None,
        target_channel: Tensor | None = None,
    ) -> dict[str, Tensor]:
        from ltx_trainer.gsparc.losses import (
            channel_mse_loss,
            confidence_weighted_loss,
            spectrum_loss,
        )

        out = self.forward(x_rx)
        if self.cfg.objective == "channel" and target_channel is not None:
            task = channel_mse_loss(out["channel"], target_channel)
        elif target_spectrum is not None:
            task = spectrum_loss(
                out["spectrum"],
                target_spectrum,
                lambda_l1=self.cfg.lambda_l1_ssim,
            )
        else:
            task = out["spectrum"].abs().mean()

        conf = out["confidence_raw"]
        total = confidence_weighted_loss(
            task, conf, alpha=self.cfg.confidence_alpha
        )
        return {"loss": total, "task_loss": task, "confidence": conf}
