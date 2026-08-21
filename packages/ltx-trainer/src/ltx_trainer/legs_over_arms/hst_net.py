"""Human Scene Transformer stub with skeletal MLP fusion (Sec. III-C)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.legs_over_arms.config import LegsOverArmsConfig
from ltx_trainer.legs_over_arms.skeleton import FeatureConfig, total_feature_dim


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HSTStub(nn.Module):
    """Multi-agent trajectory predictor with optional pose stream G(s)."""

    def __init__(
        self,
        cfg: LegsOverArmsConfig | None = None,
        feature_config: FeatureConfig = "K3D_L",
    ) -> None:
        super().__init__()
        self.cfg = cfg or LegsOverArmsConfig(feature_config=feature_config)
        self.feature_config: FeatureConfig = feature_config  # type: ignore[assignment]
        d = self.cfg.embed_dim
        h = self.cfg.hidden_dim
        traj_in = self.cfg.history_steps * 2
        feat_in = total_feature_dim(feature_config)
        self.traj_enc = FeatureMLP(traj_in, h, d)
        self.pose_enc = FeatureMLP(feat_in, h, d) if feat_in else None
        self.agent_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.mode_head = nn.Linear(d, self.cfg.num_modes)
        self.traj_head = nn.Linear(d, self.cfg.future_steps * 2 * self.cfg.num_modes)

    def forward(
        self,
        past_xy: Tensor,
        pose_feat: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b, a, t, _ = past_xy.shape
        traj_flat = past_xy.reshape(b, a, -1)
        h = self.traj_enc(traj_flat)
        if self.pose_enc is not None and pose_feat is not None:
            h = h + self.pose_enc(pose_feat)
        h_attn, _ = self.agent_attn(h, h, h)
        mode_logits = self.mode_head(h_attn)
        traj = self.traj_head(h_attn).view(
            b, a, self.cfg.num_modes, self.cfg.future_steps, 2
        )
        return {"mode_logits": mode_logits, "pred_modes": traj, "agent_hidden": h_attn}
