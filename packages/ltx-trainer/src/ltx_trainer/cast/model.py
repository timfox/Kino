"""CAST full model (Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig
from ltx_trainer.cast.encoder import CausalTransformerEncoder
from ltx_trainer.cast.retrieval import CausalRetrieval
from ltx_trainer.cast.simplex import normalize_simplex
from ltx_trainer.cast.transport import TransportHead


class CAST(nn.Module):
    """
    Causal Anchored Simplex Transport.

    p̂_{t+1} = (1 - ρ_t) a_t + ρ_t T_t a_t
    a_t = λ_t p_t + (1 - λ_t) r_t
    """

    def __init__(self, cfg: CASTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CASTConfig()
        self.encoder = CausalTransformerEncoder(self.cfg)
        self.retrieval = CausalRetrieval(self.cfg)
        self.lambda_proj = nn.Linear(self.cfg.hidden_dim, 1)
        self.transport = TransportHead(self.cfg)

    def _persistence_gate(self, hidden: Tensor) -> Tensor:
        raw = torch.sigmoid(self.lambda_proj(hidden).squeeze(-1))
        return self.cfg.lambda_min + (self.cfg.lambda_max - self.cfg.lambda_min) * raw

    def forward_step(self, current: Tensor, history: Tensor) -> dict[str, Tensor]:
        """
        current: (B, D) distribution at time t
        history: (B, T, D) causal past p_{1:t-1} (T may be 0)
        """
        if history.shape[1] == 0:
            seq = current.unsqueeze(1)
        else:
            seq = torch.cat([history, current.unsqueeze(1)], dim=1)

        hidden_seq = self.encoder(seq)
        hidden = hidden_seq[:, -1]

        if seq.shape[1] <= 1:
            retrieved = current
        else:
            hist_hidden = hidden_seq[:, :-1]
            successors = seq[:, 1:]
            retrieved = self.retrieval(hidden, hist_hidden, successors)

        lam = self._persistence_gate(hidden)
        anchor = normalize_simplex(lam.unsqueeze(-1) * current + (1 - lam.unsqueeze(-1)) * retrieved)
        transported, rho, kernel_logits = self.transport(hidden, anchor)

        rho_exp = rho.unsqueeze(-1)
        pred = normalize_simplex((1 - rho_exp) * anchor + rho_exp * transported)

        return {
            "pred": pred,
            "anchor": anchor,
            "retrieved": retrieved,
            "transported": transported,
            "lambda_t": lam,
            "rho_t": rho,
            "kernel_logits": kernel_logits,
            "hidden": hidden,
        }

    def forward(self, seq: Tensor) -> dict[str, Tensor]:
        """Teacher-forced one-step predictions for sequence (B, T, D)."""
        _, t, _ = seq.shape
        preds: list[Tensor] = []
        anchors: list[Tensor] = []
        rhos: list[Tensor] = []
        lams: list[Tensor] = []
        transported: list[Tensor] = []
        kernels: list[Tensor] = []

        for step in range(1, t):
            out = self.forward_step(seq[:, step - 1], seq[:, : step - 1])
            preds.append(out["pred"])
            anchors.append(out["anchor"])
            rhos.append(out["rho_t"])
            lams.append(out["lambda_t"])
            transported.append(out["transported"])
            kernels.append(out["kernel_logits"])

        return {
            "pred_seq": torch.stack(preds, dim=1),
            "target_seq": seq[:, 1:],
            "anchor": torch.stack(anchors, dim=1),
            "transported": torch.stack(transported, dim=1),
            "rho_t": torch.stack(rhos, dim=1),
            "lambda_t": torch.stack(lams, dim=1),
            "kernel_logits": torch.stack(kernels, dim=1),
        }

    def rollout(self, context: Tensor, horizon: int) -> Tensor:
        """Autoregressive rollout from observed context (B, T, D)."""
        seq = context
        for _ in range(horizon):
            out = self.forward_step(seq[:, -1], seq[:, :-1] if seq.shape[1] > 1 else seq[:, :0])
            seq = torch.cat([seq, out["pred"].unsqueeze(1)], dim=1)
        return seq[:, context.shape[1] :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
