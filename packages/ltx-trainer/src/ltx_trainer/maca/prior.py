"""GraphSpec structural prior for MACA (arXiv:2605.25746).

This file implements the paper's core idea in a lightweight form:
- Agent relevance Z_prior via cosine similarity + budget temperature + threshold gating (Eq. 4)
- Interaction plausibility P_prior via a small MLP that consumes (e_i, e_j, e_task) (Eq. 6)
- GraphSpec assembly with modulation p_ij * q_j (Eq. 7)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.maca.config import MACAConfig
from ltx_trainer.maca.embeddings import AgentProfile, budget_temperature, cosine, embed_text


def agent_relevance_scores(
    task_text: str,
    agents: list[AgentProfile],
    *,
    budget_tokens: int,
    cfg: MACAConfig | None = None,
) -> dict[str, float]:
    """Compute q_i and apply threshold gamma to yield z_i in [0,1]."""
    cfg = cfg or MACAConfig()
    e_task = embed_text(task_text, dim=cfg.embed_dim, seed=3)
    beta = budget_temperature(budget_tokens, base=cfg.temperature_base)
    out: dict[str, float] = {}
    for a in agents:
        s = cosine(e_task, a.embed(dim=cfg.embed_dim))
        q = float(torch.sigmoid(torch.tensor(s / max(1e-6, beta))).item())
        z = q if q >= cfg.relevance_threshold_gamma else 0.0
        out[a.name] = float(z)
    return out


class InteractionMLP(nn.Module):
    """Tiny MLP for P(v_i -> v_j | x) (Eq. 6)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, e_i: Tensor, e_j: Tensor, e_task: Tensor) -> Tensor:
        x = torch.cat([e_i, e_j, e_task], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


@dataclass(frozen=True)
class GraphSpec:
    """GraphSpec prior GS(x,b) = (Z_prior, P_prior)."""

    agent_names: tuple[str, ...]
    z_prior: Tensor  # [N]
    p_prior: Tensor  # [N,N] directed edge probs in [0,1]

    def mask(self, *, z_threshold: float = 0.0, p_threshold: float = 0.0) -> Tensor:
        """Hard mask H over actions: keep nodes with z>threshold and edges with p>threshold."""
        z_ok = (self.z_prior > z_threshold).to(torch.bool)
        m = (self.p_prior > p_threshold).to(torch.bool)
        # If a node is filtered out, block its selection as next action (column) and outgoing edges (row).
        for j in range(len(self.agent_names)):
            if not bool(z_ok[j]):
                m[:, j] = False
                m[j, :] = False
        return m


def build_graphspec(
    task_text: str,
    agents: list[AgentProfile],
    *,
    budget_tokens: int,
    cfg: MACAConfig | None = None,
    mlp: InteractionMLP | None = None,
) -> GraphSpec:
    cfg = cfg or MACAConfig()
    e_task = embed_text(task_text, dim=cfg.embed_dim, seed=3)
    z_map = agent_relevance_scores(task_text, agents, budget_tokens=budget_tokens, cfg=cfg)
    names = [a.name for a in agents]
    z = torch.tensor([z_map[n] for n in names], dtype=torch.float32)

    mlp = mlp or InteractionMLP(cfg.embed_dim)
    e_agents = torch.stack([a.embed(dim=cfg.embed_dim) for a in agents], dim=0)  # [N,D]
    e_task_b = e_task.expand(e_agents.shape[0], -1)

    N = e_agents.shape[0]
    p = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        e_i = e_agents[i].expand(N, -1)
        p[i] = mlp(e_i, e_agents, e_task_b)

    # Eq. (7) modulation: \tilde p_{ij} = p_{ij} * q_j (we use z_j as gated q_j).
    p = p * z.view(1, N)
    return GraphSpec(agent_names=tuple(names), z_prior=z, p_prior=p.clamp(0.0, 1.0))

