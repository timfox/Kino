"""Token-aware orchestration demo for MACA (arXiv:2605.25746).

This is a *reference implementation* for the repo's paper stubs:
- action space is a pool of agents plus STOP
- GraphSpec induces a hard mask (Eq. 11)
- a soft KL anchor toward π_mix = 0.5(π_ref + π_prior) (Eq. 12)

We implement a tiny policy network and a GRPO-like group objective (Eq. 9–10) for a single update step
on synthetic tasks. The goal is to validate the logic and provide reproducible diagnostics, not to
replicate full benchmark training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.maca.config import MACAConfig
from ltx_trainer.maca.embeddings import AgentProfile, embed_text
from ltx_trainer.maca.prior import GraphSpec, build_graphspec


@dataclass(frozen=True)
class Trajectory:
    actions: tuple[int, ...]  # indices into action space (agents + STOP)
    token_cost: int
    solved: bool


class OrchestrationPolicy(nn.Module):
    """π_θ(a_t | s_t) over agents + STOP."""

    def __init__(self, dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, num_actions),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    big_neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    masked = torch.where(mask, logits, big_neg)
    return torch.softmax(masked, dim=-1)


def _kl(p: Tensor, q: Tensor) -> Tensor:
    """KL(p||q) for categorical distributions, safe for zeros."""
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def prior_action_distribution(gs: GraphSpec) -> Tensor:
    """π_prior over next-agent action (exclude STOP) from GraphSpec edge mass."""
    p = gs.p_prior
    col_mass = p.sum(dim=0)  # incoming plausibility
    if float(col_mass.sum().item()) <= 1e-12:
        col_mass = torch.ones_like(col_mass)
    return (col_mass / col_mass.sum()).clamp(0.0, 1.0)


def sample_trajectory(
    task_text: str,
    agents: list[AgentProfile],
    *,
    budget_tokens: int,
    cfg: MACAConfig | None = None,
    policy: OrchestrationPolicy | None = None,
    with_graphspec: bool = True,
    seed: int = 0,
) -> tuple[Trajectory, dict[str, float]]:
    """Sample a single trajectory; return diagnostics with KL and entropy."""
    cfg = cfg or MACAConfig()
    torch.manual_seed(seed)
    stop_idx = len(agents)
    policy = policy or OrchestrationPolicy(cfg.embed_dim, stop_idx + 1)

    gs = build_graphspec(task_text, agents, budget_tokens=budget_tokens, cfg=cfg) if with_graphspec else None

    # Hard mask: allow all actions if no GraphSpec; else allow STOP and agent nodes with z>0.
    agent_ok = torch.ones(len(agents), dtype=torch.bool)
    if gs is not None:
        agent_ok = (gs.z_prior > 0.0).to(torch.bool)
        if not bool(agent_ok.any()):
            agent_ok[:] = True
    mask = torch.cat([agent_ok, torch.tensor([True])], dim=0)  # STOP always allowed

    # State is a deterministic embedding of the task text.
    state = embed_text(task_text, dim=cfg.embed_dim, seed=11)
    logits = policy(state)
    pi_theta = _masked_softmax(logits, mask)

    # Reference distribution: uniform over allowed actions.
    pi_ref = mask.to(torch.float32)
    pi_ref = pi_ref / pi_ref.sum()

    # Prior distribution: derived from GraphSpec (agents only) + small STOP mass.
    if gs is not None:
        pi_p = prior_action_distribution(gs)
        pi_prior = torch.cat([pi_p, torch.tensor([0.05])], dim=0)
        pi_prior = pi_prior / pi_prior.sum()
    else:
        pi_prior = pi_ref

    pi_mix = 0.5 * (pi_ref + pi_prior)
    kl_mix = float(_kl(pi_theta, pi_mix).item())
    ent = float((-torch.sum(pi_theta * torch.log(pi_theta.clamp_min(1e-12)))).item())

    # Sample actions until STOP or max steps; approximate "solved" by invoking a minimal set:
    # - for code tasks: needs CodeWriting + (UnitTestWriter or CodeReviewer)
    # - for math: needs MathSolver + ArithmeticChecker
    # - else: needs AnalyzeAgent + QASynthesizer (not in default pool; treat TaskPlanner+Summarizer as proxy)
    acts: list[int] = []
    tokens = 0
    name_by_idx = {i: a.name for i, a in enumerate(agents)}
    required: set[str] = set()
    t = task_text.lower()
    if "code" in t or "function" in t or "python" in t:
        required = {"CodeWriting", "UnitTestWriter"}
    elif "math" in t or "solve" in t or "equation" in t:
        required = {"MathSolver", "ArithmeticChecker"}
    else:
        required = {"TaskPlanner", "Summarizer"}

    seen: set[str] = set()
    for _ in range(cfg.max_steps):
        a_idx = int(torch.multinomial(pi_theta, 1).item())
        acts.append(a_idx)
        if a_idx == stop_idx:
            break
        nm = name_by_idx.get(a_idx, "")
        seen.add(nm)
        tokens += int(agents[a_idx].expected_cost_tokens)
        if tokens >= budget_tokens:
            acts.append(stop_idx)
            break

    solved = required.issubset(seen)
    traj = Trajectory(actions=tuple(acts), token_cost=tokens, solved=bool(solved))

    # r' = acc - beta * cost - alpha * KL(pi_theta||pi_mix)
    acc = 1.0 if solved else 0.0
    reward = float(acc - cfg.token_cost_beta * (tokens / 1000.0) - cfg.kl_alpha * kl_mix)
    return traj, {"reward": reward, "kl_mix": kl_mix, "entropy": ent, "tokens": float(tokens), "solved": float(acc)}


def grpo_update_demo(
    task_text: str,
    agents: list[AgentProfile],
    *,
    budget_tokens: int,
    cfg: MACAConfig | None = None,
    with_graphspec: bool = True,
) -> dict[str, float]:
    """One GRPO-like policy update step on a single synthetic task."""
    cfg = cfg or MACAConfig()
    stop_idx = len(agents)
    policy = OrchestrationPolicy(cfg.embed_dim, stop_idx + 1)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)

    # Group sample
    trajs: list[Trajectory] = []
    rewards: list[float] = []
    kls: list[float] = []
    ents: list[float] = []
    solved: list[float] = []
    tokens: list[float] = []
    for k in range(cfg.group_size):
        tr, diag = sample_trajectory(
            task_text,
            agents,
            budget_tokens=budget_tokens,
            cfg=cfg,
            policy=policy,
            with_graphspec=with_graphspec,
            seed=100 + k,
        )
        trajs.append(tr)
        rewards.append(diag["reward"])
        kls.append(diag["kl_mix"])
        ents.append(diag["entropy"])
        solved.append(diag["solved"])
        tokens.append(diag["tokens"])

    r = torch.tensor(rewards, dtype=torch.float32)
    adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

    # Compute a simple policy gradient surrogate on the first action only (demo).
    state = embed_text(task_text, dim=cfg.embed_dim, seed=11)
    logits = policy(state)
    logp = torch.log_softmax(logits, dim=-1)

    first_actions = torch.tensor([t.actions[0] for t in trajs], dtype=torch.long)
    logp_a = logp[first_actions]
    loss = -(adv * logp_a).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    return {
        "loss": float(loss.item()),
        "reward_mean": float(r.mean().item()),
        "reward_std": float(r.std(unbiased=False).item()),
        "solved_rate": float(torch.tensor(solved).mean().item()),
        "tokens_mean": float(torch.tensor(tokens).mean().item()),
        "kl_mean": float(torch.tensor(kls).mean().item()),
        "entropy_mean": float(torch.tensor(ents).mean().item()),
    }


def compare_with_without_graphspec(
    task_text: str,
    agents: list[AgentProfile],
    *,
    budget_tokens: int,
    cfg: MACAConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Small diagnostic: GraphSpec should reduce entropy and KL variance on the same task."""
    cfg = cfg or MACAConfig()
    out: dict[str, dict[str, float]] = {}
    for flag, key in ((True, "with_graphspec"), (False, "without_graphspec")):
        diags: list[dict[str, float]] = []
        for k in range(12):
            _, d = sample_trajectory(
                task_text,
                agents,
                budget_tokens=budget_tokens,
                cfg=cfg,
                with_graphspec=flag,
                seed=7 + k,
            )
            diags.append(d)
        out[key] = {
            "reward_mean": float(torch.tensor([d["reward"] for d in diags]).mean().item()),
            "entropy_mean": float(torch.tensor([d["entropy"] for d in diags]).mean().item()),
            "kl_mean": float(torch.tensor([d["kl_mix"] for d in diags]).mean().item()),
            "tokens_mean": float(torch.tensor([d["tokens"] for d in diags]).mean().item()),
            "solved_rate": float(torch.tensor([d["solved"] for d in diags]).mean().item()),
        }
    return out

