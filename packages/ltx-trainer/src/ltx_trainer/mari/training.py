"""Competitive multi-adapter training loop (Section 4.2, Eq. 7–8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.mari.adapters import MultiAdapterBank, competitive_train_step
from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.losses import direction_diversity_penalty, inter_adapter_overlap_penalty
from ltx_trainer.mari.routing import usage_balance_penalty


@dataclass
class CompetitiveTrainReport:
    n_steps: int
    mean_winner_loss: float
    usage_fractions: list[float]
    usage_balance: float
    inter_overlap: float
    direction_diversity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "mean_winner_loss": round(self.mean_winner_loss, 6),
            "usage_fractions": [round(x, 4) for x in self.usage_fractions],
            "usage_balance": round(self.usage_balance, 6),
            "inter_overlap": round(self.inter_overlap, 6),
            "direction_diversity": round(self.direction_diversity, 6),
        }


def _finite_diff_adapter_step(
    bank: MultiAdapterBank,
    adapter_idx: int,
    h: np.ndarray,
    option_embeddings: np.ndarray,
    label: int,
    *,
    lr: float,
    eps: float = 1e-4,
) -> float:
    """One SGD-like step on winner adapter U,V via coordinate-wise finite differences."""
    ad = bank.adapters[adapter_idx]

    def loss_at(adapt) -> float:
        logits = [np.asarray(option_embeddings @ adapt.intervene(h, gamma=bank.gamma), dtype=np.float64)]
        from ltx_trainer.mari.losses import mc_loss_from_scores

        return mc_loss_from_scores(logits[0], label)

    base = loss_at(ad)
    for arr_name in ("U", "V"):
        arr = getattr(ad, arr_name)
        flat = arr.ravel()
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + eps
            g = (loss_at(ad) - base) / eps
            flat[j] = old - lr * g
    return float(loss_at(ad))


def train_competitive_epoch(
    bank: MultiAdapterBank,
    batch: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    lr: float = 1e-3,
    lambda_balance: float = 0.01,
    lambda_overlap: float = 0.01,
    lambda_direction: float = 0.01,
) -> CompetitiveTrainReport:
    """
    One epoch of winner-take-gradient updates (Eq. 8).

    batch: list of (hidden, option_embeddings, label).
    """
    winners: list[int] = []
    losses: list[float] = []
    h0 = batch[0][0] if batch else np.zeros(1)

    for h, opts, label in batch:
        step = competitive_train_step(bank, h, opts, label)
        winners.append(step.winner)
        losses.append(step.winner_loss)
        _finite_diff_adapter_step(bank, step.winner, h, opts, label, lr=lr)

    k = len(bank.adapters)
    usage = np.bincount(winners, minlength=k).astype(float)
    usage /= max(usage.sum(), 1.0)
    deltas = [bank.gamma * a.scale * a.delta(h0) for a in bank.adapters]

    _ = lambda_balance, lambda_overlap, lambda_direction  # reserved for full optimizer

    return CompetitiveTrainReport(
        n_steps=len(batch),
        mean_winner_loss=float(np.mean(losses)) if losses else 0.0,
        usage_fractions=usage.tolist(),
        usage_balance=usage_balance_penalty(usage),
        inter_overlap=inter_adapter_overlap_penalty(bank.adapters),
        direction_diversity=direction_diversity_penalty(deltas),
    )


def demo_train(cfg: MARIConfig | None = None, *, seed: int = 0, n: int = 12) -> dict[str, Any]:
    from ltx_trainer.mari.pipeline import synthetic_corpus

    cfg = cfg or MARIConfig(hidden_dim=32, num_adapters=3, adapter_rank=4)
    rng = np.random.default_rng(seed)
    bank = MultiAdapterBank.create(cfg, seed=seed)
    batch: list[tuple[np.ndarray, np.ndarray, int]] = []
    for ex in synthetic_corpus(cfg, seed=seed, n=n):
        opts = rng.standard_normal((4, cfg.hidden_dim))
        opts /= np.linalg.norm(opts, axis=1, keepdims=True) + 1e-9
        opts[ex.label] *= 1.2
        batch.append((ex.hidden, opts, ex.label))
    before = train_competitive_epoch(bank, batch, lr=0.0)
    after = train_competitive_epoch(bank, batch, lr=5e-4)
    return {"before": before.to_dict(), "after": after.to_dict()}
