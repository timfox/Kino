"""Competitive multi-adapter bank (Section 4.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.mari.config import MARIConfig
from ltx_trainer.mari.low_rank import LowRankAdapter, init_adapter
from ltx_trainer.mari.losses import direction_diversity_penalty, inter_adapter_overlap_penalty, mc_loss_from_scores
from ltx_trainer.mari.routing import inference_entropy_route, training_winner


@dataclass
class MultiAdapterBank:
    adapters: list[LowRankAdapter]
    gamma: float = 1.0

    @classmethod
    def create(cls, cfg: MARIConfig, seed: int = 0) -> MultiAdapterBank:
        rng = np.random.default_rng(seed)
        adapters = [init_adapter(cfg.hidden_dim, cfg.adapter_rank, rng) for _ in range(cfg.num_adapters)]
        return cls(adapters=adapters, gamma=cfg.global_scale_gamma)

    def option_logits(
        self,
        h: np.ndarray,
        option_embeddings: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> list[np.ndarray]:
        """Score each option via dot product after intervention (toy head)."""
        h = np.asarray(h, dtype=np.float64)
        opts = np.asarray(option_embeddings, dtype=np.float64)
        out: list[np.ndarray] = []
        for ad in self.adapters:
            h2 = ad.intervene(h, gamma=self.gamma, alpha=alpha)
            out.append(opts @ h2)
        return out

    def train_route_losses(
        self,
        h: np.ndarray,
        option_embeddings: np.ndarray,
        label: int,
        *,
        alpha: float = 1.0,
    ) -> tuple[int, list[float]]:
        logits = self.option_logits(h, option_embeddings, alpha=alpha)
        losses = [mc_loss_from_scores(z, label) for z in logits]
        return training_winner(losses), losses

    def infer_route(
        self,
        h: np.ndarray,
        option_embeddings: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> int:
        logits = self.option_logits(h, option_embeddings, alpha=alpha)
        return inference_entropy_route(logits)

    def diversity_loss(self, h: np.ndarray, *, alpha: float = 1.0) -> dict[str, float]:
        deltas = [self.gamma * a.scale * alpha * a.delta(h) for a in self.adapters]
        return {
            "inter_overlap": inter_adapter_overlap_penalty(self.adapters),
            "direction_cosine": direction_diversity_penalty(deltas),
        }


@dataclass
class CompetitiveTrainStep:
    winner: int
    losses: list[float]
    winner_loss: float

    def to_dict(self):
        return {
            "winner": self.winner,
            "losses": self.losses,
            "winner_loss": self.winner_loss,
        }


def competitive_train_step(
    bank: MultiAdapterBank,
    h: np.ndarray,
    option_embeddings: np.ndarray,
    label: int,
    *,
    alpha: float = 1.0,
) -> CompetitiveTrainStep:
    winner, losses = bank.train_route_losses(h, option_embeddings, label, alpha=alpha)
    return CompetitiveTrainStep(winner=winner, losses=losses, winner_loss=losses[winner])
