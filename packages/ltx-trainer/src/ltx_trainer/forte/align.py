"""Parameter-efficient audio projection and contrastive alignment (arXiv:2606.05812 §3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProjectionMLP:
    """Two-layer MLP with residual + LayerNorm (toy numpy stub)."""

    dim: int = 512
    hidden: int = 1024
    dropout: float = 0.1
    w1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    b1: np.ndarray = field(default_factory=lambda: np.zeros(1))
    w2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    b2: np.ndarray = field(default_factory=lambda: np.zeros(1))
    gamma: float = 0.07
    mu_logic: float = 0.1

    def __post_init__(self) -> None:
        if self.w1.shape == (1, 1):
            rng = np.random.default_rng(42)
            self.w1 = rng.standard_normal((self.hidden, self.dim)) * 0.02
            self.b1 = np.zeros(self.hidden)
            self.w2 = rng.standard_normal((self.dim, self.hidden)) * 0.02
            self.b2 = np.zeros(self.dim)

    @property
    def num_params(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1.T + self.b1)  # GELU approx with ReLU
        out = h @ self.w2.T + self.b2
        return x + out  # residual

    def project_batch(self, audio_emb: np.ndarray) -> np.ndarray:
        return np.stack([self.forward(a) for a in audio_emb], axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def infonce_loss(q_star: np.ndarray, audio_proj: np.ndarray, *, gamma: float = 0.07) -> float:
    """Symmetric InfoNCE stub (Eq. 5)."""
    n = len(audio_proj)
    losses = []
    for i in range(n):
        sims = np.array([cosine(q_star[i], audio_proj[j]) for j in range(n)]) / gamma
        sims = sims - sims.max()
        exp_s = np.exp(sims)
        losses.append(-np.log(exp_s[i] / exp_s.sum()))
    return float(np.mean(losses))


def logic_contrastive_loss(
    q_star: np.ndarray,
    q_neg_text_emb: np.ndarray,
    audio_proj: np.ndarray,
) -> float:
    """Eq. 6 stub: penalise audio closer to negative logical query."""
    losses = []
    for i in range(len(q_star)):
        pos = cosine(q_star[i], audio_proj[i])
        neg = cosine(q_neg_text_emb[i], audio_proj[i])
        losses.append(-np.log(1.0 / (1.0 + np.exp(-(pos - neg)))))
    return float(np.mean(losses))


def total_alignment_loss(
    q_star: np.ndarray,
    q_neg: np.ndarray,
    audio_proj: np.ndarray,
    module: ProjectionMLP,
) -> float:
    l_align = infonce_loss(q_star, audio_proj, gamma=module.gamma)
    l_logic = logic_contrastive_loss(q_star, q_neg, audio_proj)
    return l_align + module.mu_logic * l_logic


def train_step_stub(module: ProjectionMLP, batch_size: int = 8, dim: int = 512) -> dict[str, float]:
    """One synthetic training step (no autograd)."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((batch_size, dim))
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    a = rng.standard_normal((batch_size, dim))
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    q_neg = q + 0.1 * rng.standard_normal(q.shape)
    proj = module.project_batch(a)
    loss = total_alignment_loss(q, q_neg, proj, module)
    return {"loss": loss, "params_m": module.num_params / 1e6, "gamma": module.gamma}
