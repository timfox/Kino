"""Discriminative vs generative readout for omni-modal MSA (arXiv:2606.05713)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ltx_trainer.omni_msa.metrics import parse_generative_score


class ReadoutMode(str, Enum):
    DISCRIMINATIVE = "discriminative"
    GENERATIVE_ZERO_SHOT = "generative_zero_shot"
    GENERATIVE_TRAINED = "generative_trained"


@dataclass
class RegressionHeadConfig:
    hidden_size: int = 3584
    head_hidden: int = 256
    dropout: float = 0.2


def last_non_pad_index(attention_mask: np.ndarray) -> int:
    """Index ℓ of last non-padding token (Eq. 2)."""
    mask = np.asarray(attention_mask, dtype=np.int64)
    if mask.ndim != 1:
        raise ValueError("attention_mask must be 1-D")
    return int(np.sum(mask) - 1)


def pool_last_non_pad(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Pool z = h_ℓ from [B, T, d] hidden states."""
    h = np.asarray(hidden_states, dtype=np.float64)
    if h.ndim == 2:
        idx = last_non_pad_index(attention_mask)
        return h[idx]
    if h.ndim != 3:
        raise ValueError("hidden_states must be [T,d] or [B,T,d]")
    idx = last_non_pad_index(attention_mask)
    return h[0, idx]


class DiscriminativeHead:
    """Lightweight MLP: Linear-LN-ReLU-Dropout-Linear (Eq. 3)."""

    def __init__(self, cfg: RegressionHeadConfig | None = None, *, seed: int = 0) -> None:
        cfg = cfg or RegressionHeadConfig()
        rng = np.random.default_rng(seed)
        d, h = cfg.hidden_size, cfg.head_hidden
        self.w1 = rng.normal(0, 0.02, size=(h, d))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0, 0.02, size=(1, h))
        self.b2 = np.zeros(1)
        self.dropout = cfg.dropout
        self._ln_gamma = np.ones(h)
        self._ln_beta = np.zeros(h)
        self._eps = 1e-5

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        var = np.var(x)
        x_hat = (x - mean) / np.sqrt(var + self._eps)
        return self._ln_gamma * x_hat + self._ln_beta

    def forward(self, z: np.ndarray, *, train: bool = False) -> float:
        x = np.asarray(z, dtype=np.float64).reshape(-1)
        h = self.w1 @ x + self.b1
        h = self._layer_norm(h)
        h = np.maximum(h, 0.0)
        if train and self.dropout > 0:
            pass  # smoke stub — no stochastic dropout
        y = float((self.w2 @ h + self.b2).squeeze())
        return y

    def predict_batch(
        self,
        hidden_states: np.ndarray,
        attention_masks: np.ndarray,
        *,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Denormalized sentiment predictions."""
        hs = np.asarray(hidden_states)
        masks = np.asarray(attention_masks)
        if hs.ndim == 2:
            hs = hs[None, ...]
            masks = masks[None, ...]
        out = []
        for i in range(hs.shape[0]):
            z = pool_last_non_pad(hs[i], masks[i])
            y_norm = self.forward(z)
            out.append(y_norm * sigma + mu)
        return np.clip(np.asarray(out), -3.0, 3.0)


def generative_decode(
    logits_stub: np.ndarray,
    *,
    mode: ReadoutMode,
    seed: int = 0,
) -> str:
    """Toy greedy decode of numeric sentiment string."""
    rng = np.random.default_rng(seed)
    if mode == ReadoutMode.GENERATIVE_ZERO_SHOT:
        choices = ["1.5", "positive", "maybe", "0.0", "neutral sentiment"]
        return str(rng.choice(choices))
    if mode == ReadoutMode.GENERATIVE_TRAINED:
        choices = ["1.50", "-0.75", "2.00", "0.00", "not a number"]
        return str(rng.choice(choices))
    return "0.00"


def generative_predict(
    decode_outputs: Sequence[str] | list[str],
    *,
    clip: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    from ltx_trainer.omni_msa.metrics import out_of_range_rate, unparsable_rate

    parsed: list[float] = []
    for s in decode_outputs:
        val = parse_generative_score(s)
        if val is None:
            parsed.append(0.0)
        elif clip:
            parsed.append(float(np.clip(val, -3.0, 3.0)))
        else:
            parsed.append(val)
    reliability = {
        "unparsable_pct": unparsable_rate(decode_outputs),
        "oob_pct": out_of_range_rate(decode_outputs),
    }
    return np.asarray(parsed), reliability
