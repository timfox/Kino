"""LoRA skill adapters: compile, mount, scale, compose (Eqs. 2–3, 6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ltx_trainer.latentskill.config import LatentSkillConfig


@dataclass
class LoRAUpdate:
    """Low-rank update for one linear module: W' = W + (alpha/r) B @ A."""

    a: np.ndarray  # (r, d_in)
    b: np.ndarray  # (d_out, r)
    module: str = "attn_o"
    layer: int = 0

    def delta(self, alpha: float, rank: int) -> np.ndarray:
        return (alpha / rank) * (self.b @ self.a)

    def frobenius_norm(self) -> float:
        return float(np.linalg.norm(self.delta(1.0, self.a.shape[0]), ord="fro"))


def compose_adapters(adapters: list[LatentSkillAdapter], alphas: list[float]) -> LatentSkillAdapter:
    """∆K = Σ α_k C[k] over aligned module/layer keys."""
    if not adapters:
        return LatentSkillAdapter(skill_id="composed", updates=[])
    rank = adapters[0].updates[0].a.shape[0] if adapters[0].updates else 64
    acc: dict[tuple[str, int], np.ndarray] = {}
    for adapter, alpha in zip(adapters, alphas, strict=False):
        for u in adapter.updates:
            key = (u.module, u.layer)
            delta = alpha * u.delta(alpha=1.0, rank=rank)
            acc[key] = acc.get(key, np.zeros_like(delta)) + delta
    merged: list[LoRAUpdate] = []
    for (mod, layer), delta in acc.items():
        u, s, vt = np.linalg.svd(delta, full_matrices=False)
        r = min(rank, s.shape[0])
        a = vt[:r]
        b = (u[:, :r] * s[:r]).astype(np.float64)
        merged.append(LoRAUpdate(a=a, b=b, module=mod, layer=layer))
    return LatentSkillAdapter(skill_id="composed", updates=merged)


@dataclass
class LatentSkillAdapter:
    """Collection of LoRA updates for one compiled skill."""

    skill_id: str
    updates: list[LoRAUpdate] = field(default_factory=list)
    text_digest: str = ""

    def compose_with(self, others: Iterable["LatentSkillAdapter"], alphas: Iterable[float]) -> "LatentSkillAdapter":
        """Parameter-space sum with self included first (Eq. 6)."""
        all_adapters = [self, *list(others)]
        all_alphas = [1.0, *list(alphas)]
        return compose_adapters(all_adapters, all_alphas)

    def frobenius_mean(self) -> float:
        if not self.updates:
            return 0.0
        return float(np.mean([u.frobenius_norm() for u in self.updates]))


def _seed_from_text(text: str) -> int:
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h)


def compile_skill_lora(
    skill_text: str,
    skill_id: str,
    cfg: LatentSkillConfig | None = None,
    *,
    modules: tuple[str, ...] | None = None,
    layers: tuple[int, ...] | None = None,
) -> LatentSkillAdapter:
    """Hypernetwork stub: deterministic low-rank weights from skill text hash."""
    cfg = cfg or LatentSkillConfig()
    rng = np.random.default_rng(_seed_from_text(skill_text))
    r = cfg.stub_lora_rank
    d = cfg.stub_hidden_dim
    mods = modules or cfg.stub_modules
    layer_ids = layers if layers is not None else cfg.stub_layer_range
    updates: list[LoRAUpdate] = []
    for layer in layer_ids:
        for mod in mods:
            a = rng.standard_normal((r, d)) * 0.02
            b = rng.standard_normal((d, r)) * 0.02
            updates.append(LoRAUpdate(a=a, b=b, module=mod, layer=layer))
    digest = hex(_seed_from_text(skill_text))[-8:]
    return LatentSkillAdapter(skill_id=skill_id, updates=updates, text_digest=digest)


def mount_delta(base_weight: np.ndarray, update: LoRAUpdate, *, alpha: float, rank: int) -> np.ndarray:
    """W' = W + α/r B A (Eq. 3)."""
    d_out, d_in = base_weight.shape
    delta = update.delta(alpha, rank)
    if delta.shape != base_weight.shape:
        delta = delta[:d_out, :d_in]
    return base_weight + delta


def stable_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    return float((s.sum() ** 2) / (s @ s))


def cumulative_energy(matrix: np.ndarray, k: int) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    total = float((s ** 2).sum())
    if total <= 0:
        return 0.0
    return float((s[:k] ** 2).sum() / total)
