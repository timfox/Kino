"""GRPO trainer hook for TSG/MTR temporal rewards (MUSTBENCH Sec. 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.mustbench.config import MustBenchConfig
from ltx_trainer.mustbench.qa_samples import QASample, builtin_qa_samples
from ltx_trainer.mustbench.rewards import mtr_grpo_reward, tsg_grpo_reward


@dataclass
class GrpoSample:
    task: str
    pred: Any
    gold: Any
    audio_duration: float = 180.0


def group_advantages(rewards: list[float]) -> np.ndarray:
    """Group-relative baseline (GRPO): A_i = r_i - mean(r)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return r
    return r - r.mean()


def logprob_proxy(pred: float, gold: float, *, scale: float = 15.0) -> float:
    """Gaussian log-probability proxy for continuous timestamp predictions."""
    return float(-0.5 * ((pred - gold) / scale) ** 2 - np.log(scale * np.sqrt(2 * np.pi)))


def clip_surrogate(advantages: np.ndarray, logprobs: np.ndarray, *, clip_eps: float = 0.2) -> float:
    """PPO-style clipped surrogate on group-normalized advantages."""
    ratio = np.exp(logprobs - logprobs.mean())
    unclipped = advantages * ratio
    clipped = advantages * np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return float(-np.mean(np.minimum(unclipped, clipped)))


def sample_to_reward(sample: GrpoSample, *, cfg: MustBenchConfig | None = None) -> float:
    cfg = cfg or MustBenchConfig()
    if sample.task == "TSG":
        return tsg_grpo_reward(float(sample.pred), float(sample.gold), sample.audio_duration, cfg=cfg)
    if sample.task == "MTR":
        return mtr_grpo_reward(list(sample.pred), list(sample.gold), sample.audio_duration)
    return 0.0


def grpo_step(
    samples: list[GrpoSample],
    *,
    cfg: MustBenchConfig | None = None,
    clip_eps: float = 0.2,
) -> dict[str, Any]:
    """One GRPO update step over a sampled group."""
    cfg = cfg or MustBenchConfig()
    rewards = [sample_to_reward(s, cfg=cfg) for s in samples]
    adv = group_advantages(rewards)
    logprobs = np.array(
        [
            logprob_proxy(float(s.pred), float(s.gold), scale=cfg.grpo_tsg_scale_s)
            if s.task == "TSG"
            else float(sample_to_reward(s, cfg=cfg))
            for s in samples
        ],
        dtype=np.float64,
    )
    loss = clip_surrogate(adv, logprobs, clip_eps=clip_eps)
    return {
        "n_samples": len(samples),
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_advantage": float(np.mean(adv)),
        "grpo_loss": loss,
        "rewards": rewards,
    }


def grpo_step_from_qa(samples: list[QASample] | None = None, *, cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    raw = samples or builtin_qa_samples()
    group: list[GrpoSample] = []
    for s in raw:
        if s.pred is None or s.task not in {"TSG", "MTR"}:
            continue
        group.append(GrpoSample(task=s.task, pred=s.pred, gold=s.gold, audio_duration=180.0))
    return grpo_step(group, cfg=cfg)


@dataclass
class GrpoCheckpoint:
    step: int
    mean_reward: float
    policy_offset: float


def _synthetic_grpo_group(
    gold_t: float,
    *,
    policy_offset: float,
    rng: np.random.Generator,
    audio_duration: float = 180.0,
) -> list[GrpoSample]:
    """TSG group with policy-controlled timestamp predictions."""
    preds = [
        gold_t + policy_offset + rng.normal(0, 1.5),
        gold_t + policy_offset + rng.normal(0, 4.0),
        gold_t + policy_offset + rng.normal(0, 8.0),
    ]
    return [GrpoSample("TSG", float(p), gold_t, audio_duration) for p in preds]


def run_grpo_epochs(
    *,
    steps: int = 8,
    cfg: MustBenchConfig | None = None,
    seed: int = 42,
    lr: float = 0.15,
) -> dict[str, Any]:
    """Multi-step GRPO loop with a scalar policy offset checkpoint."""
    cfg = cfg or MustBenchConfig()
    rng = np.random.default_rng(seed)
    offset = float(rng.normal(0, 3.0))
    gold_t = 45.0
    history: list[dict[str, Any]] = []
    checkpoints: list[GrpoCheckpoint] = []

    for step in range(steps):
        group = _synthetic_grpo_group(gold_t, policy_offset=offset, rng=rng)
        out = grpo_step(group, cfg=cfg)
        offset -= lr * float(out["mean_advantage"])
        row = {"step": step, "mean_reward": out["mean_reward"], "offset": offset}
        history.append(row)
        checkpoints.append(GrpoCheckpoint(step=step, mean_reward=out["mean_reward"], policy_offset=offset))

    rewards = [h["mean_reward"] for h in history]
    return {
        "steps": steps,
        "initial_reward": rewards[0],
        "final_reward": rewards[-1],
        "reward_improved": rewards[-1] >= rewards[0],
        "checkpoints": [{"step": c.step, "mean_reward": c.mean_reward, "policy_offset": c.policy_offset} for c in checkpoints],
        "history": history,
    }


def grpo_trainer_smoke(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    out = grpo_step_from_qa(cfg=cfg)
    # Perturbed group should retain ordering signal
    worse = grpo_step(
        [
            GrpoSample("TSG", 90.0, 45.0, 180.0),
            GrpoSample("TSG", 46.0, 45.0, 180.0),
            GrpoSample("TSG", 45.0, 45.0, 180.0),
        ],
        cfg=cfg,
    )
    epochs = run_grpo_epochs(steps=6, cfg=cfg, seed=0)
    return {
        "grpo_ran": out["n_samples"] >= 2,
        "mean_reward_finite": np.isfinite(out["mean_reward"]),
        "grpo_loss_finite": np.isfinite(out["grpo_loss"]),
        "exact_beats_far": worse["rewards"][2] > worse["rewards"][0],
        "epochs_ran": epochs["steps"] >= 1,
        "reward_improved": epochs["reward_improved"],
        **{k: out[k] for k in ("n_samples", "mean_reward", "grpo_loss")},
    }
