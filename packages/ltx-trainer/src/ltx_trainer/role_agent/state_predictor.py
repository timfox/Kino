"""World-In-Agent state prediction (Eq. 2–4)."""

from __future__ import annotations

import re
from typing import Callable

from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.prompts import STATE_PREDICTION_TEMPLATE
from ltx_trainer.role_agent.wia import (
    StepRecord,
    compute_step_returns,
    predictive_reward_matrix,
    task_return,
)


PredictFn = Callable[[str, str, int], str]


def build_prediction_prompt(state: str, action: str, horizon: int) -> str:
    return STATE_PREDICTION_TEMPLATE.format(state=state, action=action, horizon=horizon)


def parse_predicted_state(text: str) -> str:
    text = text.strip()
    for tag in ("<pred>", "<state>", "<prediction>"):
        if tag in text:
            m = re.search(rf"{tag}\s*(.*?)\s*</", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return text.split("\n")[0].strip()


def carry_forward_predictor(_state: str, action: str, horizon: int) -> str:
    """Heuristic: echo action effect placeholder."""
    return f"After '{action}', environment advances {horizon} step(s)."


def oracle_predictor(future_states: list[str]) -> PredictFn:
    """Returns predictor that reads ground truth when available (eval / upper bound)."""

    def _predict(state: str, action: str, horizon: int) -> str:
        idx = None
        for i, s in enumerate(future_states):
            if s == state:
                idx = i
                break
        if idx is None:
            return carry_forward_predictor(state, action, horizon)
        target = idx + horizon
        if target < len(future_states):
            return future_states[target]
        return future_states[-1]

    return _predict


def collect_predictions(
    steps: list[StepRecord],
    *,
    horizon: int,
    predict_fn: PredictFn | None = None,
) -> dict[int, dict[int, str]]:
    """E_pre = {E_pre,t} with predictions for h=1..H."""
    fn = predict_fn or carry_forward_predictor
    out: dict[int, dict[int, str]] = {}
    states = [s.state for s in steps]
    for t, step in enumerate(steps):
        out[t] = {}
        for h in range(1, horizon + 1):
            out[t][h] = fn(step.state, step.action, h)
        if predict_fn is None and t + 1 < len(states):
            for h in range(1, min(horizon + 1, len(states) - t)):
                out[t][h] = states[t + h]
    return out


def wia_rewards_for_rollout(
    steps: list[StepRecord],
    predictions: dict[int, dict[int, str]],
    *,
    cfg: RoleAgentConfig,
    domain: str,
) -> tuple[list[list[float]], list[float]]:
    """Compute predictive matrix and per-step modulated returns."""
    if not cfg.enable_wia:
        rewards = [s.reward for s in steps]
        returns = [task_return(rewards, t, gamma=cfg.gamma) for t in range(len(steps))]
        return [], returns
    horizon = cfg.prediction_horizon(domain)
    for t in range(len(steps)):
        for h in range(1, horizon + 1):
            predictions.setdefault(t, {}).setdefault(
                h,
                carry_forward_predictor(steps[t].state, steps[t].action, h),
            )
    states = [s.state for s in steps]
    matrix = predictive_reward_matrix(predictions, states, horizon=horizon, gamma=cfg.gamma)
    returns = compute_step_returns(steps, predictions, horizon=horizon, gamma=cfg.gamma)
    return matrix, returns
