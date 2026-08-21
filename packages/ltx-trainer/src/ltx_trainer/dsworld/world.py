"""Compiler / Simulator transitions and reflective optimization stub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.dsworld.state import DSState, RouteDecision, construct_state, route_action


@dataclass
class Transition:
    state: DSState
    action: str
    next_state: DSState
    mode: str
    execution_ok: bool = True
    error_type: str | None = None
    performance: float | None = None
    reflection: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action[:120],
            "mode": self.mode,
            "execution_ok": self.execution_ok,
            "error_type": self.error_type,
            "performance": self.performance,
            "next_progress": self.next_state.progress,
            "reflection": self.reflection[:200] if self.reflection else "",
        }


def _simulate_next(state: DSState, action: str) -> tuple[DSState, bool, str | None, float | None]:
    """LLM Simulator S(St, At) stub — predicts next state without real run."""
    a = action.lower()
    ok = "raise " not in a and "syntaxerror" not in a
    err = None if ok else "RuntimeError"
    perf = None
    if "fit(" in a or "logistic" in a or "lgb" in a:
        perf = 0.72 if ok else None
    nxt = construct_state(
        task=state.task,
        n_rows=state.data_stats.get("n_rows", 1000),
        n_cols=state.data_stats.get("n_cols", 10),
        libraries=list(state.env.get("libraries", [])),
        last_error=err,
    )
    nxt.progress = "simulated_ok" if ok else "simulated_fail"
    nxt.logs.append(f"sim:{'ok' if ok else err}")
    return nxt, ok, err, perf


def _compile_next(state: DSState, action: str) -> tuple[DSState, bool, str | None, float | None]:
    """Compiler C(St, At) stub — 'real' lightweight execution."""
    a = action.lower()
    ok = True
    err = None
    if "bad_column" in a:
        ok = False
        err = "KeyError"
    nxt = construct_state(
        task=state.task,
        n_rows=state.data_stats.get("n_rows", 1000),
        n_cols=state.data_stats.get("n_cols", 10),
        libraries=list(state.env.get("libraries", [])),
        last_error=err,
    )
    nxt.progress = "compiled_ok" if ok else "compiled_fail"
    nxt.logs.append(f"compile:{'ok' if ok else err}")
    return nxt, ok, err, None


def predict_transition(
    state: DSState,
    action: str,
    *,
    cost_threshold: float = 5.0,
    force_timeout: bool = False,
) -> Transition:
    """World model W(St, At) via hybrid execute/simulate (Eq. 4)."""
    decision = route_action(state, action, cost_threshold=cost_threshold, force_timeout=force_timeout)
    if decision.mode == "execute":
        nxt, ok, err, perf = _compile_next(state, action)
    else:
        nxt, ok, err, perf = _simulate_next(state, action)
    return Transition(
        state=state,
        action=action,
        next_state=nxt,
        mode=decision.mode,
        execution_ok=ok,
        error_type=err,
        performance=perf,
    )


@dataclass
class ReflectiveStep:
    """Reflective World Model Optimization: predict → reflect → refine."""

    original: Transition
    refined: Transition
    reward: float
    feedback: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original.as_dict(),
            "refined": self.refined.as_dict(),
            "reward": round(self.reward, 4),
            "feedback": self.feedback,
        }


def reflective_optimize(
    state: DSState,
    action: str,
    *,
    ground_truth_ok: bool = True,
    n_rollouts: int = 2,
) -> list[ReflectiveStep]:
    """Stub of reflective RL: compare prediction to GT, refine, score."""
    steps = []
    for _ in range(n_rollouts):
        pred = predict_transition(state, action, cost_threshold=0.0)  # force simulate path for RL
        # Force simulate for reflection demo when training-like
        pred_sim = predict_transition(state, action, cost_threshold=0.01)
        mismatch = pred_sim.execution_ok != ground_truth_ok
        feedback = "correct" if not mismatch else "execution_status_mismatch: refine"
        refined = Transition(
            state=state,
            action=action,
            next_state=pred_sim.next_state,
            mode="simulate",
            execution_ok=ground_truth_ok,
            error_type=None if ground_truth_ok else "RuntimeError",
            performance=pred_sim.performance,
            reflection=feedback,
        )
        reward = 1.0 if refined.execution_ok == ground_truth_ok else 0.0
        if pred_sim.performance is not None and ground_truth_ok:
            reward = 0.5 * reward + 0.5 * max(0.0, 1.0 - abs((pred_sim.performance or 0) - 0.72))
        steps.append(ReflectiveStep(original=pred_sim, refined=refined, reward=reward, feedback=feedback))
    return steps


def group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Eq. 10: Ai = (Ri − μ) / (σ + ε)."""
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = var**0.5
    return [(r - mu) / (sigma + eps) for r in rewards]
