"""DSWorld evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dsworld.config import DSWorldConfig
from ltx_trainer.dsworld.state import construct_state, estimate_action_cost, route_action
from ltx_trainer.dsworld.world import (
    group_relative_advantages,
    predict_transition,
    reflective_optimize,
)


def run_routing_demo(config: DSWorldConfig | None = None) -> dict[str, Any]:
    cfg = config or DSWorldConfig()
    state = construct_state(task="Predict employee access needs", n_rows=5000, n_cols=12)
    light = "df = pd.read_csv('train.csv'); print(df.head())"
    heavy = "model = lgb.LGBMClassifier(); model.fit(X_train, y_train)"
    return {
        "light": route_action(state, light, cost_threshold=cfg.simulate_cost_threshold).as_dict(),
        "heavy": route_action(state, heavy, cost_threshold=cfg.simulate_cost_threshold).as_dict(),
        "timeout": route_action(state, light, force_timeout=True).as_dict(),
        "light_cost": estimate_action_cost(light),
        "heavy_cost": estimate_action_cost(heavy),
    }


def run_transition_demo() -> dict[str, Any]:
    state = construct_state(task="Sentiment classification", n_rows=10000, n_cols=3)
    light_t = predict_transition(state, "print(df.describe())", cost_threshold=5.0)
    heavy_t = predict_transition(
        state,
        "vectorizer.fit_transform(X); model.fit(X_train, y_train)",
        cost_threshold=5.0,
    )
    return {
        "light": light_t.as_dict(),
        "heavy": heavy_t.as_dict(),
        "routes_differ": light_t.mode != heavy_t.mode,
    }


def run_reflective_demo() -> dict[str, Any]:
    state = construct_state(task="Tabular ML", n_rows=2000)
    steps = reflective_optimize(state, "model.fit(X, y)", ground_truth_ok=True, n_rollouts=4)
    rewards = [s.reward for s in steps]
    adv = group_relative_advantages(rewards)
    return {
        "n_steps": len(steps),
        "mean_reward": sum(rewards) / len(rewards),
        "advantages_sum": sum(adv),
        "steps": [s.as_dict() for s in steps[:2]],
    }


def evaluation_demo() -> dict[str, Any]:
    routing = run_routing_demo()
    trans = run_transition_demo()
    refl = run_reflective_demo()
    return {
        "routing": routing,
        "transitions": trans,
        "reflective": refl,
        "routes_heavy_to_simulate": routing["heavy"]["mode"] == "simulate",
        "routes_light_to_execute": routing["light"]["mode"] == "execute",
        "timeout_to_simulate": routing["timeout"]["mode"] == "simulate",
        "hybrid_routes_differ": trans["routes_differ"],
        "reflective_improves": refl["mean_reward"] > 0.4,
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.dsworld.benchmarks import PAPER_ANCHORS, TABLE_1_AVG

    demo = evaluation_demo()
    checks = {
        "routes_heavy_to_simulate": demo["routes_heavy_to_simulate"],
        "routes_light_to_execute": demo["routes_light_to_execute"],
        "timeout_to_simulate": demo["timeout_to_simulate"],
        "hybrid_routes_differ": demo["hybrid_routes_differ"],
        "reflective_improves": demo["reflective_improves"],
        "paper_avg": abs(float(TABLE_1_AVG["DSWorld"]) - 0.781) < 1e-6,
        "paper_speedup_rl": abs(float(PAPER_ANCHORS["rl_speedup_x"]) - 14.0) < 1e-6,
        "paper_gain_pct": abs(float(PAPER_ANCHORS["transition_gain_vs_o4mini"]) - 0.356) < 1e-6,
        "paper_n_train": int(PAPER_ANCHORS["n_train_trajectories"]) == 8000,
        "paper_esp": abs(float(PAPER_ANCHORS["esp"]) - 0.950) < 1e-6,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()
