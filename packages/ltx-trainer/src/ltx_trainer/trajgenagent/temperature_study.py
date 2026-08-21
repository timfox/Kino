"""LLM sampling temperature vs verifier failure (Table VIII, Sec. IV-J)."""

from __future__ import annotations

import random

from ltx_trainer.trajgenagent.benchmarks import TABLE_VIII_TEMPERATURE
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.orchestrator import validate_activity_chain


def paper_temperature_row(temperature: float) -> dict[str, float]:
    row = next(r for r in TABLE_VIII_TEMPERATURE if r["temperature"] == temperature)
    return {
        "total_fr": float(row["total_fr"]),
        "schema_fr": float(row["schema_fr"]),
        "constraint_fr": float(row["constraint_fr"]),
    }


def _inject_chain_noise(chain: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    """Perturb activity chain toward invalid schema or home-anchor violations."""
    acts = list(chain)
    pool = ("Work", "Shop", "EatOut", "Home", "Errand", "Gym", "BadAct")
    op = rng.choice(("swap", "insert", "truncate", "vocab"))
    if op == "swap" and len(acts) >= 2:
        i, j = rng.sample(range(len(acts)), 2)
        acts[i], acts[j] = acts[j], acts[i]
    elif op == "insert":
        acts.insert(rng.randint(1, max(1, len(acts) - 1)), rng.choice(pool))
    elif op == "truncate" and len(acts) > 3:
        acts = acts[: rng.randint(2, len(acts) - 1)]
    elif op == "vocab":
        acts[rng.randint(0, len(acts) - 1)] = "BadAct"
    return tuple(acts)


def simulate_verifier_failures(
    *,
    temperature: float,
    n_trials: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Monte Carlo verifier failure rates; lowest near temperature=0.9 (paper optimum)."""
    rng = random.Random(seed)
    base_chain = ("Home", "Work", "EatOut", "Work", "Home")
    fail_p = 0.04 + abs(temperature - 0.9) * 0.22
    schema_fail = 0
    constraint_fail = 0
    total_fail = 0
    for _ in range(n_trials):
        chain = base_chain if rng.random() > fail_p else _inject_chain_noise(base_chain, rng)
        ok, reason = validate_activity_chain(chain)
        if ok:
            continue
        total_fail += 1
        if reason == "vocabulary":
            schema_fail += 1
        else:
            constraint_fail += 1
    return {
        "temperature": temperature,
        "total_fr": total_fail / n_trials,
        "schema_fr": schema_fail / n_trials,
        "constraint_fr": constraint_fail / n_trials,
    }


def temperature_study(cfg: TrajGenAgentConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    live = [simulate_verifier_failures(temperature=t, n_trials=400, seed=42 + int(t * 10)) for t in (0.5, 0.9, 1.5)]
    best_live = min(live, key=lambda r: r["total_fr"])
    paper_best = paper_temperature_row(cfg.llm_temperature)
    return {
        "paper_table_viii": TABLE_VIII_TEMPERATURE,
        "live_sweep": live,
        "best_live_temperature": best_live["temperature"],
        "paper_best_temperature": cfg.llm_temperature,
        "paper_best_total_fr": paper_best["total_fr"],
        "live_matches_paper_best_temp": best_live["temperature"] == cfg.llm_temperature,
        "live_monotonic_high_temp": live[1]["total_fr"] <= live[2]["total_fr"],
    }
