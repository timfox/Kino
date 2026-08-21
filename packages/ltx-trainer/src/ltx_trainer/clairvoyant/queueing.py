"""Queueing-theoretic helpers and discrete-event simulation (Sec. 2.4, 5.5)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ltx_trainer.clairvoyant.constants import UTIL_PEAK_RHO
from ltx_trainer.clairvoyant.scheduler import SJFScheduler


def fcfs_mean_waiting_time(rho: float, mean_service_s: float, c2_s: float) -> float:
    """Pollaczek-Khinchine mean queue wait (Eq. 1)."""
    if rho >= 1.0:
        return float("inf")
    return rho * mean_service_s * (1.0 + c2_s) / (2.0 * (1.0 - rho))


@dataclass
class SimRequest:
    req_id: int
    arrival: float
    service: float
    p_long: float
    is_short: bool


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(q * len(xs))))
    return xs[idx]


def discrete_event_simulation(
    *,
    n_requests: int = 2000,
    arrival_rate: float = 0.12,
    short_mean: float = 3.5,
    short_std: float = 0.8,
    long_mean: float = 8.9,
    long_std: float = 2.0,
    short_fraction: float = 0.5,
    starvation_tau_s: float = 10.5,
    policy: str = "sjf",  # sjf | fcfs
    seed: int = 42,
) -> dict[str, Any]:
    """
    Poisson arrivals with Gaussian service times (Sec. 5.5 Table 9 regime).
    """
    rng = random.Random(seed)
    requests: list[SimRequest] = []
    t = 0.0
    for i in range(n_requests):
        t += rng.expovariate(arrival_rate)
        is_short = rng.random() < short_fraction
        mu = short_mean if is_short else long_mean
        sigma = short_std if is_short else long_std
        service = max(0.5, rng.gauss(mu, sigma))
        p_long = rng.uniform(0.05, 0.25) if is_short else rng.uniform(0.65, 0.95)
        requests.append(SimRequest(i, t, service, p_long, is_short))

    sched = SJFScheduler(starvation_tau_s=starvation_tau_s)
    queue: list[SimRequest] = []
    idx = 0
    server_free = 0.0
    short_sojourn: list[float] = []
    long_sojourn: list[float] = []

    while idx < len(requests) or queue or sched:
        while idx < len(requests) and requests[idx].arrival <= server_free:
            r = requests[idx]
            if policy == "fcfs":
                queue.append(r)
            else:
                sched.enqueue(str(r.req_id), "", p_long=r.p_long, now=r.arrival)
            idx += 1

        if policy == "fcfs":
            if not queue:
                if idx < len(requests):
                    server_free = requests[idx].arrival
                continue
            r = queue.pop(0)
        else:
            nxt = sched.pop_next(now=server_free)
            if nxt is None:
                if idx < len(requests):
                    server_free = max(server_free, requests[idx].arrival)
                continue
            rid = int(nxt.request_id)
            r = next(x for x in requests if x.req_id == rid)

        start = max(server_free, r.arrival)
        finish = start + r.service
        sojourn = finish - r.arrival
        if r.is_short:
            short_sojourn.append(sojourn)
        else:
            long_sojourn.append(sojourn)
        server_free = finish

    mean_s = short_fraction * short_mean + (1 - short_fraction) * long_mean
    rho = arrival_rate * mean_s
    return {
        "policy": policy,
        "n_requests": n_requests,
        "rho": round(rho, 3),
        "starvation_tau_s": starvation_tau_s,
        "short_p50": round(_percentile(short_sojourn, 0.5), 2),
        "short_p95": round(_percentile(short_sojourn, 0.95), 2),
        "long_p50": round(_percentile(long_sojourn, 0.5), 2),
        "long_p95": round(_percentile(long_sojourn, 0.95), 2),
        "seed": seed,
    }


def tau_sensitivity_table(*, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    """Aggregate Table 9-style rows across seeds."""
    rows: list[dict[str, Any]] = []
    fcfs_runs = [discrete_event_simulation(policy="fcfs", seed=s) for s in seeds]
    fcfs = {
        "tau": "FCFS",
        "short_p50": sum(r["short_p50"] for r in fcfs_runs) / len(fcfs_runs),
        "short_p95": sum(r["short_p95"] for r in fcfs_runs) / len(fcfs_runs),
        "long_p50": sum(r["long_p50"] for r in fcfs_runs) / len(fcfs_runs),
        "long_p95": sum(r["long_p95"] for r in fcfs_runs) / len(fcfs_runs),
    }
    rows.append(fcfs)

    mu_short = 3.5
    for mult, label in ((1.0, "1.0×μ_short"), (3.0, "3.0× (default)"), (5.0, "5.0×"), (float("inf"), "∞ (pure SJF)")):
        tau = 1e9 if mult == float("inf") else mult * mu_short
        runs = [
            discrete_event_simulation(policy="sjf", starvation_tau_s=tau, seed=s)
            for s in seeds
        ]
        rows.append(
            {
                "tau": label,
                "short_p50": round(sum(r["short_p50"] for r in runs) / len(runs), 2),
                "short_p95": round(sum(r["short_p95"] for r in runs) / len(runs), 2),
                "long_p50": round(sum(r["long_p50"] for r in runs) / len(runs), 2),
                "long_p95": round(sum(r["long_p95"] for r in runs) / len(runs), 2),
            }
        )
    return rows


def utilisation_benefit_curve() -> list[dict[str, float]]:
    """Figure 3 anchor points: SJF short P50 reduction vs ρ."""
    return [
        {"rho": 0.45, "short_p50_reduction_pct": 2.0},
        {"rho": 0.55, "short_p50_reduction_pct": 8.0},
        {"rho": UTIL_PEAK_RHO, "short_p50_reduction_pct": 17.0},
        {"rho": 0.80, "short_p50_reduction_pct": 14.0},
        {"rho": 0.85, "short_p50_reduction_pct": 10.0},
    ]
