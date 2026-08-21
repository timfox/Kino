"""Bandwidth measurement samples (Eq. 1–2, §2.3)."""

from __future__ import annotations


def delivery_rates(
    *,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
    a2: float,
    a4: float,
) -> tuple[float, float]:
    """Equations (1) and (2): send_rate and ack_rate over [T1,T3] / [T2,T4]."""
    send_dt = t3 - t1
    ack_dt = t4 - t2
    if send_dt <= 0.0 or ack_dt <= 0.0:
        return 0.0, 0.0
    delta = a4 - a2
    return delta / send_dt, delta / ack_dt


def bandwidth_sample(
    *,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
    a2: float,
    a4: float,
    app_limited_in_window: bool,
) -> float | None:
    """Return min(send_rate, ack_rate) or None if sample is inaccurate."""
    if app_limited_in_window:
        return None
    send_rate, ack_rate = delivery_rates(t1=t1, t2=t2, t3=t3, t4=t4, a2=a2, a4=a4)
    if send_rate <= 0.0 or ack_rate <= 0.0:
        return None
    return min(send_rate, ack_rate)


def rmse(samples: list[float], true_bw_bps: list[float]) -> float:
    if not samples or len(samples) != len(true_bw_bps):
        return 0.0
    err = [(s - t) ** 2 for s, t in zip(samples, true_bw_bps, strict=True)]
    return float(sum(err) / len(err)) ** 0.5
