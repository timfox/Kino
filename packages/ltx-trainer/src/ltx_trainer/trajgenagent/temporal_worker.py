"""Stage 2 temporal worker: kinematics travel + LLM duration (Eq. 9)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.llm_backend import estimate_duration_llm
from ltx_trainer.trajgenagent.profiles import IndividualProfile
from ltx_trainer.trajgenagent.trajectory import haversine_km


@dataclass(frozen=True)
class DurationEstimate:
    duration_minutes: float
    source: str


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def default_speed_kmh(prev_activity: str | None, activity: str) -> float:
    if prev_activity == "Home" and activity == "Work":
        return 25.0
    if activity in ("EatOut", "Shop", "Leisure"):
        return 18.0
    return 22.0


def estimate_travel_minutes(
    *,
    prev_lat: float,
    prev_lon: float,
    lat: float,
    lon: float,
    prev_activity: str | None,
    activity: str,
    profile: IndividualProfile | None = None,
    cfg: TrajGenAgentConfig | None = None,
) -> float:
    cfg = cfg or TrajGenAgentConfig()
    if not cfg.use_kinematics:
        return 15.0
    dist_km = haversine_km(prev_lat, prev_lon, lat, lon)
    tkey = f"{prev_activity}->{activity}"
    if profile and tkey in profile.transition_speed_kmh:
        speed = profile.transition_speed_kmh[tkey]
    else:
        speed = default_speed_kmh(prev_activity, activity)
    minutes = dist_km / max(speed, 1e-6) * 60.0
    return clip(minutes, cfg.travel_min_minutes, cfg.travel_max_minutes)


def estimate_duration(
    activity: str,
    *,
    start: datetime,
    remaining_activities: tuple[str, ...],
    profile: IndividualProfile | None = None,
    cfg: TrajGenAgentConfig | None = None,
) -> DurationEstimate:
    cfg = cfg or TrajGenAgentConfig()
    historical_mean = profile.duration_mean_min.get(activity) if profile else None
    llm_minutes = estimate_duration_llm(
        activity,
        remaining_activities=remaining_activities,
        profile=profile or IndividualProfile(individual_id="default"),
        cfg=cfg,
    )
    if llm_minutes is not None:
        return DurationEstimate(
            duration_minutes=clip(llm_minutes, cfg.duration_min_minutes, cfg.duration_max_minutes),
            source="vllm",
        )
    priors = {
        "Home": 480.0,
        "Work": 480.0,
        "EatOut": 45.0,
        "Shop": 60.0,
        "Leisure": 120.0,
        "Errand": 30.0,
        "Education": 180.0,
        "Healthcare": 90.0,
    }
    base = historical_mean if historical_mean is not None else priors.get(activity, 60.0)
    pressure = max(0.85, 1.0 - 0.05 * len(remaining_activities))
    minutes = clip(base * pressure, cfg.duration_min_minutes, cfg.duration_max_minutes)
    return DurationEstimate(duration_minutes=minutes, source="profile_prior")


def estimate_duration_llm_stub(
    activity: str,
    *,
    start: datetime,
    remaining_activities: tuple[str, ...],
    historical_mean_min: float | None = None,
    profile: IndividualProfile | None = None,
    cfg: TrajGenAgentConfig | None = None,
) -> DurationEstimate:
    _ = start, historical_mean_min
    return estimate_duration(
        activity,
        start=start,
        remaining_activities=remaining_activities,
        profile=profile,
        cfg=cfg,
    )


def cold_start_time(
    *,
    weekday: str = "Monday",
    profile: IndividualProfile | None = None,
    rng: random.Random | None = None,
) -> datetime:
    rng = rng or random.Random(0)
    if profile and weekday in profile.first_start_hour:
        hour = int(profile.first_start_hour[weekday])
        minute = int((profile.first_start_hour[weekday] - hour) * 60)
    else:
        hour = 8 if weekday not in ("Saturday", "Sunday") else 9
        minute = rng.randint(0, 30)
    return datetime(2024, 6, 10, hour, minute)
