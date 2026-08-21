"""Individual evidence profile Πu (Eq. 4) and chain-only priors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ltx_trainer.trajgenagent.trajectory import DailyTrajectory, haversine_km


@dataclass
class IndividualProfile:
    individual_id: str
    activity_freq: dict[str, float] = field(default_factory=dict)
    transition_freq: dict[str, float] = field(default_factory=dict)
    poi_given_activity: dict[str, dict[str, float]] = field(default_factory=dict)
    duration_mean_min: dict[str, float] = field(default_factory=dict)
    transition_distance_km: dict[str, float] = field(default_factory=dict)
    transition_speed_kmh: dict[str, float] = field(default_factory=dict)
    first_start_hour: dict[str, float] = field(default_factory=dict)
    exemplar_chains: tuple[tuple[str, ...], ...] = ()


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values()) or 1.0
    return {k: v / total for k, v in counts.items()}


def build_profile_from_activity_chains(
    individual_id: str,
    chains: Sequence[Sequence[str]],
) -> IndividualProfile:
    act_counts: dict[str, float] = {}
    trans_counts: dict[str, float] = {}
    for chain in chains:
        for a in chain:
            act_counts[a] = act_counts.get(a, 0.0) + 1.0
        for i in range(len(chain) - 1):
            key = f"{chain[i]}->{chain[i + 1]}"
            trans_counts[key] = trans_counts.get(key, 0.0) + 1.0
    return IndividualProfile(
        individual_id=individual_id,
        activity_freq=_normalize(act_counts),
        transition_freq=_normalize(trans_counts),
        exemplar_chains=tuple(tuple(c) for c in chains[:3]),
    )


def build_profile_from_trajectories(
    individual_id: str,
    trajectories: Sequence[DailyTrajectory],
) -> IndividualProfile:
    act_counts: dict[str, float] = {}
    trans_counts: dict[str, float] = {}
    poi_counts: dict[str, dict[str, float]] = {}
    dur_sums: dict[str, float] = {}
    dur_counts: dict[str, float] = {}
    dist_sums: dict[str, float] = {}
    dist_counts: dict[str, float] = {}
    speed_sums: dict[str, float] = {}
    speed_counts: dict[str, float] = {}
    start_hours: dict[str, list[float]] = {}
    chains: list[tuple[str, ...]] = []

    for traj in trajectories:
        chains.append(tuple(v.activity for v in traj.visits))
        wkey = traj.weekday
        if traj.visits:
            start_hours.setdefault(wkey, []).append(traj.visits[0].start.hour + traj.visits[0].start.minute / 60.0)
        for i, visit in enumerate(traj.visits):
            act = visit.activity
            act_counts[act] = act_counts.get(act, 0.0) + 1.0
            poi_counts.setdefault(act, {})
            poi_counts[act][visit.poi_id] = poi_counts[act].get(visit.poi_id, 0.0) + 1.0
            dur_sums[act] = dur_sums.get(act, 0.0) + visit.duration_minutes
            dur_counts[act] = dur_counts.get(act, 0.0) + 1.0
            if i > 0:
                prev = traj.visits[i - 1]
                tkey = f"{prev.activity}->{act}"
                trans_counts[tkey] = trans_counts.get(tkey, 0.0) + 1.0
                dist = haversine_km(prev.lat, prev.lon, visit.lat, visit.lon)
                gap_h = max((visit.start - prev.end).total_seconds() / 3600.0, 1e-6)
                speed = dist / gap_h
                dist_sums[tkey] = dist_sums.get(tkey, 0.0) + dist
                dist_counts[tkey] = dist_counts.get(tkey, 0.0) + 1.0
                speed_sums[tkey] = speed_sums.get(tkey, 0.0) + speed
                speed_counts[tkey] = speed_counts.get(tkey, 0.0) + 1.0

    return IndividualProfile(
        individual_id=individual_id,
        activity_freq=_normalize(act_counts),
        transition_freq=_normalize(trans_counts),
        poi_given_activity={a: _normalize(c) for a, c in poi_counts.items()},
        duration_mean_min={a: dur_sums[a] / dur_counts[a] for a in dur_sums},
        transition_distance_km={k: dist_sums[k] / dist_counts[k] for k in dist_sums},
        transition_speed_kmh={k: speed_sums[k] / speed_counts[k] for k in speed_sums},
        first_start_hour={w: sum(hs) / len(hs) for w, hs in start_hours.items() if hs},
        exemplar_chains=tuple(chains[:3]),
    )


def profile_summary(profile: IndividualProfile) -> dict[str, object]:
    return {
        "individual_id": profile.individual_id,
        "top_activities": sorted(profile.activity_freq.items(), key=lambda x: -x[1])[:5],
        "n_transitions": len(profile.transition_freq),
        "n_poi_activities": len(profile.poi_given_activity),
        "duration_mean_min": profile.duration_mean_min,
        "first_start_hour": profile.first_start_hour,
        "n_exemplars": len(profile.exemplar_chains),
    }
