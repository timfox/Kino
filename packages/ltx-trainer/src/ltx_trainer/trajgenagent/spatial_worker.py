"""Stage 2 spatial worker: peer-augmented POI retrieval (Eq. 6–8)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from typing import Sequence

from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.peer_retrieval import augmented_poi_pool
from ltx_trainer.trajgenagent.profiles import IndividualProfile
from ltx_trainer.trajgenagent.trajectory import haversine_km, synthetic_poi_pool


@dataclass(frozen=True)
class PoiCandidate:
    poi_id: str
    lat: float
    lon: float
    individual_freq: float
    peer_freq: float


def build_candidate_pool(
    activity: str,
    *,
    individual_history: dict[str, list[str]] | None = None,
    peer_history: dict[str, list[str]] | None = None,
    poi_pool: dict[str, tuple[float, float]] | None = None,
) -> tuple[PoiCandidate, ...]:
    poi_pool = poi_pool or synthetic_poi_pool()
    individual_history = individual_history or {}
    peer_history = peer_history or {}
    ind_pois = individual_history.get(activity, list(poi_pool.keys())[:8])
    peer_pois = peer_history.get(activity, list(poi_pool.keys())[8:16])
    candidates: list[PoiCandidate] = []
    seen: set[str] = set()
    for poi_id in ind_pois + peer_pois:
        if poi_id in seen or poi_id not in poi_pool:
            continue
        seen.add(poi_id)
        lat, lon = poi_pool[poi_id]
        candidates.append(
            PoiCandidate(
                poi_id=poi_id,
                lat=lat,
                lon=lon,
                individual_freq=1.0 / max(1, len(ind_pois)),
                peer_freq=1.0 / max(1, len(peer_pois)),
            )
        )
    return tuple(candidates)


def build_candidate_pool_from_profile(
    activity: str,
    *,
    profile: IndividualProfile,
    peer_profiles: Sequence[IndividualProfile] = (),
    poi_pool: dict[str, tuple[float, float]] | None = None,
) -> tuple[PoiCandidate, ...]:
    poi_pool = poi_pool or synthetic_poi_pool()
    freq_map = augmented_poi_pool(activity, profile, list(peer_profiles))
    if not freq_map:
        return build_candidate_pool(activity, poi_pool=poi_pool)
    candidates: list[PoiCandidate] = []
    for poi_id, freq in freq_map.items():
        if poi_id not in poi_pool:
            continue
        lat, lon = poi_pool[poi_id]
        ind_freq = profile.poi_given_activity.get(activity, {}).get(poi_id, 0.0)
        peer_freq = max(0.0, freq - (1 - 0.15) * ind_freq) if ind_freq else freq
        candidates.append(
            PoiCandidate(
                poi_id=poi_id,
                lat=lat,
                lon=lon,
                individual_freq=ind_freq or freq,
                peer_freq=peer_freq or freq * 0.15,
            )
        )
    return tuple(candidates) if candidates else build_candidate_pool(activity, poi_pool=poi_pool)


def score_candidate(
    candidate: PoiCandidate,
    *,
    prev_lat: float | None,
    prev_lon: float | None,
    mean_transition_km: float,
    cfg: TrajGenAgentConfig | None = None,
) -> float:
    cfg = cfg or TrajGenAgentConfig()
    alpha = cfg.exploration_alpha
    s_freq = (1 - alpha) * candidate.individual_freq + alpha * candidate.peer_freq
    if prev_lat is None or prev_lon is None:
        s_dist = 1.0
    else:
        dist = haversine_km(prev_lat, prev_lon, candidate.lat, candidate.lon)
        s_dist = pow(2.718281828, -cfg.distance_beta * abs(dist - mean_transition_km))
    return cfg.score_lambda_freq * s_freq + cfg.score_lambda_dist * s_dist


def sample_poi(
    activity: str,
    *,
    prev_lat: float | None = None,
    prev_lon: float | None = None,
    prev_activity: str | None = None,
    profile: IndividualProfile | None = None,
    peer_profiles: Sequence[IndividualProfile] = (),
    rng: random.Random | None = None,
    cfg: TrajGenAgentConfig | None = None,
) -> PoiCandidate | None:
    cfg = cfg or TrajGenAgentConfig()
    rng = rng or random.Random(0)
    if profile is not None:
        pool = build_candidate_pool_from_profile(
            activity,
            profile=profile,
            peer_profiles=peer_profiles,
        )
    else:
        pool = build_candidate_pool(activity)
    if not pool:
        return None
    tkey = f"{prev_activity}->{activity}" if prev_activity else None
    if profile and tkey and tkey in profile.transition_distance_km:
        mean_km = profile.transition_distance_km[tkey]
    else:
        mean_km = 3.5 if prev_activity in (None, "Home") else 8.0
    scores = [score_candidate(c, prev_lat=prev_lat, prev_lon=prev_lon, mean_transition_km=mean_km, cfg=cfg) for c in pool]
    total = sum(scores) or 1.0
    probs = [s / total for s in scores]
    return rng.choices(pool, weights=probs, k=1)[0]
